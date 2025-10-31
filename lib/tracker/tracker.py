# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2

import lib.common.camera as camera
import numpy as np
import torch
from lib.common.hand import HandModel, NUM_HANDS, scaled_hand_model
from lib.data_utils import bundles
from lib.models.regressor import RegressorOutput
from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData

from .perspective_crop import gen_crop_cameras_from_pose
from .tracking_result import SingleHandPose, TrackingResult

logger = logging.getLogger(__name__)

MM_TO_M = 0.001
M_TO_MM = 1000.0
MIN_OBSERVED_LANDMARKS = 21
CONFIDENCE_THRESHOLD = 0.5
MAX_VIEW_NUM = 2


@dataclass
class ViewData:
    image: np.ndarray
    camera: camera.CameraModel
    camera_angle: float


@dataclass
class InputFrame:
    views: List[ViewData]


@dataclass
class HandTrackerOpts:
    num_crop_points: int = 63
    enable_memory: bool = True
    use_stored_pose_for_crop: bool = True
    hand_ratio_in_crop: float = 0.95
    min_required_vis_landmarks: int = 19
    max_view_num: int = 2  # Maximum number of camera views to use (1=single-view, 2=multi-view)
    min_required_vis_landmarks_single_view: int = 10  # Relaxed threshold for single-view mode


def _warp_image(
    src_camera: camera.CameraModel,
    dst_camera: camera.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


class HandTracker:
    def __init__(self, model, opts: HandTrackerOpts) -> None:
        self._device: str = "cuda" if torch.cuda.device_count() else "cpu"
        logger.info(f"Using device: {self._device}")

        self._model = model
        self._model.to(self._device)

        self._input_size = np.array(self._model.getInputImageSizes())
        self._num_crop_points = opts.num_crop_points
        self._enable_memory = opts.enable_memory
        self._hand_ratio_in_crop: float = opts.hand_ratio_in_crop
        self._min_required_vis_landmarks: int = opts.min_required_vis_landmarks
        self._max_view_num: int = opts.max_view_num
        self._min_required_vis_landmarks_single_view: int = opts.min_required_vis_landmarks_single_view
        self._valid_tracking_history = np.zeros(2, dtype=bool)
        self._use_stored_pose_for_crop: bool = opts.use_stored_pose_for_crop
        self._last_tracking_result: Optional[TrackingResult] = None

    def reset_history(self) -> None:
        self._valid_tracking_history[:] = False
        self._last_tracking_result = None
    
    def _generate_default_poses(self) -> Dict[int, SingleHandPose]:
        """
        Generate default hand poses for bootstrapping tracking when no ground truth
        or previous tracking is available. Places hands at reasonable default positions.
        """
        from lib.common.hand import NUM_JOINTS_PER_HAND
        
        default_poses = {}
        for hand_idx in range(NUM_HANDS):
            # Place hand at center of scene, slightly forward
            default_wrist_xform = np.eye(4, dtype=np.float32)
            default_wrist_xform[2, 3] = 500.0  # 500mm forward from camera
            if hand_idx == 1:  # Right hand
                default_wrist_xform[0, 3] = 100.0  # 100mm to the right
            else:  # Left hand
                default_wrist_xform[0, 3] = -100.0  # 100mm to the left
            
            default_pose = SingleHandPose(
                joint_angles=np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32),  # Neutral pose
                wrist_xform=default_wrist_xform,
                hand_confidence=0.8  # High enough to pass threshold
            )
            default_poses[hand_idx] = default_pose
        
        return default_poses

    def gen_crop_cameras(
        self,
        cameras: List[camera.CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Optional[Dict[int, SingleHandPose]],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, camera.PinholePlaneCameraModel]]:
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]] = {}
        
        # Determine which poses to use for generating crop cameras
        poses_to_use = {}
        
        # First, try to use ground truth tracking if available
        if gt_tracking:
            poses_to_use = gt_tracking
        # If no ground truth and we should use stored poses, use previous tracking result
        elif self._use_stored_pose_for_crop and self._last_tracking_result:
            poses_to_use = self._last_tracking_result.hand_poses
            if poses_to_use:
                logger.debug(f"Using stored pose from previous frame for crop camera generation (hands: {list(poses_to_use.keys())})")
        
        # Bootstrap: If no poses available and we're in inference mode, try default pose
        if not poses_to_use and self._use_stored_pose_for_crop:
            logger.info("No ground truth or previous tracking available. Attempting bootstrap with default hand pose.")
            poses_to_use = self._generate_default_poses()
        
        # If still no poses available, return empty crop cameras
        if not poses_to_use:
            return crop_cameras

        for hand_idx, hand_pose in poses_to_use.items():
            if hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                continue
            # Use relaxed visibility threshold for single-view mode
            min_vis_landmarks = self._min_required_vis_landmarks_single_view if self._max_view_num == 1 else self._min_required_vis_landmarks
            crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
                cameras,
                camera_angles,
                hand_model,
                hand_pose,
                hand_idx,
                self._num_crop_points,
                self._input_size,
                max_view_num=self._max_view_num,
                sort_camera_index=True,
                focal_multiplier=self._hand_ratio_in_crop,
                mirror_right_hand=True,
                min_required_vis_landmarks=min_vis_landmarks,
            )

        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]
 
        return crop_cameras

    def track_frame(
        self,
        sample: InputFrame,
        hand_model: HandModel,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            result = TrackingResult()
            # Don't store empty results - they won't help bootstrap
            return result

        frame_data, frame_desc, skeleton_data = self._make_inputs(
            sample, hand_model, crop_cameras
        )
        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_use_skeleton(
                    frame_data, frame_desc, skeleton_data
                ),
                torch.device("cpu"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )
        
        # Store tracking result for use in next frame if enabled
        # Only store if we actually have hand poses (not empty)
        if self._use_stored_pose_for_crop and tracking_result.hand_poses:
            self._last_tracking_result = tracking_result
            logger.debug(f"Stored tracking result for {len(tracking_result.hand_poses)} hand(s) for next frame")
        
        return tracking_result

    def track_frame_and_calibrate_scale(
        self,
        sample: InputFrame,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            result = TrackingResult()
            # Don't store empty results - they won't help bootstrap
            return result
        frame_data, frame_desc, _ = self._make_inputs(sample, None, crop_cameras)

        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_pred_skel_scale(frame_data, frame_desc),
                torch.device("cpu"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )
        
        # Store tracking result for use in next frame if enabled
        # Only store if we actually have hand poses (not empty)
        if self._use_stored_pose_for_crop and tracking_result.hand_poses:
            self._last_tracking_result = tracking_result
            logger.debug(f"Stored tracking result for {len(tracking_result.hand_poses)} hand(s) for next frame")
        
        return tracking_result

    def _make_inputs(
        self,
        sample: InputFrame,
        hand_model_mm: Optional[HandModel],
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ):
        image_idx = 0
        left_images = []
        intrinsics = []
        extrinsics_xf = []
        sample_range_n_hands = []
        hand_indices = []
        for hand_idx, crop_camera_info in crop_cameras.items():
            sample_range_start = image_idx
            for cam_idx, crop_camera in crop_camera_info.items():
                view_data = sample.views[cam_idx]
                crop_image = _warp_image(view_data.camera, crop_camera, view_data.image)
                left_images.append(crop_image.astype(np.float32) / 255.0)
                intrinsics.append(crop_camera.uv_to_window_matrix())

                crop_world_to_eye_xf = np.linalg.inv(crop_camera.camera_to_world_xf)
                crop_world_to_eye_xf[:3, 3] *= MM_TO_M
                extrinsics_xf.append(crop_world_to_eye_xf)

                image_idx += 1

            if image_idx > sample_range_start:
                hand_indices.append(hand_idx)
                sample_range_n_hands.append(np.array([sample_range_start, image_idx]))

        hand_indices = np.array(hand_indices)
        frame_data = InputFrameData(
            left_images=torch.from_numpy(np.stack(left_images)).float(),
            intrinsics=torch.from_numpy(np.stack(intrinsics)).float(),
            extrinsics_xf=torch.from_numpy(np.stack(extrinsics_xf)).float(),
        )
        frame_desc = InputFrameDesc(
            sample_range=torch.from_numpy(np.stack(sample_range_n_hands)).long(),
            memory_idx=torch.from_numpy(hand_indices).long(),
            # use memory if the hand is previously valid
            use_memory=torch.from_numpy(
                self._valid_tracking_history[hand_indices]
            ).bool(),
            hand_idx=torch.from_numpy(hand_indices).long(),
        )
        skeleton_data = None
        if hand_model_mm is not None:
            # m -> mm
            hand_model_m = scaled_hand_model(hand_model_mm, MM_TO_M)
            skeleton_data = InputSkeletonData(
                joint_rotation_axes=hand_model_m.joint_rotation_axes.float(),
                joint_rest_positions=hand_model_m.joint_rest_positions.float(),
            )
        return bundles.to_device((frame_data, frame_desc, skeleton_data), self._device)

    def _gen_tracking_result(
        self,
        regressor_output: RegressorOutput,
        hand_indices: np.ndarray,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:

        output_joint_angles = regressor_output.joint_angles.to("cpu").numpy()
        output_wrist_xforms = regressor_output.wrist_xfs.to("cpu").numpy()
        output_wrist_xforms[..., :3, 3] *= M_TO_MM
        output_scales = None
        if regressor_output.skel_scales is not None:
            output_scales = regressor_output.skel_scales.to("cpu").numpy()

        hand_poses = {}
        num_views = {}
        predicted_scales = {}

        for output_idx, hand_idx in enumerate(hand_indices):
            raw_handpose = SingleHandPose(
                joint_angles=output_joint_angles[output_idx],
                wrist_xform=output_wrist_xforms[output_idx],
                hand_confidence=1.0,
            )
            hand_poses[hand_idx] = raw_handpose
            num_views[hand_idx] = len(crop_cameras[hand_idx])
            if output_scales is not None:
                predicted_scales[hand_idx] = output_scales[output_idx]

        for hand_idx in range(NUM_HANDS):
            hand_valid = False
            if hand_idx in hand_poses:
                self._valid_tracking_history[hand_idx] = True
                hand_valid = True
            if hand_valid:
                continue
            self._valid_tracking_history[hand_idx] = False
        
        return TrackingResult(
            hand_poses=hand_poses,
            num_views=num_views,
            predicted_scales=predicted_scales,
        )
