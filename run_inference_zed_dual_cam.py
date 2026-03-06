# Copyright (c) Chuer Pan
"""
Run UmeTrack hand pose estimation on two synchronized ZED stereo cameras.
Each ZED provides a stereo pair (left + right), so 4 images total per frame.

The ArUco calibration JSON provides the world-space extrinsics for each camera
(same format used by viser_dual_cam_pcd_world_foundation_stereo.py).
Per-camera stereo intrinsics are read directly from the SVO2 files via ZED SDK.

Usage:
    python run_inference_zed_dual_cam.py \
        --svo1 ~/Data/ZedData/20260305/ZED-2i_SN37220564_2026-03-05_13-33-35.svo2 \
        --svo2 ~/Data/ZedData/20260305/ZED-2i_SN38334762_2026-03-05_13-33-35.svo2 \
        --calib ~/Data/ZedData/20260305/calibration_aruco.json \
        --model-path pretrained_models/pretrained_weights.torch \
        --output-dir zed_pred
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import lib.data_utils.fs as fs
from lib.common.camera import PinholePlaneCameraModel, NoDistortion
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND, HandModel, scaled_hand_model
from lib.models.model_loader import load_pretrained_model
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import _load_json, load_hand_model_from_dict

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed not available. Install the ZED SDK Python bindings.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MM_TO_M = 0.001

# ---------------------------------------------------------------------------
# ArUco calibration loader (mirrors viser_dual_cam_pcd_world_foundation_stereo.py)
# ---------------------------------------------------------------------------

def load_aruco_extrinsics(calib_path: Path) -> Dict[str, np.ndarray]:
    """
    Load per-camera world-space 4×4 transforms from an ArUco calibration JSON.
    Keys are camera serial numbers (strings). The matrix maps camera→world.
    """
    with open(calib_path) as f:
        data = json.load(f)
    result = {}
    for serial, entry in data.items():
        if not isinstance(entry, dict) or "FusionConfiguration" not in entry:
            continue
        flat = list(map(float, entry["FusionConfiguration"]["pose"].split()))
        result[serial] = np.array(flat, dtype=np.float64).reshape(4, 4)
    return result


# ---------------------------------------------------------------------------
# ZED helpers
# ---------------------------------------------------------------------------

def open_svo(path: Path) -> sl.Camera:
    cam = sl.Camera()
    p = sl.InitParameters()
    p.set_from_svo_file(str(path))
    p.svo_real_time_mode = False
    p.coordinate_units = sl.UNIT.METER
    p.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    p.depth_mode = sl.DEPTH_MODE.NONE  # No depth needed; only images
    err = cam.open(p)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Cannot open {path}: {err}")
    sn = cam.get_camera_information().serial_number
    n = cam.get_svo_number_of_frames()
    logger.info(f"Opened SN{sn}  ({n} frames)  [{path.name}]")
    return cam


def get_zed_stereo_transforms(cam: sl.Camera) -> Tuple[
    PinholePlaneCameraModel, PinholePlaneCameraModel, np.ndarray
]:
    """
    Return (left_camera_model, right_camera_model, T_right_in_left).

    T_right_in_left is the 4×4 homogeneous transform of the right camera
    expressed in the left camera frame (camera→parent convention).
    """
    info = cam.get_camera_information()
    cal = info.camera_configuration.calibration_parameters
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height

    def make_model(cam_params) -> PinholePlaneCameraModel:
        return PinholePlaneCameraModel(
            width=w, height=h,
            f=(cam_params.fx, cam_params.fy),
            c=(cam_params.cx, cam_params.cy),
            distort_coeffs=NoDistortion(),
        )

    left_model = make_model(cal.left_cam)
    right_model = make_model(cal.right_cam)

    # stereo_transform: pose of right camera in left camera frame
    # sl.Transform stores a 4×4 matrix accessible via .m
    T_right_in_left = np.array(cal.stereo_transform.m, dtype=np.float64).reshape(4, 4)
    return left_model, right_model, T_right_in_left


def grab_gray_stereo(cam: sl.Camera) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Grab one frame and return (left_gray, right_gray) as H×W uint8 arrays.
    Returns None at end-of-file.
    """
    rt = sl.RuntimeParameters()
    err = cam.grab(rt)
    if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        return None
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Grab failed: {err}")

    mat_l = sl.Mat()
    mat_r = sl.Mat()
    cam.retrieve_image(mat_l, sl.VIEW.LEFT_GRAY)
    cam.retrieve_image(mat_r, sl.VIEW.RIGHT_GRAY)

    # sl.Mat.get_data() returns H×W×C; grayscale has C=4 (BGRA) or C=1
    left = mat_l.get_data()
    right = mat_r.get_data()

    # Convert to single-channel grayscale if needed
    if left.ndim == 3:
        left = left[:, :, 0]
    if right.ndim == 3:
        right = right[:, :, 0]

    return left.astype(np.uint8), right.astype(np.uint8)


# ---------------------------------------------------------------------------
# Fake HandPoseLabels (no GT annotations for ZED live data)
# ---------------------------------------------------------------------------

@dataclass
class _FakeHandPoseLabels:
    cameras: list
    camera_angles: List[float]
    hand_model: Optional[HandModel] = None

    def __len__(self):
        return 0  # no GT frames


# ---------------------------------------------------------------------------
# DualSvoStream: 4-view InputFrame iterator from two ZED SVO2 files
# ---------------------------------------------------------------------------

class DualSvoStream:
    """
    Synchronized stream from two ZED stereo cameras.

    Yields (InputFrame, gt_tracking) tuples where InputFrame has 4 views:
        index 0 – cam1 left
        index 1 – cam1 right
        index 2 – cam2 left
        index 3 – cam2 right

    World-space extrinsics come from the ArUco calibration JSON.
    Camera intrinsics and stereo baseline come from the SVO2 files via ZED SDK.
    """

    def __init__(self, svo1: Path, svo2: Path, calib_path: Path,
                 hand_model: Optional[HandModel] = None):
        logger.info(f"Loading ArUco extrinsics from {calib_path}")
        self._extrinsics = load_aruco_extrinsics(calib_path)

        logger.info("Opening SVO cameras …")
        self._cam1 = open_svo(svo1)
        self._cam2 = open_svo(svo2)

        sn1 = str(self._cam1.get_camera_information().serial_number)
        sn2 = str(self._cam2.get_camera_information().serial_number)

        if sn1 not in self._extrinsics:
            raise ValueError(f"SN{sn1} not found in calibration. "
                             f"Available: {list(self._extrinsics.keys())}")
        if sn2 not in self._extrinsics:
            raise ValueError(f"SN{sn2} not found in calibration. "
                             f"Available: {list(self._extrinsics.keys())}")

        # World-space poses for the LEFT camera of each stereo pair
        self._T_cam1_world = self._extrinsics[sn1]   # cam1_left → world
        self._T_cam2_world = self._extrinsics[sn2]   # cam2_left → world

        # Per-camera stereo intrinsics and right-in-left transforms
        cam1_left_model, cam1_right_model, T_cam1_right_in_left = \
            get_zed_stereo_transforms(self._cam1)
        cam2_left_model, cam2_right_model, T_cam2_right_in_left = \
            get_zed_stereo_transforms(self._cam2)

        # World-space poses for all 4 cameras
        T_cam1_right_world = self._T_cam1_world @ T_cam1_right_in_left
        T_cam2_right_world = self._T_cam2_world @ T_cam2_right_in_left

        self._cam_to_world = [
            self._T_cam1_world,     # 0: cam1 left
            T_cam1_right_world,     # 1: cam1 right
            self._T_cam2_world,     # 2: cam2 left
            T_cam2_right_world,     # 3: cam2 right
        ]

        # UmeTrack camera models (intrinsics; world transform injected per-frame)
        self._base_models = [
            cam1_left_model,
            cam1_right_model,
            cam2_left_model,
            cam2_right_model,
        ]

        # Camera angles: rough angular offset for crop-camera selection.
        # Left cameras at 0°, right cameras at 0° (same pointing direction
        # within a stereo pair; exact angles aren't critical for ZED data).
        camera_angles = [0.0, 0.0, 0.0, 0.0]

        self._hand_pose_labels = _FakeHandPoseLabels(
            cameras=self._base_models,
            camera_angles=camera_angles,
            hand_model=hand_model,
        )

        n1 = self._cam1.get_svo_number_of_frames()
        n2 = self._cam2.get_svo_number_of_frames()
        self._length = min(n1, n2)
        logger.info(f"Stream length: {self._length} frames "
                    f"(cam1={n1}, cam2={n2})")

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Tuple[InputFrame, dict]]:
        self._cam1.set_svo_position(0)
        self._cam2.set_svo_position(0)

        for frame_idx in range(self._length):
            pair1 = grab_gray_stereo(self._cam1)
            pair2 = grab_gray_stereo(self._cam2)

            if pair1 is None or pair2 is None:
                logger.warning(f"End of SVO reached at frame {frame_idx}, stopping.")
                break

            left1, right1 = pair1
            left2, right2 = pair2

            images = [left1, right1, left2, right2]

            views = []
            for cam_idx, (img, T_world) in enumerate(zip(images, self._cam_to_world)):
                model_with_pose = self._base_models[cam_idx].copy(
                    camera_to_world_xf=T_world
                )
                views.append(ViewData(
                    image=img,
                    camera=model_with_pose,
                    camera_angle=self._hand_pose_labels.camera_angles[cam_idx],
                ))

            yield InputFrame(views=views), {}

    def close(self):
        self._cam1.close()
        self._cam2.close()


# ---------------------------------------------------------------------------
# Calibration and tracking (mirrors run_inference_zed.py)
# ---------------------------------------------------------------------------

def _track_sequence_and_calibrate(
    stream: DualSvoStream,
    tracker: HandTracker,
    generic_hand_model: HandModel,
    n_calibration_samples: int,
) -> HandModel:
    predicted_scale_samples = []

    for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            stream._hand_pose_labels.camera_angles,
            generic_hand_model,
            gt_tracking,
            min_num_crops=1,
        )
        res = tracker.track_frame_and_calibrate_scale(input_frame, crop_cameras)
        for hand_idx in res.hand_poses.keys():
            predicted_scale_samples.append(res.predicted_scales[hand_idx])
        if n_calibration_samples != 0 and len(predicted_scale_samples) >= n_calibration_samples:
            predicted_scale_samples = predicted_scale_samples[:n_calibration_samples]
            break

    assert len(predicted_scale_samples) > 0, "No samples collected for scale calibration!"
    mean_scale = np.mean(predicted_scale_samples)
    logger.info(f"Calibrated mean scale: {mean_scale:.4f} from {len(predicted_scale_samples)} samples")
    return scaled_hand_model(generic_hand_model, mean_scale)


def _track_dual_svo(
    svo1: Path,
    svo2: Path,
    calib_path: Path,
    model_path: str,
    generic_hand_model: HandModel,
    output_dir: str,
    n_calibration_samples: int = 30,
    override: bool = False,
) -> None:
    # Build output filename from SVO names
    stem1 = svo1.stem
    stem2 = svo2.stem
    output_name = f"dual_{stem1}__{stem2}"
    output_path = os.path.join(output_dir, output_name + ".npy")

    if not override and os.path.exists(output_path):
        logger.info(f"Skipping: output already exists at {output_path}")
        return

    logger.info(f"Processing dual-cam: {svo1.name} + {svo2.name}")

    model = load_pretrained_model(model_path)
    model.eval()

    stream = DualSvoStream(svo1, svo2, calib_path)
    n_frames = len(stream)

    if n_frames == 0:
        logger.warning("Stream is empty, skipping.")
        stream.close()
        return

    tracker = HandTracker(model, HandTrackerOpts())

    # --- Calibration pass ---
    logger.info("Calibration pass …")
    calibrated_hand_model = _track_sequence_and_calibrate(
        stream, tracker, generic_hand_model, n_calibration_samples
    )

    # --- Tracking pass ---
    logger.info("Tracking pass …")
    tracker.reset_history()

    tracked_keypoints      = np.zeros([NUM_HANDS, n_frames, NUM_LANDMARKS_PER_HAND, 3])
    tracked_keypoints_eye  = np.zeros([NUM_HANDS, n_frames, NUM_LANDMARKS_PER_HAND, 3])
    tracked_keypoints_2d   = np.zeros([NUM_HANDS, n_frames, NUM_LANDMARKS_PER_HAND, 2])
    valid_tracking         = np.zeros([NUM_HANDS, n_frames], dtype=bool)

    # Use cam1-left (index 0) for eye-frame and 2-D projections
    _PROJ_CAM_IDX = 0

    for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            stream._hand_pose_labels.camera_angles,
            calibrated_hand_model,
            gt_tracking,
            min_num_crops=1,
        )
        res = tracker.track_frame(input_frame, calibrated_hand_model, crop_cameras)

        if not input_frame.views:
            logger.warning(f"No views at frame {frame_idx}, skipping.")
            continue

        proj_camera = input_frame.views[_PROJ_CAM_IDX].camera

        for hand_idx in res.hand_poses.keys():
            kp_world = landmarks_from_hand_pose(
                calibrated_hand_model, res.hand_poses[hand_idx], hand_idx
            )
            kp_eye = proj_camera.world_to_eye(kp_world)
            kp_2d  = proj_camera.eye_to_window(kp_eye)

            tracked_keypoints    [hand_idx, frame_idx] = kp_world
            tracked_keypoints_eye[hand_idx, frame_idx] = kp_eye
            tracked_keypoints_2d [hand_idx, frame_idx] = kp_2d
            valid_tracking       [hand_idx, frame_idx] = True

        if frame_idx % 50 == 0:
            n_hands = len(res.hand_poses)
            logger.info(f"  frame {frame_idx}/{n_frames}  hands detected: {n_hands}")

    stream.close()

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, {
        "tracked_keypoints": tracked_keypoints,
        "valid_tracking": valid_tracking,
    }, allow_pickle=True)
    logger.info(f"World-space keypoints saved to {output_path}")

    np.save(output_path.replace(".npy", "_eye_3d.npy"), {
        "tracked_keypoints": tracked_keypoints_eye,
        "valid_tracking": valid_tracking,
    }, allow_pickle=True)

    np.save(output_path.replace(".npy", "_2d.npy"), {
        "tracked_keypoints": tracked_keypoints_2d,
        "valid_tracking": valid_tracking,
    }, allow_pickle=True)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Run UmeTrack on two synchronized ZED stereo cameras (4 views)."
    )
    parser.add_argument("--svo1", type=Path, required=True,
                        help="Path to SVO2 file for camera 1")
    parser.add_argument("--svo2", type=Path, required=True,
                        help="Path to SVO2 file for camera 2")
    parser.add_argument("--calib", type=Path, required=True,
                        help="Path to ArUco calibration JSON with FusionConfiguration poses")
    parser.add_argument("--model-path", type=Path,
                        default=root / "pretrained_models" / "pretrained_weights.torch")
    parser.add_argument("--generic-hand-model", type=Path,
                        default=root / "UmeTrack_data" / "raw_data" / "real" /
                                "hand_hand" / "training" / "user_00" / "recording_00.json")
    parser.add_argument("--output-dir", type=str,
                        default=str(root / "zed_pred"))
    parser.add_argument("--n-calibration-samples", type=int, default=30)
    parser.add_argument("--override", action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    for p in (args.svo1, args.svo2, args.calib, args.model_path, args.generic_hand_model):
        if not Path(p).exists():
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    generic_hand_model = load_hand_model_from_dict(
        _load_json(str(args.generic_hand_model))["hand_model"]
    )

    _track_dual_svo(
        svo1=args.svo1,
        svo2=args.svo2,
        calib_path=args.calib,
        model_path=str(args.model_path),
        generic_hand_model=generic_hand_model,
        output_dir=args.output_dir,
        n_calibration_samples=args.n_calibration_samples,
        override=args.override,
    )


if __name__ == "__main__":
    main()
