# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass

from typing import Iterator, List

import av
import lib.data_utils.fs as fs
import numpy as np
import torch
from lib.common.camera import CameraModel, read_camera_from_json
from lib.common.hand import HandModel
from lib.tracker.tracker import InputFrame, ViewData

from .tracking_result import SingleHandPose


@dataclass
class HandPoseLabels:
    cameras: List[CameraModel]
    camera_angles: List[float]
    camera_to_world_transforms: np.ndarray
    hand_model: HandModel
    joint_angles: np.ndarray
    wrist_transforms: np.ndarray
    hand_confidences: np.ndarray

    def __len__(self):
        return len(self.joint_angles)


class VideoStream:
    def __init__(self, data_path: str):
        self._data_path = data_path

    def __len__(self) -> int:
        try:
            container = av.open(self._data_path)
            # take first video stream
            stream = container.streams.video[0]
            return stream.frames
        except (av.error.InvalidDataError, av.error.OSError, FileNotFoundError, IndexError) as e:
            print(f"Warning: Could not open video file {self._data_path}: {e}")
            # Return 0 as a fallback for invalid/corrupted files
            return 0
        except Exception as e:
            print(f"Warning: Unexpected error opening video file {self._data_path}: {e}")
            return 0

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            container = av.open(self._data_path)
            # take first video stream
            stream = container.streams.video[0]
            print(f"Opened ({int(stream.average_rate)} fps) video from {self._data_path}")

            for idx, frame in enumerate(container.decode(stream)):
                raw_mono_image_np = np.array(frame.to_image())[..., 0]
                yield raw_mono_image_np
        except (av.error.InvalidDataError, av.error.OSError, FileNotFoundError, IndexError) as e:
            print(f"Warning: Could not open video file {self._data_path}: {e}")
            # Return empty iterator for invalid/corrupted files
            return
        except Exception as e:
            print(f"Warning: Unexpected error opening video file {self._data_path}: {e}")
            return


def _load_json(p: str):
    with fs.open(p, "rb") as bf:
        return json.load(bf)


def load_hand_model_from_dict(hand_model_dict) -> HandModel:
    hand_tensor_dict = {}
    for k, v in hand_model_dict.items():
        if isinstance(v, list):
            hand_tensor_dict[k] = torch.Tensor(v)
        else:
            hand_tensor_dict[k] = v

    hand_model = HandModel(**hand_tensor_dict)
    return hand_model


def _load_hand_pose_labels(p: str) -> HandPoseLabels:
    labels = _load_json(p)
    cameras = [read_camera_from_json(c) for c in labels["cameras"]]
    camera_angles = labels["camera_angles"]
    hand_model = load_hand_model_from_dict(labels["hand_model"])
    joint_angles = np.array(labels["joint_angles"])
    wrist_transforms = np.array(labels["wrist_transforms"])
    hand_confidences = np.array(labels["hand_confidences"])
    camera_to_world_transforms = np.array(labels["camera_to_world_transforms"])

    return HandPoseLabels(
        cameras=cameras,
        camera_angles=camera_angles,
        camera_to_world_transforms=camera_to_world_transforms,
        hand_model=hand_model,
        joint_angles=joint_angles,
        wrist_transforms=wrist_transforms,
        hand_confidences=hand_confidences,
    )

"""
Image Sequence Pose Stream for ZED stereo camera data.

Similar to SyncedImagePoseStream but loads from image sequences instead of MP4 videos.
Compatible with ZED camera output format.
"""

import json
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional
from dataclasses import dataclass
from PIL import Image
import sys
import os

# Try to import UmeTrack modules if available
try:
    from lib.common.camera import CameraModel, read_camera_from_json
    from lib.common.hand import HandModel
    from lib.tracker.tracker import InputFrame, ViewData
    from lib.tracker.tracking_result import SingleHandPose
    UMETRACK_AVAILABLE = True
except ImportError:
    UMETRACK_AVAILABLE = False
    print("Warning: UmeTrack modules not available. Some functionality will be limited.")


class ImageSequenceStream:
    """Stream images from a directory of numbered PNG/JPG files."""
    
    def __init__(self, image_dir: str, image_format: str = 'png'):
        """
        Initialize image sequence stream.
        
        Args:
            image_dir: Directory containing numbered images (frame_000000.png, etc.)
            image_format: Image file extension ('png' or 'jpg')
        """
        self.image_dir = Path(image_dir)
        self.image_format = image_format
        
        # Find all images in directory
        self.image_files = sorted(list(self.image_dir.glob(f'*.{image_format}')))
        
        if not self.image_files:
            # Try alternative naming pattern
            self.image_files = sorted(list(self.image_dir.glob(f'[0-9]*.{image_format}')))
        
        if not self.image_files:
            raise FileNotFoundError(f"No {image_format} images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate through images, yielding them as numpy arrays (grayscale)."""
        for img_file in self.image_files:
            img = Image.open(img_file)
            # Convert to grayscale (UmeTrack expects grayscale images)
            if img.mode != 'L':
                img = img.convert('L')  # Convert RGB/RGBA to grayscale
            yield np.array(img)


class StereoImageSequenceStream:
    """Stream synchronized left and right image sequences."""
    
    def __init__(self, left_dir: str, right_dir: str, image_format: str = 'png'):
        """
        Initialize stereo image sequence stream.
        
        Args:
            left_dir: Directory with left camera images
            right_dir: Directory with right camera images  
            image_format: Image file extension
        """
        self.left_stream = ImageSequenceStream(left_dir, image_format)
        self.right_stream = ImageSequenceStream(right_dir, image_format)
        
        if len(self.left_stream) != len(self.right_stream):
            print(f"Warning: Left ({len(self.left_stream)}) and right ({len(self.right_stream)}) "
                  f"streams have different lengths. Using minimum length.")
            self._length = min(len(self.left_stream), len(self.right_stream))
        else:
            self._length = len(self.left_stream)
    
    def __len__(self) -> int:
        return self._length
    
    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate through synchronized left and right images."""
        left_iter = iter(self.left_stream)
        right_iter = iter(self.right_stream)
        
        for _ in range(self._length):
            left_img = next(left_iter)
            right_img = next(right_iter)
            yield left_img, right_img


@dataclass
class HandPoseLabels:
    """Hand pose labels compatible with UmeTrack format."""
    cameras: List
    camera_angles: List[float]
    camera_to_world_transforms: np.ndarray
    hand_model: Optional[any]
    joint_angles: np.ndarray
    wrist_transforms: np.ndarray
    hand_confidences: np.ndarray

    def __len__(self):
        return len(self.joint_angles)


def load_hand_model_from_dict(hand_model_dict):
    """Load hand model from dictionary."""
    if not UMETRACK_AVAILABLE:
        return None
    
    import torch
    hand_tensor_dict = {}
    for k, v in hand_model_dict.items():
        if isinstance(v, list):
            hand_tensor_dict[k] = torch.Tensor(v)
        else:
            hand_tensor_dict[k] = v
    
    return HandModel(**hand_tensor_dict)


def _load_json(p: str):
    """Load JSON file."""
    with open(p, 'r') as f:
        return json.load(f)


def _create_camera_from_intrinsics(K, width, height, distortion_coeffs=None):
    """Create a CameraModel from intrinsics matrix."""
    if not UMETRACK_AVAILABLE:
        return {"K": K, "width": width, "height": height}
    
    from lib.common.camera import PinholePlaneCameraModel, NoDistortion
    
    # Extract focal lengths and principal point from K matrix
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    
    # Use pinhole model with no distortion (ZED cameras are already undistorted)
    # PinholePlaneCameraModel expects: (width, height, (fx,fy), (cx,cy), distort_coeffs)
    return PinholePlaneCameraModel(
        width=width,
        height=height,
        f=(fx, fy),
        c=(cx, cy),
        distort_coeffs=NoDistortion(),  # No distortion for pinhole model
    )


def _load_hand_pose_labels(json_path: str, num_frames: int = 0, image_width: int = 2208, image_height: int = 1242) -> HandPoseLabels:
    """
    Load hand pose labels from JSON file.
    
    Supports two formats:
    1. Full UmeTrack format with cameras, camera_angles, hand_model, etc.
    2. Simple camera intrinsics format with K_left, K_right, etc.
    
    Args:
        json_path: Path to JSON file
        num_frames: Number of frames (for creating empty pose data)
        image_width: Image width (for simple format)
        image_height: Image height (for simple format)
    """
    labels = _load_json(json_path)
    
    # Detect format: check if it has "cameras" key (UmeTrack format) or "K_left" (simple format)
    is_simple_format = "K_left" in labels or "K_right" in labels
    
    if is_simple_format:
        print("Detected simple camera intrinsics format (K_left/K_right)")
        
        # Create cameras from K matrices
        cameras = []
        camera_angles = []
        
        if "K_left" in labels:
            cam_left = _create_camera_from_intrinsics(
                labels["K_left"], 
                image_width, 
                image_height,
                labels.get("distortion_coefficients_left")
            )
            cameras.append(cam_left)
            camera_angles.append(0.0)  # Left camera at 0 degrees
        
        if "K_right" in labels:
            cam_right = _create_camera_from_intrinsics(
                labels["K_right"], 
                image_width, 
                image_height,
                labels.get("distortion_coefficients_right")
            )
            cameras.append(cam_right)
            camera_angles.append(90.0)  # Right camera at 90 degrees (standard stereo)
        
        # Check if hand_model is provided (hybrid format)
        hand_model = None
        if "hand_model" in labels and UMETRACK_AVAILABLE:
            try:
                hand_model = load_hand_model_from_dict(labels["hand_model"])
                print(f"  Loaded hand model from JSON")
            except (TypeError, KeyError) as e:
                print(f"  Warning: Could not load hand model: {e}. Using None.")
        
        # Check if pose data is provided, otherwise use empty arrays
        joint_angles = np.array(labels.get("joint_angles", []))
        wrist_transforms = np.array(labels.get("wrist_transforms", []))
        hand_confidences = np.array(labels.get("hand_confidences", []))
        camera_to_world_transforms = np.array(labels.get("camera_to_world_transforms", []))
        
        # If no pose data, create empty arrays
        if len(joint_angles) == 0:
            joint_angles = np.zeros((num_frames, 2, 15))  # (frames, 2 hands, 15 joints)
        if len(wrist_transforms) == 0:
            wrist_transforms = np.zeros((num_frames, 2, 4, 4))  # (frames, 2 hands, 4x4 transform)
        if len(hand_confidences) == 0:
            hand_confidences = np.zeros((num_frames, 2))  # (frames, 2 hands) - all zero = no hands
        if len(camera_to_world_transforms) == 0:
            camera_to_world_transforms = np.tile(np.eye(4), (num_frames, len(cameras), 1, 1))  # Identity transforms
        
        print(f"  Created {len(cameras)} camera(s)")
        if len(labels.get("joint_angles", [])) > 0:
            print(f"  Loaded hand pose data with {len(joint_angles)} frames")
        else:
            print(f"  No hand pose data (empty arrays with {num_frames} frames)")
        
    else:
        print("Detected full UmeTrack format with hand pose labels")
        
        # Original UmeTrack format
        if UMETRACK_AVAILABLE:
            cameras = [read_camera_from_json(c) for c in labels["cameras"]]
        else:
            cameras = labels["cameras"]
        
        camera_angles = labels["camera_angles"]
        
        # Load hand model if available
        hand_model = None
        if "hand_model" in labels and UMETRACK_AVAILABLE:
            try:
                hand_model = load_hand_model_from_dict(labels["hand_model"])
            except (TypeError, KeyError) as e:
                print(f"Warning: Could not load hand model: {e}. Using None.")
        
        # Load pose data
        joint_angles = np.array(labels.get("joint_angles", []))
        wrist_transforms = np.array(labels.get("wrist_transforms", []))
        hand_confidences = np.array(labels.get("hand_confidences", []))
        camera_to_world_transforms = np.array(labels.get("camera_to_world_transforms", []))
    
    return HandPoseLabels(
        cameras=cameras,
        camera_angles=camera_angles,
        camera_to_world_transforms=camera_to_world_transforms,
        hand_model=hand_model,
        joint_angles=joint_angles,
        wrist_transforms=wrist_transforms,
        hand_confidences=hand_confidences,
    )


class ImageSequencePoseStream:
    """
    Image sequence pose stream for ZED stereo camera data.
    
    Similar to SyncedImagePoseStream but loads from PNG/JPG sequences
    instead of MP4 videos.
    """
    
    def __init__(self, left_dir: str, right_dir: str, json_path: str, 
                 image_format: str = 'png'):
        """
        Initialize image sequence pose stream.
        
        Args:
            left_dir: Directory with left camera images
            right_dir: Directory with right camera images
            json_path: Path to JSON file with camera intrinsics and pose data
            image_format: Image file extension ('png' or 'jpg')
        """
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.json_path = json_path
        
        # Load image streams
        self._image_stream = StereoImageSequenceStream(left_dir, right_dir, image_format)
        sequence_length = len(self._image_stream)
        
        if sequence_length == 0:
            print(f"Warning: No images found, stream is invalid")
            self._is_valid = False
            self._hand_pose_labels = None
        else:
            # Get image dimensions from first image
            first_img_path = self._image_stream.left_stream.image_files[0]
            first_img = Image.open(first_img_path)
            image_width, image_height = first_img.size
            
            # Load pose labels (with image dimensions for simple format)
            self._hand_pose_labels = _load_hand_pose_labels(
                json_path, 
                num_frames=sequence_length,
                image_width=image_width,
                image_height=image_height
            )
            self._is_valid = True
            
            # Validate lengths if pose data is present
            if len(self._hand_pose_labels.joint_angles) > 0:
                if len(self._hand_pose_labels) != sequence_length:
                    print(f"Warning: Mismatch between pose labels ({len(self._hand_pose_labels)}) "
                          f"and image frames ({sequence_length})")
    
    def __len__(self) -> int:
        if not self._is_valid:
            return 0
        return len(self._image_stream)
    
    def __iter__(self):
        """Iterate through frames, yielding InputFrame and ground truth tracking."""
        if not self._is_valid or self._hand_pose_labels is None:
            return
        
        if not UMETRACK_AVAILABLE:
            print("Error: UmeTrack modules required for iteration")
            return
        
        for frame_idx, (left_img, right_img) in enumerate(self._image_stream):
            # Prepare ground truth tracking if available
            gt_tracking = {}
            if len(self._hand_pose_labels.joint_angles) > frame_idx:
                for hand_idx in range(0, 2):
                    if self._hand_pose_labels.hand_confidences[frame_idx, hand_idx] > 0:
                        gt_tracking[hand_idx] = SingleHandPose(
                            joint_angles=self._hand_pose_labels.joint_angles[frame_idx, hand_idx],
                            wrist_xform=self._hand_pose_labels.wrist_transforms[frame_idx, hand_idx],
                            hand_confidence=self._hand_pose_labels.hand_confidences[frame_idx, hand_idx],
                        )
            
            # Stack left and right images for multi-view
            # Assuming 2 cameras (left and right)
            multi_view_images = np.stack([left_img, right_img], axis=1)
            
            # Check camera to world transforms
            invalid_camera_to_world = True
            if len(self._hand_pose_labels.camera_to_world_transforms) > frame_idx:
                invalid_camera_to_world = (
                    self._hand_pose_labels.camera_to_world_transforms[frame_idx].sum() == 0
                )
            
            if invalid_camera_to_world:
                if gt_tracking:
                    print(f"Warning: Frame {frame_idx} has tracking but no camera transforms")
            
            # Create views for each camera
            views = []
            for cam_idx in range(len(self._hand_pose_labels.cameras)):
                cur_camera = self._hand_pose_labels.cameras[cam_idx].copy(
                    camera_to_world_xf=self._hand_pose_labels.camera_to_world_transforms[
                        frame_idx, cam_idx
                    ] if len(self._hand_pose_labels.camera_to_world_transforms) > frame_idx else np.eye(4),
                )
                
                views.append(
                    ViewData(
                        image=multi_view_images[:, cam_idx, :],
                        camera=cur_camera,
                        camera_angle=self._hand_pose_labels.camera_angles[cam_idx],
                    )
                )
            
            input_frame = InputFrame(views=views)
            yield input_frame, gt_tracking


class SyncedImagePoseStream:
    def __init__(self, data_path: str):
        self._data_path = data_path
        self._image_stream = VideoStream(data_path)
        video_length = len(self._image_stream)
        
        if video_length == 0:
            print(f"Warning: Video file {data_path} appears to be invalid or corrupted, skipping...")
            # Set a flag to indicate this stream is invalid
            self._is_valid = False
            self._hand_pose_labels = None
        else:
            # Only load pose labels if video is valid
            label_path = data_path[:-4] + ".json"
            self._hand_pose_labels = _load_hand_pose_labels(label_path)
            self._is_valid = True
            assert len(self._hand_pose_labels) == video_length, f"Mismatch between pose labels ({len(self._hand_pose_labels)}) and video frames ({video_length})"

    def __len__(self) -> int:
        if not self._is_valid:
            return 0
        return len(self._image_stream)

    def __iter__(self):
        if not self._is_valid or self._hand_pose_labels is None:
            return
        for frame_idx, raw_mono in enumerate(self._image_stream):
            gt_tracking = {}
            for hand_idx in range(0, 2):
                if self._hand_pose_labels.hand_confidences[frame_idx, hand_idx] > 0:
                    gt_tracking[hand_idx] = SingleHandPose(
                        joint_angles=self._hand_pose_labels.joint_angles[
                            frame_idx, hand_idx
                        ],
                        wrist_xform=self._hand_pose_labels.wrist_transforms[
                            frame_idx, hand_idx
                        ],
                        hand_confidence=self._hand_pose_labels.hand_confidences[
                            frame_idx, hand_idx
                        ],
                    )

            multi_view_images = raw_mono.reshape(
                raw_mono.shape[0], len(self._hand_pose_labels.cameras), -1
            )
            invalid_camera_to_world = (
                self._hand_pose_labels.camera_to_world_transforms[frame_idx].sum() == 0
            )
            if invalid_camera_to_world:
                assert (
                    not gt_tracking
                ), f"Cameras are not tracked, expecting no ground truth tracking!"

            views = []
            for cam_idx in range(0, len(self._hand_pose_labels.cameras)):
                cur_camera = self._hand_pose_labels.cameras[cam_idx].copy(
                    camera_to_world_xf=self._hand_pose_labels.camera_to_world_transforms[
                        frame_idx, cam_idx
                    ],
                )

                views.append(
                    ViewData(
                        image=multi_view_images[:, cam_idx, :],
                        camera=cur_camera,
                        camera_angle=self._hand_pose_labels.camera_angles[cam_idx],
                    )
                )

            input_frame = InputFrame(views=views)
            yield input_frame, gt_tracking
