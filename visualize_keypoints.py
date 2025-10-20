#!/usr/bin/env python3
"""
Visualization script for UmeTrack hand keypoints on video frames.

This script overlays predicted hand keypoints on video frames and saves the result as a video.
It can visualize both ground truth and predicted keypoints with different colors.

Usage:
    python visualize_keypoints.py --input_video path/to/video.mp4 --output_video path/to/output.mp4
    python visualize_keypoints.py --input_video path/to/video.mp4 --output_video path/to/output.mp4 --show_gt
    python visualize_keypoints.py --input_video path/to/video.mp4 --output_video path/to/output.mp4 --eval_results path/to/eval_results.npy
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts
from lib.tracker.tracking_result import SingleHandPose
from lib.models.model_loader import load_pretrained_model
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND

logger = logging.getLogger(__name__)

# Colors for different hands and keypoint types
HAND_COLORS = {
    0: (0, 255, 0),    # Green for left hand
    1: (255, 0, 0),    # Blue for right hand
}

GT_COLORS = {
    0: (0, 255, 255),  # Yellow for left hand GT
    1: (255, 0, 255),  # Magenta for right hand GT
}

PRED_COLORS = {
    0: (0, 255, 0),    # Green for left hand prediction
    1: (255, 0, 0),    # Blue for right hand prediction
}

# Keypoint connections for hand skeleton visualization (MediaPipe-style)
# Based on MediaPipe hand landmark model structure
HAND_CONNECTIONS = [
    # Thumb (4 landmarks: tip to base)
    (0, 7), (7, 6),
    # Index finger (4 landmarks: tip to base)
    (1, 10), (10, 9), (9, 8),
    # Middle finger (4 landmarks: tip to base)
    (2, 13), (13, 12), (12, 11), 
    # Ring finger (4 landmarks: tip to base)
    (3, 16), (16, 15), (15, 14),
    # Pinky (3 landmarks: tip to base)
    (4, 19), (19, 18), (18, 17),
    # Palm connections (connecting finger bases in order)
    (17, 5), (5, 6), (6, 8), (8, 11), (11, 14), (14, 17), (5, 20), (20, 11)
]


def project_keypoints_to_image(keypoints_3d: np.ndarray, camera) -> np.ndarray:
    """
    Project 3D keypoints to 2D image coordinates.
    
    Args:
        keypoints_3d: Array of shape (N, 3) with 3D keypoints
        camera: Camera model for projection
        
    Returns:
        Array of shape (N, 2) with 2D keypoints
    """
    if len(keypoints_3d) == 0:
        return np.array([])
    
    # Project 3D points to 2D: world -> eye -> window
    eye_points = camera.world_to_eye(keypoints_3d)
    keypoints_2d = camera.eye_to_window(eye_points)
    return keypoints_2d


def draw_hand_skeleton(image: np.ndarray, keypoints_2d: np.ndarray, color: Tuple[int, int, int], 
                      thickness: int = 2, radius: int = 3, show_indices: bool = False, 
                      highlight_index: int = -1) -> np.ndarray:
    """
    Draw hand skeleton on the image.
    
    Args:
        image: Input image
        keypoints_2d: Array of shape (21, 2) with 2D keypoints
        color: Color for drawing (B, G, R)
        thickness: Line thickness
        radius: Keypoint radius
        show_indices: Whether to show keypoint indices as text labels
        highlight_index: Index of keypoint to highlight with text (-1 for none)
        
    Returns:
        Image with drawn skeleton
    """
    if len(keypoints_2d) == 0:
        return image
    
    image = image.copy()
    
    # Draw keypoints
    for idx, point in enumerate(keypoints_2d):
        if not np.isnan(point).any() and point[0] >= 0 and point[1] >= 0:
            # Make highlighted keypoint larger and brighter
            if idx == highlight_index:
                cv2.circle(image, (int(point[0]), int(point[1])), radius + 2, color, -1)
                # Add index label for highlighted keypoint
                if show_indices:
                    text_pos = (int(point[0]) + radius + 5, int(point[1]) - radius - 5)
                    cv2.putText(image, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 255), 1)  # Red text
            else:
                cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if (start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d) and
            not np.isnan(keypoints_2d[start_idx]).any() and not np.isnan(keypoints_2d[end_idx]).any()):
            start_point = (int(keypoints_2d[start_idx][0]), int(keypoints_2d[start_idx][1]))
            end_point = (int(keypoints_2d[end_idx][0]), int(keypoints_2d[end_idx][1]))
            
            # Check if points are within image bounds
            if (0 <= start_point[0] < image.shape[1] and 0 <= start_point[1] < image.shape[0] and
                0 <= end_point[0] < image.shape[1] and 0 <= end_point[1] < image.shape[0]):
                cv2.line(image, start_point, end_point, color, thickness)
    
    return image


def draw_keypoints_on_camera_view(
    image: np.ndarray, 
    view, 
    input_frame, 
    gt_tracking, 
    image_pose_stream, 
    model, 
    tracker, 
    eval_data, 
    frame_idx: int, 
    show_gt: bool, 
    show_predictions: bool,
    show_indices: bool = False,
    highlight_index: int = -1,
    hand_filter: str = "both"
) -> np.ndarray:
    """
    Draw keypoints on a single camera view.
    
    Args:
        image: Input image for this camera view
        view: Camera view data
        input_frame: Full input frame data
        gt_tracking: Ground truth tracking data
        image_pose_stream: Image pose stream
        model: Pretrained model
        tracker: Hand tracker
        eval_data: Evaluation data
        frame_idx: Current frame index
        show_gt: Whether to show ground truth
        show_predictions: Whether to show predictions
        show_indices: Whether to show keypoint indices as labels
        highlight_index: Index of keypoint to highlight with text (-1 for none)
        hand_filter: Which hands to show ("left", "right", or "both")
        
    Returns:
        Image with keypoints drawn
    """
    # Draw ground truth keypoints
    if show_gt and gt_tracking:
        hand_model = image_pose_stream._hand_pose_labels.hand_model
        for hand_idx, gt_pose in gt_tracking.items():
            # Apply hand filter
            if hand_filter == "left" and hand_idx != 0:
                continue
            if hand_filter == "right" and hand_idx != 1:
                continue
            
            if gt_pose.hand_confidence > 0.5:  # Only draw confident detections
                gt_keypoints_3d = landmarks_from_hand_pose(hand_model, gt_pose, hand_idx)
                gt_keypoints_2d = project_keypoints_to_image(gt_keypoints_3d, view.camera)
                
                if len(gt_keypoints_2d) > 0:
                    color = GT_COLORS.get(hand_idx, (0, 255, 255))
                    image = draw_hand_skeleton(image, gt_keypoints_2d, color, thickness=2, radius=3, 
                                            show_indices=show_indices, highlight_index=highlight_index)
    
    # Draw predicted keypoints
    if show_predictions and tracker and model:
        try:
            hand_model = image_pose_stream._hand_pose_labels.hand_model
            
            # Try to generate crop cameras from ground truth first
            crop_cameras = tracker.gen_crop_cameras(
                [v.camera for v in input_frame.views],
                image_pose_stream._hand_pose_labels.camera_angles,
                hand_model,
                gt_tracking,
                min_num_crops=1,
            )
            
            
            # If no crop cameras from GT, try to use previous frame's tracking result
            if not crop_cameras and hasattr(tracker, '_last_tracking_result') and tracker._last_tracking_result:
                # Use previous frame's hand poses to generate crop cameras
                prev_hand_poses = tracker._last_tracking_result.hand_poses
                if prev_hand_poses:
                    # Create a temporary gt_tracking from previous predictions
                    temp_gt_tracking = {}
                    for hand_idx, prev_pose in prev_hand_poses.items():
                        temp_gt_tracking[hand_idx] = prev_pose
                    
                    crop_cameras = tracker.gen_crop_cameras(
                        [v.camera for v in input_frame.views],
                        image_pose_stream._hand_pose_labels.camera_angles,
                        hand_model,
                        temp_gt_tracking,
                        min_num_crops=1,
                    )
            
            # If still no crop cameras, try to bootstrap with a default hand pose
            if not crop_cameras and frame_idx == 0:
                logger.info("Bootstrapping tracking with default hand pose")
                # Create a default hand pose at a reasonable position
                default_poses = {}
                for hand_idx in range(2):  # Try both hands
                    # Place hand at center of the scene, slightly forward
                    default_wrist_xform = np.eye(4)
                    default_wrist_xform[2, 3] = 500  # 500mm forward
                    if hand_idx == 1:  # Right hand
                        default_wrist_xform[0, 3] = 100  # 100mm to the right
                    else:  # Left hand
                        default_wrist_xform[0, 3] = -100  # 100mm to the left
                    
                    default_pose = SingleHandPose(
                        joint_angles=np.zeros(21, dtype=np.float32),  # Neutral pose
                        wrist_xform=default_wrist_xform,
                        hand_confidence=0.8  # High confidence to pass threshold
                    )
                    default_poses[hand_idx] = default_pose
                
                crop_cameras = tracker.gen_crop_cameras(
                    [v.camera for v in input_frame.views],
                    image_pose_stream._hand_pose_labels.camera_angles,
                    hand_model,
                    default_poses,
                    min_num_crops=1,
                )
            
            if crop_cameras:
                res = tracker.track_frame(input_frame, hand_model, crop_cameras)
                # Store the result for next frame
                tracker._last_tracking_result = res
                
                for hand_idx, pred_pose in res.hand_poses.items():
                    # Apply hand filter
                    if hand_filter == "left" and hand_idx != 0:
                        continue
                    if hand_filter == "right" and hand_idx != 1:
                        continue
                    
                    pred_keypoints_3d = landmarks_from_hand_pose(hand_model, pred_pose, hand_idx)
                    pred_keypoints_2d = project_keypoints_to_image(pred_keypoints_3d, view.camera)
                    
                    if len(pred_keypoints_2d) > 0:
                        color = PRED_COLORS.get(hand_idx, (0, 255, 0))
                        image = draw_hand_skeleton(image, pred_keypoints_2d, color, thickness=2, radius=3, 
                                                show_indices=show_indices, highlight_index=highlight_index)
            else:
                logger.debug(f"No crop cameras generated for frame {frame_idx} - insufficient GT data or visibility")
        except Exception as e:
            logger.warning(f"Failed to process frame {frame_idx} for predictions: {e}")
    
    # Draw keypoints from evaluation results
    if eval_data is not None:
        if 'tracked_keypoints' in eval_data and 'valid_tracking' in eval_data:
            tracked_keypoints = eval_data['tracked_keypoints']
            valid_tracking = eval_data['valid_tracking']
            
            for hand_idx in range(NUM_HANDS):
                # Apply hand filter
                if hand_filter == "left" and hand_idx != 0:
                    continue
                if hand_filter == "right" and hand_idx != 1:
                    continue
                
                if (hand_idx < tracked_keypoints.shape[0] and 
                    frame_idx < tracked_keypoints.shape[1] and
                    valid_tracking[hand_idx, frame_idx]):
                    
                    keypoints_3d = tracked_keypoints[hand_idx, frame_idx]
                    keypoints_2d = project_keypoints_to_image(keypoints_3d, view.camera)
                    
                    if len(keypoints_2d) > 0:
                        color = PRED_COLORS.get(hand_idx, (0, 255, 0))
                        image = draw_hand_skeleton(image, keypoints_2d, color, thickness=2, radius=3, 
                                                show_indices=show_indices, highlight_index=highlight_index)
    
    return image


def generate_default_output_path(input_video_path: str, suffix: str = "_visualized") -> str:
    """
    Generate default output path based on input video path.
    
    Args:
        input_video_path: Path to input video file
        suffix: Suffix to add before file extension
        
    Returns:
        Generated output path
    """
    import os
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Extract the relative path from UmeTrack_data/raw_data/
    if "UmeTrack_data/raw_data/" in input_video_path:
        # Find the part after UmeTrack_data/raw_data/
        parts = input_video_path.split("UmeTrack_data/raw_data/")
        if len(parts) > 1:
            relative_path = parts[1]
            # Remove the .mp4 extension and add suffix
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            dir_path = os.path.dirname(relative_path)
            
            # Create output directory structure
            output_dir = os.path.join(script_dir, "pred", dir_path)
            output_filename = f"{base_name}{suffix}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            return output_path
    
    # Fallback: create output in same directory as input
    base_name = os.path.splitext(input_video_path)[0]
    return f"{base_name}{suffix}.mp4"


def visualize_video_with_keypoints(
    video_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    eval_results_path: Optional[str] = None,
    show_gt: bool = False,
    show_predictions: bool = True,
    camera_idx: int = 0,
    show_all_cameras: bool = True,
    show_indices: bool = False,
    hand_filter: str = "both"
):
    """
    Visualize hand keypoints on video frames.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        model_path: Path to pretrained model (for predictions)
        eval_results_path: Path to evaluation results .npy file
        show_gt: Whether to show ground truth keypoints
        show_predictions: Whether to show predicted keypoints
        camera_idx: Which camera view to visualize (when show_all_cameras=False)
        show_all_cameras: Whether to show all camera views in a 4-panel layout
        show_indices: Whether to show keypoint indices as labels
        hand_filter: Which hands to show ("left", "right", or "both")
    """
    logger.info(f"Processing video: {video_path}")
    
    # Load video stream
    try:
        image_pose_stream = SyncedImagePoseStream(video_path)
        if len(image_pose_stream) == 0:
            logger.error("Invalid video file or no frames available")
            return
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        return
    
    # Load evaluation results if provided
    eval_data = None
    if eval_results_path and os.path.exists(eval_results_path):
        try:
            eval_data = np.load(eval_results_path, allow_pickle=True).item()
            logger.info(f"Loaded evaluation results from: {eval_results_path}")
        except Exception as e:
            logger.error(f"Failed to load evaluation results: {e}")
            eval_data = None
    
    # Load model for predictions if needed
    model = None
    tracker = None
    if show_predictions and model_path and os.path.exists(model_path):
        try:
            model = load_pretrained_model(model_path)
            model.eval()
            tracker = HandTracker(model, HandTrackerOpts())
            # Initialize tracking result storage for temporal consistency
            tracker._last_tracking_result = None
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
            tracker = None
    
    # Get video properties
    first_frame = next(iter(image_pose_stream))
    input_frame, _ = first_frame
    
    if show_all_cameras:
        # Use all 4 camera views
        num_cameras = len(input_frame.views)
        sample_image = input_frame.views[0].image
        single_height, single_width = sample_image.shape[:2]
        # Concatenate all cameras horizontally
        total_width = single_width * num_cameras
        total_height = single_height
        logger.info(f"Using all {num_cameras} cameras: {single_height}x{single_width} each, total: {total_height}x{total_width}")
    else:
        # Use single camera view
        if camera_idx >= len(input_frame.views):
            logger.error(f"Camera index {camera_idx} out of range. Available cameras: {len(input_frame.views)}")
            return
        sample_image = input_frame.views[camera_idx].image
        total_height, total_width = sample_image.shape[:2]
        logger.info(f"Using camera {camera_idx}: {total_height}x{total_width}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (total_width, total_height))
    
    logger.info(f"Processing {len(image_pose_stream)} frames...")
    
    # Process each frame
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        # Cycle through keypoint indices (0-20) when showing indices
        highlight_index = -1
        if show_indices:
            highlight_index = frame_idx % 21  # Cycle through 0-20
        if show_all_cameras:
            # Process all camera views
            camera_images = []
            for cam_idx in range(len(input_frame.views)):
                view = input_frame.views[cam_idx]
                image = view.image.copy()
                
                # Convert to BGR for OpenCV
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw keypoints on this camera view
                image = draw_keypoints_on_camera_view(
                    image, view, input_frame, gt_tracking, 
                    image_pose_stream, model, tracker, eval_data, 
                    frame_idx, show_gt, show_predictions, show_indices, highlight_index, hand_filter
                )
                
                camera_images.append(image)
            
            # Concatenate all camera views horizontally
            image = np.concatenate(camera_images, axis=1)
        else:
            # Process single camera view
            view = input_frame.views[camera_idx]
            image = view.image.copy()
                       # Convert to BGR for OpenCV
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw keypoints on this camera view
            image = draw_keypoints_on_camera_view(
                image, view, input_frame, gt_tracking, 
                image_pose_stream, model, tracker, eval_data, 
                frame_idx, show_gt, show_predictions, show_indices, highlight_index, hand_filter
            )
        
        # Add frame information
        cv2.putText(image, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(image)
        
        if frame_idx % 30 == 0:
            logger.info(f"Processed frame {frame_idx}/{len(image_pose_stream)}")
    
    # Release everything
    out.release()
    logger.info(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hand keypoints on video")
    parser.add_argument("--input_video", required=True, help="Path to input video file")
    parser.add_argument("--output_video", help="Path to output video file (auto-generated if not provided)")
    parser.add_argument("--model_path", help="Path to pretrained model for predictions")
    parser.add_argument("--eval_results", help="Path to evaluation results .npy file")
    parser.add_argument("--show_gt", action="store_true", help="Show ground truth keypoints")
    parser.add_argument("--show_predictions", action="store_true", default=True, help="Show predicted keypoints")
    parser.add_argument("--show_indices", action="store_true", help="Show keypoint indices as labels (cycles through 0-20)")
    parser.add_argument("--hand_filter", choices=["left", "right", "both"], default="both", help="Which hands to visualize")
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera index to visualize (when --single_camera is used)")
    parser.add_argument("--single_camera", action="store_true", help="Show only single camera view instead of all 4 cameras")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        logger.error(f"Input video not found: {args.input_video}")
        return
    
    # Generate output path if not provided
    if args.output_video is None:
        # Determine suffix based on visualization type
        suffix_parts = []
        if args.show_gt:
            suffix_parts.append("gt")
        if args.show_predictions:
            suffix_parts.append("pred")
        if not args.single_camera:
            suffix_parts.append("4cam")
        else:
            suffix_parts.append(f"cam{args.camera_idx}")
        
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else "_visualized"
        args.output_video = generate_default_output_path(args.input_video, suffix)
        logger.info(f"Auto-generated output path: {args.output_video}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    # Run visualization
    visualize_video_with_keypoints(
        video_path=args.input_video,
        output_path=args.output_video,
        model_path=args.model_path,
        eval_results_path=args.eval_results,
        show_gt=args.show_gt,
        show_predictions=args.show_predictions,
        camera_idx=args.camera_idx,
        show_all_cameras=not args.single_camera,
        show_indices=args.show_indices,
        hand_filter=args.hand_filter
    )


if __name__ == "__main__":
    main()
