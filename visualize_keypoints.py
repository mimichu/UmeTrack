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

# Keypoint connections for hand skeleton visualization
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
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
                      thickness: int = 2, radius: int = 3) -> np.ndarray:
    """
    Draw hand skeleton on the image.
    
    Args:
        image: Input image
        keypoints_2d: Array of shape (21, 2) with 2D keypoints
        color: Color for drawing (B, G, R)
        thickness: Line thickness
        radius: Keypoint radius
        
    Returns:
        Image with drawn skeleton
    """
    if len(keypoints_2d) == 0:
        return image
    
    image = image.copy()
    
    # Draw keypoints
    for point in keypoints_2d:
        if not np.isnan(point).any() and point[0] >= 0 and point[1] >= 0:
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


def visualize_video_with_keypoints(
    video_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    eval_results_path: Optional[str] = None,
    show_gt: bool = False,
    show_predictions: bool = True,
    camera_idx: int = 0
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
        camera_idx: Which camera view to visualize
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
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
            tracker = None
    
    # Get video properties
    first_frame = next(iter(image_pose_stream))
    input_frame, _ = first_frame
    if camera_idx >= len(input_frame.views):
        logger.error(f"Camera index {camera_idx} out of range. Available cameras: {len(input_frame.views)}")
        return
    
    sample_image = input_frame.views[camera_idx].image
    height, width = sample_image.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    logger.info(f"Processing {len(image_pose_stream)} frames...")
    
    # Process each frame
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        # Get the camera view
        view = input_frame.views[camera_idx]
        image = view.image.copy()
        
        # Convert to BGR for OpenCV
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw ground truth keypoints
        if show_gt and gt_tracking:
            hand_model = image_pose_stream._hand_pose_labels.hand_model
            for hand_idx, gt_pose in gt_tracking.items():
                if gt_pose.hand_confidence > 0.5:  # Only draw confident detections
                    gt_keypoints_3d = landmarks_from_hand_pose(hand_model, gt_pose, hand_idx)
                    gt_keypoints_2d = project_keypoints_to_image(gt_keypoints_3d, view.camera)
                    
                    if len(gt_keypoints_2d) > 0:
                        color = GT_COLORS.get(hand_idx, (0, 255, 255))
                        image = draw_hand_skeleton(image, gt_keypoints_2d, color, thickness=2, radius=3)
        
        # Draw predicted keypoints
        if show_predictions and tracker and model:
            try:
                hand_model = image_pose_stream._hand_pose_labels.hand_model
                crop_cameras = tracker.gen_crop_cameras(
                    [v.camera for v in input_frame.views],
                    image_pose_stream._hand_pose_labels.camera_angles,
                    hand_model,
                    gt_tracking,
                    min_num_crops=1,
                )
                
                if crop_cameras:
                    res = tracker.track_frame(input_frame, hand_model, crop_cameras)
                    
                    for hand_idx, pred_pose in res.hand_poses.items():
                        pred_keypoints_3d = landmarks_from_hand_pose(hand_model, pred_pose, hand_idx)
                        pred_keypoints_2d = project_keypoints_to_image(pred_keypoints_3d, view.camera)
                        
                        if len(pred_keypoints_2d) > 0:
                            color = PRED_COLORS.get(hand_idx, (0, 255, 0))
                            image = draw_hand_skeleton(image, pred_keypoints_2d, color, thickness=2, radius=3)
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_idx}: {e}")
        
        # Draw keypoints from evaluation results
        if eval_data is not None:
            if 'tracked_keypoints' in eval_data and 'valid_tracking' in eval_data:
                tracked_keypoints = eval_data['tracked_keypoints']
                valid_tracking = eval_data['valid_tracking']
                
                for hand_idx in range(NUM_HANDS):
                    if (hand_idx < tracked_keypoints.shape[0] and 
                        frame_idx < tracked_keypoints.shape[1] and
                        valid_tracking[hand_idx, frame_idx]):
                        
                        keypoints_3d = tracked_keypoints[hand_idx, frame_idx]
                        keypoints_2d = project_keypoints_to_image(keypoints_3d, view.camera)
                        
                        if len(keypoints_2d) > 0:
                            color = PRED_COLORS.get(hand_idx, (0, 255, 0))
                            image = draw_hand_skeleton(image, keypoints_2d, color, thickness=2, radius=3)
        
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
    parser.add_argument("--output_video", required=True, help="Path to output video file")
    parser.add_argument("--model_path", help="Path to pretrained model for predictions")
    parser.add_argument("--eval_results", help="Path to evaluation results .npy file")
    parser.add_argument("--show_gt", action="store_true", help="Show ground truth keypoints")
    parser.add_argument("--show_predictions", action="store_true", default=True, help="Show predicted keypoints")
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera index to visualize")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        logger.error(f"Input video not found: {args.input_video}")
        return
    
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
        camera_idx=args.camera_idx
    )


if __name__ == "__main__":
    main()
