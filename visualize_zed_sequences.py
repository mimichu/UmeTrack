#!/usr/bin/env python3
"""
Visualization script for ZED stereo image sequences with hand keypoints.

This script overlays hand keypoints on ZED stereo image sequences and saves the result as a video.
Adapted from visualize_keypoints.py to work with ImageSequencePoseStream.

Usage:
    python visualize_zed_sequences.py \
        --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left \
        --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_right \
        --json ~/Documents/ZED/zed_stereo_intr.json \
        --output output_video.mp4
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from image_sequence_pose_stream import ImageSequencePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_LANDMARKS_PER_HAND

logger = logging.getLogger(__name__)

# Colors for different hands
HAND_COLORS = {
    0: (0, 255, 0),    # Green for left hand
    1: (255, 0, 0),    # Blue for right hand
}

GT_COLORS = {
    0: (0, 255, 255),  # Yellow for left hand GT
    1: (255, 0, 255),  # Magenta for right hand GT
}

# Keypoint connections for hand skeleton visualization
HAND_CONNECTIONS = [
    # Thumb
    (0, 7), (7, 6),
    # Index finger
    (1, 10), (10, 9), (9, 8),
    # Middle finger
    (2, 13), (13, 12), (12, 11), 
    # Ring finger
    (3, 16), (16, 15), (15, 14),
    # Pinky
    (4, 19), (19, 18), (18, 17),
    # Palm connections
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
                      thickness: int = 2, radius: int = 3, show_indices: bool = False) -> np.ndarray:
    """
    Draw hand skeleton on the image.
    
    Args:
        image: Input image (BGR format)
        keypoints_2d: Array of shape (21, 2) with 2D keypoints
        color: Color for drawing (B, G, R)
        thickness: Line thickness
        radius: Circle radius for keypoints
        show_indices: Whether to show keypoint indices
        
    Returns:
        Image with skeleton drawn
    """
    if len(keypoints_2d) == 0:
        return image
    
    image = image.copy()
    
    # Draw connections (bones)
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            start_pt = tuple(keypoints_2d[start_idx].astype(int))
            end_pt = tuple(keypoints_2d[end_idx].astype(int))
            cv2.line(image, start_pt, end_pt, color, thickness)
    
    # Draw keypoints (joints)
    for idx, keypoint in enumerate(keypoints_2d):
        pt = tuple(keypoint.astype(int))
        cv2.circle(image, pt, radius, color, -1)
        
        if show_indices:
            # Draw keypoint index
            cv2.putText(image, str(idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, (255, 255, 255), 1)
    
    return image


def visualize_frame(frame_gray: np.ndarray, input_frame, gt_tracking, 
                   cam_idx: int = 0, show_gt: bool = True) -> np.ndarray:
    """
    Visualize hand keypoints on a single frame.
    
    Args:
        frame_gray: Grayscale frame to draw on
        input_frame: InputFrame with camera information
        gt_tracking: Ground truth tracking dict
        cam_idx: Camera index to visualize
        show_gt: Whether to show ground truth keypoints
        
    Returns:
        Frame with keypoints drawn (BGR for OpenCV)
    """
    # Convert grayscale to BGR for visualization (so we can draw colored keypoints)
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    camera = input_frame.views[cam_idx].camera
    
    # Draw ground truth if available
    if show_gt and gt_tracking:
        for hand_idx, gt_hand_pose in gt_tracking.items():
            # Get hand model from stream (need to pass it in)
            # For now, just show that hands are detected
            color = GT_COLORS.get(hand_idx, (255, 255, 255))
            text = f"Hand {hand_idx} detected"
            cv2.putText(frame_bgr, text, (10, 30 + hand_idx * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame_bgr


def main():
    parser = argparse.ArgumentParser(
        description='Visualize hand keypoints on ZED stereo image sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_zed_sequences.py \\
        --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left \\
        --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_right \\
        --json ~/Documents/ZED/zed_stereo_intr.json \\
        --output visualization.mp4 \\
        --camera 0
        """
    )
    
    parser.add_argument('--left-dir', required=True,
                       help='Directory with left camera images')
    parser.add_argument('--right-dir', required=True,
                       help='Directory with right camera images')
    parser.add_argument('--json', required=True,
                       help='JSON file with camera intrinsics')
    parser.add_argument('--output', '-o', required=True,
                       help='Output video file path')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to visualize (0=left, 1=right)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--show-gt', action='store_true',
                       help='Show ground truth keypoints')
    parser.add_argument('--show-indices', action='store_true',
                       help='Show keypoint indices')
    
    args = parser.parse_args()
    
    # Expand paths
    left_dir = os.path.expanduser(args.left_dir)
    right_dir = os.path.expanduser(args.right_dir)
    json_path = os.path.expanduser(args.json)
    output_path = os.path.expanduser(args.output)
    
    # Validate paths
    if not os.path.exists(left_dir):
        print(f"Error: Left directory not found: {left_dir}")
        return 1
    if not os.path.exists(right_dir):
        print(f"Error: Right directory not found: {right_dir}")
        return 1
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    print("Loading ZED image sequence...")
    print(f"  Left:  {left_dir}")
    print(f"  Right: {right_dir}")
    print(f"  JSON:  {json_path}")
    
    # Load image sequence stream
    stream = ImageSequencePoseStream(left_dir, right_dir, json_path)
    
    if len(stream) == 0:
        print("Error: No frames found in stream")
        return 1
    
    print(f"  Total frames: {len(stream)}")
    print(f"  Visualizing camera: {args.camera}")
    
    # Get first frame to determine video size
    first_frame_processed = False
    video_writer = None
    
    frame_count = 0
    max_frames = args.max_frames if args.max_frames else len(stream)
    
    print(f"\nProcessing frames...")
    
    try:
        for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
            if frame_idx >= max_frames:
                break
            
            # Get image from the specified camera
            if args.camera >= len(input_frame.views):
                print(f"Error: Camera {args.camera} not available (only {len(input_frame.views)} cameras)")
                return 1
            
            # Get the grayscale image from stream (UmeTrack uses grayscale)
            img_gray = input_frame.views[args.camera].image
            
            # Initialize video writer on first frame
            if not first_frame_processed:
                height, width = img_gray.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # VideoWriter for color output (BGR, 3 channels)
                video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))
                print(f"  Output: {output_path}")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {args.fps}")
                first_frame_processed = True
            
            # Visualize keypoints (converts grayscale to BGR internally)
            vis_frame = visualize_frame(img_gray, input_frame, gt_tracking, 
                                       args.camera, args.show_gt)
            
            # Add frame number
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            video_writer.write(vis_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count}/{max_frames} frames...")
    
    finally:
        if video_writer:
            video_writer.release()
    
    print(f"\n✓ Done! Processed {frame_count} frames")
    print(f"  Output saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

