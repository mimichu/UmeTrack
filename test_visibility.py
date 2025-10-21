#!/usr/bin/env python3
"""
Debug script to check landmark visibility
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose
import numpy as np

video_path = "UmeTrack_data/raw_data/real/test_format/recording_00.mp4"

print(f"Loading video: {video_path}")
image_pose_stream = SyncedImagePoseStream(video_path)

# Get first frame
for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
    if frame_idx > 0:
        break
        
    print(f"\nFrame {frame_idx}:")
    print(f"  Number of cameras: {len(input_frame.views)}")
    
    hand_model = image_pose_stream._hand_pose_labels.hand_model
    
    for hand_idx, gt_hand_pose in gt_tracking.items():
        print(f"\n  Hand {hand_idx} (confidence: {gt_hand_pose.hand_confidence}):")
        
        # Get landmarks in world coordinates
        landmarks_world = landmarks_from_hand_pose(hand_model, gt_hand_pose, hand_idx)
        print(f"    Total landmarks: {len(landmarks_world)}")
        
        # Check visibility in each camera
        for cam_idx, view_data in enumerate(input_frame.views):
            camera = view_data.camera
            
            # Transform to camera space
            landmarks_eye = camera.world_to_eye(landmarks_world)
            landmarks_win = camera.eye_to_window(landmarks_eye)
            
            # Check which landmarks are visible
            visible_mask = (
                (landmarks_win[..., 0] >= 0)
                & (landmarks_win[..., 0] <= camera.width - 1)
                & (landmarks_win[..., 1] >= 0)
                & (landmarks_win[..., 1] <= camera.height - 1)
                & (landmarks_eye[..., 2] > 0)
            )
            
            n_visible = visible_mask.sum()
            
            print(f"    Camera {cam_idx}:")
            print(f"      Resolution: {camera.width}x{camera.height}")
            print(f"      Visible landmarks: {n_visible}/{len(landmarks_world)}")
            print(f"      Threshold for multi-view (19): {'PASS' if n_visible >= 19 else 'FAIL'}")
            print(f"      Threshold for single-view (10): {'PASS' if n_visible >= 10 else 'FAIL'}")
            
            if n_visible < 10:
                print(f"      Landmark positions (in camera):")
                for i, (lm, vis) in enumerate(zip(landmarks_win, visible_mask)):
                    if not vis:
                        print(f"        Landmark {i}: ({lm[0]:.1f}, {lm[1]:.1f}) - OUT OF BOUNDS")

print("\n=== Done ===")

