#!/usr/bin/env python3
"""
Debug script to test single-view tracking
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.tracker import HandTracker, HandTrackerOpts
from lib.models.model_loader import load_pretrained_model

video_path = "UmeTrack_data/raw_data/real/test_format/recording_00.mp4"
model_path = "pretrained_models/pretrained_weights.torch"

print(f"Loading video: {video_path}")
image_pose_stream = SyncedImagePoseStream(video_path)
print(f"Video loaded, length: {len(image_pose_stream)}")

print(f"\nLoading model: {model_path}")
model = load_pretrained_model(model_path)
model.eval()

print("\n=== Testing with max_views=1 ===")
tracker_opts = HandTrackerOpts(max_view_num=1)
tracker = HandTracker(model, tracker_opts)

# Get first frame
for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
    if frame_idx > 0:
        break
        
    print(f"\nFrame {frame_idx}:")
    print(f"  Number of views in input_frame: {len(input_frame.views)}")
    print(f"  GT tracking hands: {list(gt_tracking.keys())}")
    
    if gt_tracking:
        hand_model = image_pose_stream._hand_pose_labels.hand_model
        
        # Try to generate crop cameras
        print(f"\n  Calling gen_crop_cameras with:")
        print(f"    - num cameras: {len([v.camera for v in input_frame.views])}")
        print(f"    - max_view_num: {tracker._max_view_num}")
        print(f"    - min_num_crops: 1")
        
        crop_cameras = tracker.gen_crop_cameras(
            [v.camera for v in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            hand_model,
            gt_tracking,
            min_num_crops=1,
        )
        
        print(f"\n  Result:")
        print(f"    crop_cameras: {crop_cameras}")
        print(f"    Number of hands with crop cameras: {len(crop_cameras)}")
        
        if crop_cameras:
            for hand_idx, hand_crops in crop_cameras.items():
                print(f"    Hand {hand_idx}: {len(hand_crops)} crop cameras")
                print(f"      Camera indices: {list(hand_crops.keys())}")
        else:
            print("    WARNING: No crop cameras generated!")
            
            # Debug: Check hand confidence
            for hand_idx, pose in gt_tracking.items():
                print(f"\n    Debug for hand {hand_idx}:")
                print(f"      Hand confidence: {pose.hand_confidence}")
                print(f"      Confidence threshold: 0.5")

print("\n=== Done ===")

