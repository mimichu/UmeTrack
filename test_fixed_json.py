#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose

video_path = 'UmeTrack_data/raw_data/real/test_format/recording_00_fixed.mp4'
print(f'Testing: {video_path}')
stream = SyncedImagePoseStream(video_path)

for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
    if frame_idx > 0:
        break
    print(f'Number of cameras in video: {len(input_frame.views)}')
    hand_model = stream._hand_pose_labels.hand_model
    
    for hand_idx, gt_hand_pose in gt_tracking.items():
        landmarks_world = landmarks_from_hand_pose(hand_model, gt_hand_pose, hand_idx)
        camera = input_frame.views[0].camera
        print(f'\nHand {hand_idx}:')
        print(f'  Camera resolution: {camera.width}x{camera.height}')
        
        landmarks_eye = camera.world_to_eye(landmarks_world)
        landmarks_win = camera.eye_to_window(landmarks_eye)
        
        visible_mask = (
            (landmarks_win[..., 0] >= 0) &
            (landmarks_win[..., 0] <= camera.width - 1) &
            (landmarks_win[..., 1] >= 0) &
            (landmarks_win[..., 1] <= camera.height - 1) &
            (landmarks_eye[..., 2] > 0)
        )
        n_visible = visible_mask.sum()
        print(f'  Visible landmarks: {n_visible}/21')
        print(f'  Status: {"✓ PASS" if n_visible >= 10 else "✗ FAIL"}')

print('\n=== Done ===')

