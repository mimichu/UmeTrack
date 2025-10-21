# Example: Show only left hand with cycling keypoint indices (multi-view)
# python visualize_keypoints.py --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/hand_hand/training/user_01/recording_01.mp4 --model_path pretrained_models/pretrained_weights.torch --show_predictions --show_indices --hand_filter left --single_camera --camera_idx 1

# Example: Use single-view tracking (--max_views 1)
# python visualize_keypoints.py --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/test_format/recording_00.mp4 --model_path pretrained_models/pretrained_weights.torch --show_predictions --single_camera --camera_idx 1 --max_views 1

# Current command: multi-view tracking (default --max_views 2)
# python visualize_keypoints.py --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/test_format/recording_00.mp4 --model_path pretrained_models/pretrained_weights.torch --show_gt

# python visualize_keypoints.py \
#     --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/test_format_pane_singlev/recording_00.mp4  \
#     --model_path pretrained_models/pretrained_weights.torch \
#     --show_predictions \
#     --single_camera --camera_idx 1 \
#     --max_views 1

# CORRECT: Use multi-camera video with camera_idx 1 (which actually sees the hands!)
# Camera 0 and 3 don't see hands, use camera 1 or 2 instead
# python visualize_keypoints.py \
#     --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_00.mp4  \
#     --model_path pretrained_models/pretrained_weights.torch \
#     --show_predictions \
#     --single_camera --camera_idx 1 \
#     --max_views 1

# WRONG: This video is cropped (636x480) and has incorrect calibration!
# WORKING: Single-camera video from camera 1 (which sees the hands!)
python visualize_keypoints.py \
    --input_video /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/test_format/recording_00_cam1.mp4  \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 0 \
    --max_views 1