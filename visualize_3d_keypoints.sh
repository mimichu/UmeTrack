# python3 visualize_3d_keypoints.py \
#     --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_left \
#     --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_right \
#     --json ~/Documents/ZED/zed_umetrack_full_format.json \
#     --model pretrained_models/pretrained_weights.torch

# python3 /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/run_inference_zed.py \
#     --input-file /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/hand_hand/testing/user_05/recording_00.mp4 \
#     --model-path pretrained_models/pretrained_weights.torch 

# python3 run_inference_zed.py \
#     --input-file ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left \
#     --model-path pretrained_models/pretrained_weights.torch \
#     --json-path /home/chuerpan/Documents/ZED/zed_umetrack_full_format.json

# python visualize_3d_keypoints.py \
#     --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_left \
#     --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_right \
#     --json ~/Documents/ZED/zed_umetrack_full_format.json \
#     --predictions ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left.npy \
#     --model pretrained_models/pretrained_weights.torch

# python visualize_3d_keypoints.py \
#     --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \
#     --predictions /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/tmp/eval_results_known_skeleton/real/hand_hand/testing/user_05/recording_00.npy

# python visualize_3d_keypoints.py \
#     --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \
#     --model pretrained_models/pretrained_weights.torch \
#     --generic-hand-model dataset/generic_hand_model.json

svo_name="real_separate_hand_testing_user_19_recording_01_cam1_images"
frame_idx="000000"

python /home/chuerpan/repo/InteractionRetarget/submodules/FoundationStereo/scripts/run_demo.py \
    --left_file /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_01_cam1_images/recording_01_cam1_images_${frame_idx}_rectified.png \
    --right_file /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_01_cam1_images/recording_01_cam2_images_${frame_idx}_rectified.png \
    --intrinsic_file /home/chuerpan/repo/InteractionRetarget/cam_configs/umetrack/recording_01_intrinsics.txt \
    --ckpt_dir /home/chuerpan/repo/InteractionRetarget/submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --out_dir /home/chuerpan/repo/InteractionRetarget/submodules/FoundationStereo/test_outputs/UmeTrack/${svo_name}_${frame_idx}/ \
    --hiera 1 \
    --remove_invisible 0 