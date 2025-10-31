# python3 visualize_3d_keypoints.py \
#     --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_left \
#     --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-07_right \
#     --json ~/Documents/ZED/zed_umetrack_full_format.json \
#     --model pretrained_models/pretrained_weights.torch

# python3 run_inference_zed.py \
#     --input-file /home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack/UmeTrack_data/raw_data/real/hand_hand/testing/user_05/recording_00.mp4 \
#     --model-path pretrained_models/pretrained_weights.torch 

python3 run_inference_zed.py \
    --input-file ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left \
    --model-path pretrained_models/pretrained_weights.torch \
    --json-path /home/chuerpan/Documents/ZED/zed_umetrack_full_format.json

