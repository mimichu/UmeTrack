#!/bin/bash

# Script to visualize keypoints for all MP4 files in raw_data directory
# Usage: ./visualize_all.sh

# Set the base directory
BASE_DIR="/home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack"
RAW_DATA_DIR="$BASE_DIR/UmeTrack_data/raw_data"
MODEL_PATH="pretrained_models/pretrained_weights.torch"

# Check if model exists
if [ ! -f "$BASE_DIR/$MODEL_PATH" ]; then
    echo "Error: Model file not found at $BASE_DIR/$MODEL_PATH"
    exit 1
fi

# Counter for processed files
count=0
total_files=$(find "$RAW_DATA_DIR" -name "*.mp4" | wc -l)

echo "Found $total_files MP4 files to process"
echo "Starting visualization..."

# Find all MP4 files and process them
find "$RAW_DATA_DIR" -name "*.mp4" | while read -r video_file; do
    count=$((count + 1))
    echo "[$count/$total_files] Processing: $video_file"
    
    # Run visualization
    python "$BASE_DIR/visualize_keypoints.py" \
        --input_video "$video_file" \
        --model_path "$MODEL_PATH" \
        --show_predictions \
        --single_camera \
        --camera_idx 1
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $video_file"
    else
        echo "✗ Failed to process: $video_file"
    fi
    
    echo "----------------------------------------"
done

echo "All files processed!"
