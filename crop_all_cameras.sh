#!/bin/bash

# Script to crop a single camera from all 1x4 horizontal panel videos in raw_data
# Usage: ./crop_all_cameras.sh [camera_idx]
# Camera indices: 0 (leftmost), 1 (second from left), 2 (third), 3 (rightmost)

# Default camera index
CAMERA_IDX=1  # Second camera (index 1)

# Parse arguments
if [ $# -ge 1 ]; then
    CAMERA_IDX="$1"
fi

# Validate camera index
if [ "$CAMERA_IDX" -lt 0 ] || [ "$CAMERA_IDX" -gt 3 ]; then
    echo "Error: Camera index must be between 0 and 3"
    exit 1
fi

# Set the base directory
BASE_DIR="/home/chuerpan/repo/InteractionRetarget/submodules/UmeTrack"
RAW_DATA_DIR="$BASE_DIR/UmeTrack_data/raw_data"

# Counter for processed files
count=0
success_count=0
fail_count=0

# Get total number of MP4 files
total_files=$(find "$RAW_DATA_DIR" -name "*.mp4" -not -name "*_cam*.mp4" | wc -l)

echo "=========================================="
echo "Cropping camera $CAMERA_IDX from all videos"
echo "Found $total_files MP4 files to process"
echo "=========================================="
echo ""

# Find all MP4 files (excluding already cropped files with _cam in name) and process them
find "$RAW_DATA_DIR" -name "*.mp4" -not -name "*_cam*.mp4" | while read -r video_file; do
    count=$((count + 1))
    
    # Generate output filename
    dir=$(dirname "$video_file")
    filename=$(basename "$video_file" .mp4)
    output_file="${dir}/${filename}_cam${CAMERA_IDX}.mp4"
    
    # Skip if output file already exists
    if [ -f "$output_file" ]; then
        echo "[$count/$total_files] Skipping (already exists): $video_file"
        continue
    fi
    
    echo "[$count/$total_files] Processing: $video_file"
    
    # Get video dimensions
    video_info=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video_file" 2>/dev/null)
    
    if [ -z "$video_info" ]; then
        echo "  ✗ Failed to read video info"
        fail_count=$((fail_count + 1))
        continue
    fi
    
    total_width=$(echo "$video_info" | cut -d'x' -f1)
    total_height=$(echo "$video_info" | cut -d'x' -f2)
    
    # Calculate single camera dimensions (assumes 1x4 horizontal layout)
    single_width=$((total_width / 4))
    single_height=$total_height
    
    # Calculate crop offset for the selected camera
    crop_x=$((single_width * CAMERA_IDX))
    
    # Crop the video using ffmpeg (suppress most output)
    ffmpeg -i "$video_file" \
        -vf "crop=${single_width}:${single_height}:${crop_x}:0" \
        -c:v libx264 \
        -preset fast \
        -crf 18 \
        -c:a copy \
        "$output_file" \
        -y \
        -loglevel error
    
    # Check if ffmpeg was successful
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully cropped to: $output_file"
        success_count=$((success_count + 1))
    else
        echo "  ✗ Failed to crop video"
        fail_count=$((fail_count + 1))
    fi
    
    echo ""
done

echo "=========================================="
echo "Processing complete!"
echo "Total files: $total_files"
echo "Successfully processed: $success_count"
echo "Failed: $fail_count"
echo "=========================================="

