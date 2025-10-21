#!/bin/bash

# Script to crop a single camera from a 1x4 horizontal panel video
# Usage: ./crop_camera.sh <input_video> [camera_idx] [output_video]
# Camera indices: 0 (leftmost), 1 (second from left), 2 (third), 3 (rightmost)

# Default values
CAMERA_IDX=1  # Second camera (index 1)

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video> [camera_idx] [output_video]"
    echo "  input_video:  Path to the input video file (1x4 horizontal panel)"
    echo "  camera_idx:   Camera index to extract (0-3, default: 1)"
    echo "  output_video: Path to output video (optional, auto-generated if not provided)"
    echo ""
    echo "Example: $0 recording_00.mp4"
    echo "Example: $0 recording_00.mp4 1"
    echo "Example: $0 recording_00.mp4 1 output.mp4"
    exit 1
fi

INPUT_VIDEO="$1"

# Check if input file exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video '$INPUT_VIDEO' not found"
    exit 1
fi

# Get camera index if provided
if [ $# -ge 2 ]; then
    CAMERA_IDX="$2"
fi

# Validate camera index
if [ "$CAMERA_IDX" -lt 0 ] || [ "$CAMERA_IDX" -gt 3 ]; then
    echo "Error: Camera index must be between 0 and 3"
    exit 1
fi

# Get video dimensions
VIDEO_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$INPUT_VIDEO")
TOTAL_WIDTH=$(echo "$VIDEO_INFO" | cut -d'x' -f1)
TOTAL_HEIGHT=$(echo "$VIDEO_INFO" | cut -d'x' -f2)

# Calculate single camera dimensions (assumes 1x4 horizontal layout)
SINGLE_WIDTH=$((TOTAL_WIDTH / 4))
SINGLE_HEIGHT=$TOTAL_HEIGHT

# Calculate crop offset for the selected camera
CROP_X=$((SINGLE_WIDTH * CAMERA_IDX))

echo "Video dimensions: ${TOTAL_WIDTH}x${TOTAL_HEIGHT}"
echo "Single camera dimensions: ${SINGLE_WIDTH}x${SINGLE_HEIGHT}"
echo "Cropping camera $CAMERA_IDX at offset x=$CROP_X"

# Generate output filename if not provided
if [ $# -ge 3 ]; then
    OUTPUT_VIDEO="$3"
else
    # Auto-generate output filename by adding _cam{idx} before .mp4
    DIR=$(dirname "$INPUT_VIDEO")
    FILENAME=$(basename "$INPUT_VIDEO" .mp4)
    OUTPUT_VIDEO="${DIR}/${FILENAME}_cam${CAMERA_IDX}.mp4"
fi

echo "Output video: $OUTPUT_VIDEO"

# Crop the video using ffmpeg
# crop filter: crop=width:height:x:y
ffmpeg -i "$INPUT_VIDEO" \
    -vf "crop=${SINGLE_WIDTH}:${SINGLE_HEIGHT}:${CROP_X}:0" \
    -c:v libx264 \
    -preset fast \
    -crf 18 \
    -c:a copy \
    "$OUTPUT_VIDEO" \
    -y

# Check if ffmpeg was successful
if [ $? -eq 0 ]; then
    echo "✓ Successfully cropped camera $CAMERA_IDX to: $OUTPUT_VIDEO"
else
    echo "✗ Failed to crop video"
    exit 1
fi

