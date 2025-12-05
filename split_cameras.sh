#!/bin/bash

# Script to split a 1x4 horizontal panel video into individual camera MP4s.
# Usage: ./split_cameras.sh <input_video> [output_directory]
# Example:
#   ./split_cameras.sh recording_01.mp4
#   ./split_cameras.sh recording_01.mp4 /tmp/output

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video> [output_directory]"
    echo "  input_video:       Path to the 1x4 horizontal panel video"
    echo "  output_directory:  Directory to save cropped videos (defaults to input video directory)"
    exit 1
fi

INPUT_VIDEO="$1"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video '$INPUT_VIDEO' not found" >&2
    exit 1
fi

if [ $# -ge 2 ]; then
    OUTPUT_DIR="$2"
else
    OUTPUT_DIR="$(dirname "$INPUT_VIDEO")"
fi

mkdir -p "$OUTPUT_DIR"

# Extract video dimensions using ffprobe
VIDEO_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$INPUT_VIDEO")
if [ -z "$VIDEO_INFO" ]; then
    echo "Error: Unable to determine video dimensions for '$INPUT_VIDEO'" >&2
    exit 1
fi

TOTAL_WIDTH=$(echo "$VIDEO_INFO" | cut -d'x' -f1)
TOTAL_HEIGHT=$(echo "$VIDEO_INFO" | cut -d'x' -f2)

if [ -z "$TOTAL_WIDTH" ] || [ -z "$TOTAL_HEIGHT" ]; then
    echo "Error: Invalid width/height extracted from video metadata: '$VIDEO_INFO'" >&2
    exit 1
fi

SINGLE_WIDTH=$((TOTAL_WIDTH / 4))
SINGLE_HEIGHT=$TOTAL_HEIGHT

echo "Splitting '$INPUT_VIDEO'"
echo " - Total resolution: ${TOTAL_WIDTH}x${TOTAL_HEIGHT}"
echo " - Single camera resolution: ${SINGLE_WIDTH}x${SINGLE_HEIGHT}"
echo " - Output directory: $OUTPUT_DIR"

BASENAME="$(basename "$INPUT_VIDEO" .mp4)"

for CAMERA_IDX in 0 1 2 3; do
    CROP_X=$((SINGLE_WIDTH * CAMERA_IDX))
    OUTPUT_VIDEO="${OUTPUT_DIR}/${BASENAME}_cam${CAMERA_IDX}.mp4"

    echo "Cropping camera ${CAMERA_IDX} -> ${OUTPUT_VIDEO}"

    ffmpeg -v error -i "$INPUT_VIDEO" \
        -vf "crop=${SINGLE_WIDTH}:${SINGLE_HEIGHT}:${CROP_X}:0" \
        -c:v libx264 \
        -preset fast \
        -crf 18 \
        -c:a copy \
        -y "$OUTPUT_VIDEO"

    if [ $? -eq 0 ]; then
        echo " ✓ Camera ${CAMERA_IDX} saved to ${OUTPUT_VIDEO}"
    else
        echo " ✗ Failed to crop camera ${CAMERA_IDX}" >&2
        exit 1
    fi
done

echo "Done."

