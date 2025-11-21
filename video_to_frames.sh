#!/bin/bash

# Script to extract frames from an MP4 into a numbered PNG sequence.
# Usage: ./video_to_frames.sh <input_video> [output_directory]
# Example:
#   ./video_to_frames.sh recording_01_cam0.mp4
#   ./video_to_frames.sh recording_01_cam0.mp4 /tmp/frames

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video> [output_directory]"
    echo "  input_video:       Path to the MP4 video file"
    echo "  output_directory:  Directory for extracted frames (defaults to basename + _images)"
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
    BASENAME="$(basename "$INPUT_VIDEO" .mp4)"
    OUTPUT_DIR="$(dirname "$INPUT_VIDEO")/${BASENAME}_images"
fi

mkdir -p "$OUTPUT_DIR"

echo "Extracting frames from '$INPUT_VIDEO' -> '$OUTPUT_DIR'"

ffmpeg -v error -i "$INPUT_VIDEO" \
    -start_number 0 \
    -vsync 0 \
    "${OUTPUT_DIR}/%06d.png"

if [ $? -eq 0 ]; then
    echo " ✓ Frames saved to ${OUTPUT_DIR}"
else
    echo " ✗ Failed to extract frames" >&2
    exit 1
fi

echo "Done."

