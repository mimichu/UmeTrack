#!/usr/bin/env python3
"""
Create a corrected JSON file for a single-camera cropped video.
This extracts one camera's calibration from a multi-camera JSON file.

Usage:
  python create_single_camera_json.py \
    --input_json UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_00.json \
    --output_json UmeTrack_data/raw_data/real/test_format/recording_00.json \
    --camera_idx 0
"""

import json
import argparse
import numpy as np

def create_single_camera_json(input_json_path, output_json_path, camera_idx):
    """Extract single camera data from multi-camera JSON."""
    
    print(f"Loading multi-camera JSON: {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    num_cameras = len(data['cameras'])
    print(f"Original cameras: {num_cameras}")
    print(f"Extracting camera {camera_idx}")
    
    if camera_idx >= num_cameras:
        raise ValueError(f"Camera index {camera_idx} out of range (0-{num_cameras-1})")
    
    # Create new data with only selected camera
    new_data = {
        'cameras': [data['cameras'][camera_idx]],
        'camera_angles': [data['camera_angles'][camera_idx]],
        'hand_model': data['hand_model'],
        'joint_angles': data['joint_angles'],
        'wrist_transforms': data['wrist_transforms'],
        'hand_confidences': data['hand_confidences'],
    }
    
    # Extract only the selected camera's transforms for each frame
    camera_to_world_transforms = np.array(data['camera_to_world_transforms'])
    print(f"Original transforms shape: {camera_to_world_transforms.shape}")
    
    # Select only the specified camera's transform for each frame
    # Shape: (num_frames, num_cameras, 4, 4) -> (num_frames, 1, 4, 4)
    single_camera_transforms = camera_to_world_transforms[:, camera_idx:camera_idx+1, :, :]
    print(f"New transforms shape: {single_camera_transforms.shape}")
    
    new_data['camera_to_world_transforms'] = single_camera_transforms.tolist()
    
    # Save new JSON
    print(f"Saving single-camera JSON: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print("✓ Done!")
    print(f"\nNow you can use this video+JSON pair:")
    print(f"  Video: {output_json_path.replace('.json', '.mp4')}")
    print(f"  JSON:  {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Create single-camera JSON from multi-camera JSON")
    parser.add_argument('--input_json', required=True, help="Path to multi-camera JSON file")
    parser.add_argument('--output_json', required=True, help="Path to output single-camera JSON file")
    parser.add_argument('--camera_idx', type=int, default=0, help="Camera index to extract (0-3)")
    
    args = parser.parse_args()
    
    create_single_camera_json(args.input_json, args.output_json, args.camera_idx)

if __name__ == '__main__':
    main()


