#!/usr/bin/env python3
"""
Example script showing how to use the keypoint visualization.

This script demonstrates different ways to visualize hand keypoints:
1. Using evaluation results (.npy files)
2. Using the pretrained model for real-time predictions
3. Showing both ground truth and predictions
"""

import os
import subprocess
import sys

def run_visualization_example():
    """Run example visualizations with different options."""
    
    # Paths
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "UmeTrack_data", "raw_data", "real")
    model_path = os.path.join(script_dir, "pretrained_models", "pretrained_weights.torch")
    eval_results_dir = os.path.join(script_dir, "tmp", "eval_results_known_skeleton", "real")
    
    # Find a sample video
    sample_videos = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                sample_videos.append(os.path.join(root, file))
                break
        if sample_videos:
            break
    
    if not sample_videos:
        print("No sample videos found in the dataset!")
        return
    
    sample_video = sample_videos[0]
    print(f"Using sample video: {sample_video}")
    
    # Example 1: Visualize using evaluation results
    print("\n=== Example 1: Visualize using evaluation results ===")
    video_name = os.path.splitext(os.path.basename(sample_video))[0]
    video_dir = os.path.dirname(sample_video)
    relative_path = os.path.relpath(video_dir, data_dir)
    eval_result_path = os.path.join(eval_results_dir, relative_path, f"{video_name}.npy")
    
    if os.path.exists(eval_result_path):
        output_path = f"visualization_eval_{video_name}.mp4"
        cmd = [
            sys.executable, "visualize_keypoints.py",
            "--input_video", sample_video,
            "--output_video", output_path,
            "--eval_results", eval_result_path,
            "--show_predictions"
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Created visualization: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create evaluation visualization: {e}")
    else:
        print(f"⚠️  Evaluation results not found: {eval_result_path}")
        print("   Run the evaluation script first to generate results.")
    
    # Example 2: Visualize using pretrained model
    print("\n=== Example 2: Visualize using pretrained model ===")
    if os.path.exists(model_path):
        output_path = f"visualization_model_{video_name}.mp4"
        cmd = [
            sys.executable, "visualize_keypoints.py",
            "--input_video", sample_video,
            "--output_video", output_path,
            "--model_path", model_path,
            "--show_predictions"
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Created visualization: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create model visualization: {e}")
    else:
        print(f"⚠️  Pretrained model not found: {model_path}")
    
    # Example 3: Visualize ground truth only
    print("\n=== Example 3: Visualize ground truth only ===")
    output_path = f"visualization_gt_{video_name}.mp4"
    cmd = [
        sys.executable, "visualize_keypoints.py",
        "--input_video", sample_video,
        "--output_video", output_path,
        "--show_gt"
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Created visualization: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create GT visualization: {e}")
    
    print("\n=== Visualization Complete ===")
    print("Check the generated .mp4 files to see the keypoint visualizations!")

if __name__ == "__main__":
    run_visualization_example()
