# Hand Keypoint Visualization

This directory contains scripts for visualizing hand keypoints on video frames. The visualization overlays predicted and/or ground truth hand keypoints on the original video frames.

## Files

- `visualize_keypoints.py` - Main visualization script
- `example_visualize.py` - Example script showing different usage patterns
- `VISUALIZATION_README.md` - This documentation

## Features

- **Multiple visualization modes**:
  - Ground truth keypoints only
  - Predicted keypoints only (using pretrained model)
  - Evaluation results from .npy files
  - Combined visualizations

- **Hand skeleton rendering**:
  - Draws 21 keypoints per hand
  - Connects keypoints to show hand skeleton
  - Different colors for left/right hands
  - Different colors for GT vs predictions

- **Multi-camera support**:
  - **4-panel layout**: Show all 4 camera views side by side (default)
  - **Single camera**: Choose which camera view to visualize
  - Works with the multi-view UmeTrack data format
  - Maintains original video dimensions

## Usage

### Basic Usage

```bash
# Visualize ground truth keypoints on all 4 cameras (default)
python visualize_keypoints.py --input_video path/to/video.mp4 --output_video output.mp4 --show_gt

# Visualize using pretrained model on all 4 cameras
python visualize_keypoints.py --input_video path/to/video.mp4 --output_video output.mp4 --model_path pretrained_weights.torch

# Visualize using evaluation results on all 4 cameras
python visualize_keypoints.py --input_video path/to/video.mp4 --output_video output.mp4 --eval_results results.npy

# Visualize single camera only
python visualize_keypoints.py --input_video path/to/video.mp4 --output_video output.mp4 --show_gt --single_camera --camera_idx 0
```

### Advanced Usage

```bash
# Show both GT and predictions with different colors
python visualize_keypoints.py \
    --input_video video.mp4 \
    --output_video output.mp4 \
    --model_path pretrained_weights.torch \
    --show_gt \
    --show_predictions

# Visualize specific camera view
python visualize_keypoints.py \
    --input_video video.mp4 \
    --output_video output.mp4 \
    --camera_idx 1 \
    --show_gt

# Use evaluation results with debug logging
python visualize_keypoints.py \
    --input_video video.mp4 \
    --output_video output.mp4 \
    --eval_results results.npy \
    --log_level DEBUG
```

### Command Line Arguments

- `--input_video`: Path to input video file (required)
- `--output_video`: Path to output video file (required)
- `--model_path`: Path to pretrained model for predictions (optional)
- `--eval_results`: Path to evaluation results .npy file (optional)
- `--show_gt`: Show ground truth keypoints (flag)
- `--show_predictions`: Show predicted keypoints (default: True)
- `--camera_idx`: Camera index to visualize when using single camera mode (default: 0)
- `--single_camera`: Show only single camera view instead of all 4 cameras
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Color Scheme

- **Ground Truth**:
  - Left hand: Yellow (0, 255, 255)
  - Right hand: Magenta (255, 0, 255)

- **Predictions**:
  - Left hand: Green (0, 255, 0)
  - Right hand: Blue (255, 0, 0)

## Examples

### Example 1: Using Evaluation Results

If you have already run the evaluation script and have .npy result files:

```bash
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_13.mp4 \
    --output_video visualization_eval.mp4 \
    --eval_results tmp/eval_results_known_skeleton/real/separate_hand/testing/user_19/recording_13.npy
```

### Example 2: Using Pretrained Model

For real-time predictions using the pretrained model:

```bash
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_13.mp4 \
    --output_video visualization_model.mp4 \
    --model_path pretrained_models/pretrained_weights.torch
```

### Example 3: Ground Truth Only

To visualize only the ground truth annotations:

```bash
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_13.mp4 \
    --output_video visualization_gt.mp4 \
    --show_gt
```

### Example 4: Run All Examples

Use the example script to run multiple visualizations:

```bash
python example_visualize.py
```

## Requirements

- OpenCV (`pip install opencv-python`)
- NumPy
- PyTorch
- All UmeTrack dependencies

## Troubleshooting

### Common Issues

1. **"Invalid video file" error**: The video file might be corrupted. The script will skip invalid files gracefully.

2. **"No frames available"**: Check that the video file is valid and readable.

3. **"Model loading failed"**: Ensure the model path is correct and the model file exists.

4. **"Evaluation results not found"**: Run the evaluation script first to generate .npy result files.

### Debug Mode

Use `--log_level DEBUG` to get detailed logging information:

```bash
python visualize_keypoints.py --input_video video.mp4 --output_video output.mp4 --show_gt --log_level DEBUG
```

## Integration with Evaluation Scripts

The visualization script is designed to work seamlessly with the existing evaluation scripts:

1. Run evaluation: `python run_eval_known_skeleton.py`
2. Visualize results: `python visualize_keypoints.py --eval_results path/to/results.npy`

This workflow allows you to:
- Evaluate model performance
- Visualize the results to understand where the model succeeds/fails
- Compare predictions with ground truth visually
