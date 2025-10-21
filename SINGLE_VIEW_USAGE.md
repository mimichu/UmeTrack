# Single-View vs Multi-View Tracking

## Overview

UmeTrack now supports configurable single-view and multi-view tracking modes via the `--max_views` parameter.

## Usage

### Multi-View Tracking (Default)

By default, UmeTrack uses 2 camera views for more robust tracking:

```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --max_views 2  # Default: uses 2 camera views
```

### Single-View Tracking

To use only 1 camera view (faster but potentially less accurate):

```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --max_views 1  # Single-view mode
```

### Using More Views

You can also use 3 or 4 views if your data has more cameras:

```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --max_views 4  # Use all 4 available camera views
```

## Command-Line Parameter

```
--max_views {1,2,3,4}
    Maximum number of camera views to use for tracking
    - 1: Single-view mode (fastest, uses only 1 camera)
    - 2: Multi-view mode (default, uses 2 cameras)
    - 3: Uses 3 cameras
    - 4: Uses all 4 cameras
```

## Output Filename Convention

When predictions are shown, the output filename automatically includes the view count:

- Multi-view (2 views): `recording_pred_2v_4cam.mp4`
- Single-view (1 view): `recording_pred_1v_4cam.mp4`

## Example Commands

### Compare Single-View vs Multi-View on the Same Video

```bash
# Multi-view tracking (more accurate)
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/test_format/recording_00.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera \
    --camera_idx 1 \
    --max_views 2

# Single-view tracking (faster)
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/test_format/recording_00.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera \
    --camera_idx 1 \
    --max_views 1
```

## Technical Details

### How It Works

The `--max_views` parameter controls how many camera views the tracker uses internally:

1. **Camera Selection**: The tracker ranks all available cameras based on hand visibility
2. **View Limitation**: It then selects up to `max_views` cameras (the best ones)
3. **Tracking**: The model processes these views to estimate hand pose

### Performance vs Accuracy Trade-off

- **Single-view (`--max_views 1`)**:
  - ✓ Faster processing
  - ✓ Lower memory usage
  - ✗ Potentially less accurate (especially with occlusions)

- **Multi-view (`--max_views 2`, default)**:
  - ✓ More robust tracking
  - ✓ Better handling of occlusions
  - ✗ Slightly slower processing
  - ✗ Higher memory usage

### Code Changes

The implementation modifies:
- `lib/tracker/tracker.py`: Added `max_view_num` parameter to `HandTrackerOpts`
- `visualize_keypoints.py`: Added `--max_views` command-line argument
- The tracker now uses `self._max_view_num` instead of the hardcoded `MAX_VIEW_NUM` constant

## Notes

- The paper shows that multi-view tracking (2+ views) provides significantly better accuracy
- Single-view mode is useful for:
  - Quick testing and prototyping
  - Resource-constrained environments
  - Cases where only one camera view is available
- For production use, multi-view mode (default) is recommended

