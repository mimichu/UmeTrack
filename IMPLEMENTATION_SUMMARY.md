# Implementation Summary: Configurable Single-View vs Multi-View Tracking

## Changes Made

### 1. Modified `lib/tracker/tracker.py`

**Added `max_view_num` parameter to `HandTrackerOpts`:**
```python
@dataclass
class HandTrackerOpts:
    num_crop_points: int = 63
    enable_memory: bool = True
    use_stored_pose_for_crop: bool = True
    hand_ratio_in_crop: float = 0.95
    min_required_vis_landmarks: int = 19
    max_view_num: int = 2  # NEW: Maximum number of camera views to use
```

**Updated `HandTracker.__init__` to use the parameter:**
```python
self._max_view_num: int = opts.max_view_num
```

**Modified `gen_crop_cameras` to use instance variable instead of constant:**
```python
crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
    # ... other params ...
    max_view_num=self._max_view_num,  # Changed from MAX_VIEW_NUM
    # ... other params ...
)
```

### 2. Modified `visualize_keypoints.py`

**Added `max_views` parameter to function signature:**
```python
def visualize_video_with_keypoints(
    # ... existing params ...
    max_views: int = 2  # NEW parameter
):
```

**Updated tracker initialization:**
```python
tracker_opts = HandTrackerOpts(max_view_num=max_views)
tracker = HandTracker(model, tracker_opts)
logger.info(f"Using max_views={max_views} ({'single-view' if max_views == 1 else 'multi-view'})")
```

**Added command-line argument:**
```python
parser.add_argument("--max_views", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Maximum number of camera views to use for tracking (1=single-view, 2=multi-view, default: 2)")
```

**Updated output filename generation:**
```python
if args.show_predictions:
    suffix_parts.append("pred")
    suffix_parts.append(f"{args.max_views}v")  # Adds "1v" or "2v" etc.
```

**Updated function call in main():**
```python
visualize_video_with_keypoints(
    # ... other params ...
    max_views=args.max_views
)
```

### 3. Updated `visualize.sh`

Added examples demonstrating the new parameter:
```bash
# Example: Use single-view tracking (--max_views 1)
# python visualize_keypoints.py ... --max_views 1

# Current command: multi-view tracking (default --max_views 2)
python visualize_keypoints.py ...
```

### 4. Created Documentation

- `SINGLE_VIEW_USAGE.md`: Comprehensive usage guide
- `IMPLEMENTATION_SUMMARY.md`: This file

## Usage Examples

### Multi-View (Default)
```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1
```

Output: `recording_pred_2v_cam1.mp4`

### Single-View
```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1 \
    --max_views 1
```

Output: `recording_pred_1v_cam1.mp4`

## Benefits

1. **Flexibility**: Users can now choose between speed (single-view) and accuracy (multi-view)
2. **Backwards Compatible**: Default behavior (2 views) is preserved
3. **Clear Naming**: Output files clearly indicate the tracking mode used
4. **Extensible**: Can easily support 3 or 4 views if needed

## Testing

Verify the implementation:
```bash
# Check that the argument is recognized
python visualize_keypoints.py --help | grep -A 2 "max_views"

# Test single-view mode
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/test_format/recording_00.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1 \
    --max_views 1

# Test multi-view mode (default)
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/test_format/recording_00.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1 \
    --max_views 2
```

## Technical Notes

- The tracker selects the best camera views based on hand visibility ranking
- With `--max_views 1`, only the single best camera is used
- With `--max_views 2`, the two best cameras are used
- This affects tracking accuracy but not the visualization output format
- The `--single_camera` flag only affects visualization (what you see in the output video)
- The `--max_views` flag affects tracking (how many views the model uses internally)

## Files Modified

1. `/lib/tracker/tracker.py` - Core tracker implementation
2. `/visualize_keypoints.py` - Visualization script
3. `/visualize.sh` - Example usage script
4. `/SINGLE_VIEW_USAGE.md` - User documentation (NEW)
5. `/IMPLEMENTATION_SUMMARY.md` - Implementation details (NEW)

