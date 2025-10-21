# Single-View Tracking Issue: Camera Calibration Mismatch

## Problem

When using `--max_views 1` on a pre-cropped single-camera MP4 file, the tracker does not generate predictions.

## Root Cause

The issue is **camera calibration mismatch**:

1. The original UmeTrack videos contain 4 camera views concatenated horizontally (2544x480 = 4 × 636x480)
2. Each camera has calibration data (intrinsics and extrinsics) stored in the `.json` file
3. When you crop the video to extract a single camera view, **the video changes but the JSON file's calibration data doesn't**
4. The tracker uses the JSON calibration to project 3D hand poses into 2D camera coordinates
5. With incorrect calibration, all projected landmarks fall **outside the image bounds** (e.g., y=497 when height=480)
6. The visibility check filters out cameras with too few visible landmarks
7. Result: No crop cameras generated → No predictions

### Example from `test_format/recording_00.mp4`:

```
Camera 0: Resolution: 636x480
Hand 0 landmarks:
  Landmark 0: (134.5, 497.5) - OUT OF BOUNDS  ← y > 480!
  Landmark 1: (240.2, 547.9) - OUT OF BOUNDS
  ...all 21 landmarks are out of bounds
  
Visible landmarks: 0/21
Required for tracking: 10/21 (single-view mode)
Result: Camera filtered out, no tracking possible
```

## Solutions

### Solution 1: Use Original Multi-Camera Video (RECOMMENDED)

**Don't crop the video beforehand**. Use the original 4-camera video with `--max_views 1`:

```bash
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/separate_hand/testing/user_19/recording_00.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 0 \
    --max_views 1
```

**How this works:**
- The video still contains all 4 camera views (2544x480)
- The JSON file has correct calibration for all 4 cameras
- `--max_views 1` tells the **tracker** to use only 1 view internally
- `--single_camera --camera_idx 0` tells the **visualizer** to show only camera 0 in output
- Result: You get single-view tracking with correct calibration!

### Solution 2: Create Corrected JSON for Cropped Video

If you must use a pre-cropped video, you need to update the JSON file:

1. Keep only the camera data for the extracted camera
2. Update camera intrinsics if the crop changed the image dimensions
3. Ensure extrinsics are correct for the standalone camera

This is complex and error-prone - **Solution 1 is strongly recommended**.

### Solution 3: Disable Calibration Checking (NOT RECOMMENDED)

You could modify the code to set `min_required_vis_landmarks=0`, but this would:
- Allow tracking with no visible landmarks (nonsensical)
- Likely produce garbage results
- Defeat the purpose of the visibility check

## Understanding the Parameters

### `--max_views N`
Controls **how many camera views the tracker uses internally** for pose estimation:
- Affects: Tracker's crop camera selection
- Does NOT affect: Video format or visualization

### `--single_camera --camera_idx N`
Controls **which camera view is shown in the output video**:
- Affects: Visualization only
- Does NOT affect: Tracking algorithm

### Typical Usage

**Multi-view tracking, show all cameras:**
```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions
    # max_views=2 (default), shows all 4 cameras
```

**Multi-view tracking, show single camera:**
```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1
    # max_views=2 (default), shows only camera 1
```

**Single-view tracking, show single camera:**
```bash
python visualize_keypoints.py \
    --input_video recording.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 1 \
    --max_views 1
    # max_views=1, shows only camera 1
```

## What Was Fixed

We added adaptive visibility thresholds:

- **Multi-view mode (`--max_views 2`)**: Requires 19/21 landmarks visible
- **Single-view mode (`--max_views 1`)**: Requires 10/21 landmarks visible

This helps when a single camera has limited view, but **doesn't solve calibration mismatch**.

## Verification

To verify your video/JSON combination is valid:

```bash
python test_visibility.py
```

Expected output for valid setup:
```
Camera 0:
  Visible landmarks: 19/21 (or at least 10/21)
  Threshold for single-view (10): PASS
```

If you see `Visible landmarks: 0/21`, your calibration is wrong.

## Recommended Workflow

1. **Start with original UmeTrack data** (multi-camera videos with correct JSON files)
2. **Use `--max_views 1`** to enable single-view tracking
3. **Use `--single_camera --camera_idx N`** to visualize specific camera
4. **Optionally crop output video** after visualization for final presentation

Do NOT crop input videos before tracking!

