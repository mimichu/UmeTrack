# ✅ Working Solution: True Single-Camera Video Tracking

## Yes, Single-Camera Videos DO Work!

You asked: **"If I only have a video from one camera, would that not work?"**

**Answer: YES, it works!** But you need:
1. A video from a camera that **actually sees the hands**
2. Correct JSON calibration for that specific camera

## The Problem With Your Original Setup

Your `test_format/recording_00.mp4` was from **camera 0**, which doesn't see the hands:
- Camera 0: 0/21 landmarks visible ❌ **Empty view!**
- Camera 1: 21/21 landmarks visible ✅ **Perfect view!**

## ✅ Working Single-Camera Solution Created

I've created a proper single-camera setup for you:

### Files Created:
1. **Video**: `UmeTrack_data/raw_data/real/test_format/recording_00_cam1.mp4`
   - Cropped from camera 1 (which sees the hands)
   - Size: 636x480 (single camera)

2. **JSON**: `UmeTrack_data/raw_data/real/test_format/recording_00_cam1.json`
   - Contains only camera 1's calibration
   - Properly formatted for single-camera input

### Verification:
```
Hand 0: 21/21 landmarks visible ✓ PASS
Hand 1: 18/21 landmarks visible ✓ PASS
```

## How to Use

Your updated `visualize.sh` now uses this working single-camera video:

```bash
python visualize_keypoints.py \
    --input_video UmeTrack_data/raw_data/real/test_format/recording_00_cam1.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 0 \
    --max_views 1
```

Just run:
```bash
./visualize.sh
```

**This will work!** The video is single-camera (636x480), has correct calibration, and the camera sees the hands.

## How to Create Single-Camera Videos from Your Own Data

If you have a multi-camera video and want to create a single-camera version:

### Step 1: Find which camera sees the hands

```bash
python -c "
import sys; sys.path.append('lib')
from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose

stream = SyncedImagePoseStream('your_video.mp4')
for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
    if frame_idx > 0: break
    hand_model = stream._hand_pose_labels.hand_model
    for cam_idx, view in enumerate(input_frame.views):
        print(f'Camera {cam_idx}:', end=' ')
        total_visible = 0
        for hand_idx, pose in gt_tracking.items():
            landmarks_world = landmarks_from_hand_pose(hand_model, pose, hand_idx)
            landmarks_eye = view.camera.world_to_eye(landmarks_world)
            landmarks_win = view.camera.eye_to_window(landmarks_eye)
            visible = ((landmarks_win[..., 0] >= 0) & 
                      (landmarks_win[..., 0] <= view.camera.width - 1) &
                      (landmarks_win[..., 1] >= 0) & 
                      (landmarks_win[..., 1] <= view.camera.height - 1) &
                      (landmarks_eye[..., 2] > 0)).sum()
            total_visible += visible
        print(f'{total_visible} total landmarks visible')
"
```

### Step 2: Crop the good camera

```bash
# Replace CAMERA_IDX with the camera that sees hands (usually 1 or 2)
./crop_camera.sh input_video.mp4 CAMERA_IDX output_video.mp4
```

### Step 3: Create matching JSON

```bash
python create_single_camera_json.py \
    --input_json input_video.json \
    --output_json output_video.json \
    --camera_idx CAMERA_IDX
```

### Step 4: Use the pair together

```bash
python visualize_keypoints.py \
    --input_video output_video.mp4 \
    --model_path pretrained_models/pretrained_weights.torch \
    --show_predictions \
    --single_camera --camera_idx 0 \
    --max_views 1
```

## Key Points

✅ **Single-camera videos DO work** with proper calibration
✅ **Camera selection matters** - use a camera that sees the hands
✅ **JSON must match video** - use `create_single_camera_json.py` to ensure consistency
❌ **Camera 0 often doesn't see hands** in UmeTrack datasets
❌ **Don't use mismatched video/JSON pairs**

## Two Valid Approaches

### Approach 1: Multi-Camera Video (Simpler)
- Use original multi-camera video
- Set `--max_views 1` for single-view tracking
- Set `--single_camera --camera_idx N` for visualization
- No preprocessing needed!

### Approach 2: Single-Camera Video (What you wanted)
- Crop a camera that sees hands (usually cam 1 or 2)
- Create matching JSON with `create_single_camera_json.py`
- Use the video+JSON pair together
- Requires preprocessing but gives you a standalone single-camera file

Both work! Choose based on your workflow.

