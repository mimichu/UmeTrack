#!/usr/bin/env python3
"""
3D Visualization of hand keypoints using viser.

This script visualizes predicted 3D hand keypoints in an interactive 3D viewer.
It can show ground truth, predictions, or both, along with camera poses.

Usage:
    # To run with live tracking:
    python visualize_3d_keypoints.py \
        --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \
        --model pretrained_models/pretrained_weights.torch \
        --generic-hand-model dataset/generic_hand_model.json

    # To visualize pre-computed predictions:
    python visualize_3d_keypoints.py \
        --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \
        --predictions tmp/eval_results_unknown_skeleton/real/hand_hand/training/user_00/recording_00.npy
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

try:
    import viser
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    print("ERROR: viser not installed. Install with: pip install viser")
    sys.exit(1)

try:
    from image_sequence_pose_stream import ImageSequencePoseStream
    IMAGE_SEQUENCE_AVAILABLE = True
except ImportError:
    IMAGE_SEQUENCE_AVAILABLE = False

# Imports from your working script
from lib.tracker.video_pose_data import (
    SyncedImagePoseStream, 
    _load_json, 
    load_hand_model_from_dict
)
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import (
    NUM_LANDMARKS_PER_HAND, 
    HandModel, 
    scaled_hand_model
)

# Optional: Tracker support
try:
    from lib.tracker.tracker import HandTracker, HandTrackerOpts
    from lib.models.model_loader import load_pretrained_model
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Hand skeleton connections (same as 2D visualization)
HAND_CONNECTIONS = [
    # Thumb
    (0, 7), (7, 6),
    # Index finger
    (1, 10), (10, 9), (9, 8),
    # Middle finger
    (2, 13), (13, 12), (12, 11), 
    # Ring finger
    (3, 16), (16, 15), (15, 14),
    # Pinky
    (4, 19), (19, 18), (18, 17),
    # Palm connections
    (17, 5), (5, 6), (6, 8), (8, 11), (11, 14), (14, 17), (5, 20), (20, 11)
]

# Colors for different hands (RGB)
HAND_COLORS = {
    0: (0.0, 1.0, 0.0),    # Green for left hand
    1: (0.0, 0.0, 1.0),    # Blue for right hand
}

# Colors for ground truth vs prediction
GT_COLOR_MODIFIER = (1.0, 1.0, 0.0)  # Add yellow tint for GT
PRED_COLOR_BASE = (1.0, 1.0, 1.0)     # White for predictions


class HandVisualization3D:
    """Interactive 3D visualization of hand keypoints using viser."""
    
    def __init__(self, stream, show_cameras: bool = True, port: int = 8080, 
                 predictions=None, model=None, tracker: Optional[HandTracker] = None,
                 generic_hand_model: Optional[HandModel] = None,
                 show_skeleton: bool = True, show_keypoints: bool = True,
                 show_keypoint_indices: bool = False):
        """
        Initialize 3D visualization.
        
        Args:
            stream: Stream with hand data (SyncedImagePoseStream or ImageSequencePoseStream)
            show_cameras: Whether to show camera frustums
            port: Port number for viser server
            predictions: Dict with 'tracked_hands' predictions (from eval_results)
            model: Pretrained model for tracking
            tracker: HandTracker instance
            generic_hand_model: The generic hand model for calibration
        """
        self.stream = stream
        self.show_cameras = show_cameras
        self.port = port
        self.predictions = predictions
        self.model = model
        self.tracker = tracker
        self.initial_show_skeleton = show_skeleton
        self.initial_show_keypoints = show_keypoints
        self.initial_show_keypoint_indices = show_keypoint_indices
        
        self.calibrated_hand_model: Optional[HandModel] = None
        
        # --- NEW: Perform calibration if tracker is provided ---
        if self.tracker and generic_hand_model:
            print("Calibrating tracker scale...")
            self._calibrate_tracker(generic_hand_model)
            print(f"✓ Calibration complete. Resetting tracker history.")
            self.tracker.reset_history()
        elif self.tracker:
            logger.warning("Tracker provided but generic_hand_model is missing. "
                           "Will use stream's ground-truth hand model, which may be inaccurate for predictions.")
            if hasattr(self.stream, '_hand_pose_labels') and self.stream._hand_pose_labels is not None:
                 self.calibrated_hand_model = self.stream._hand_pose_labels.hand_model
            else:
                 logger.error("Cannot get any hand model for tracker!")
        # --- END NEW ---

        # Initialize viser server
        self.server = viser.ViserServer(port=port)
        print(f"🌐 Viser server started at: http://localhost:{port}")
        
        # Current frame index
        self.frame_idx = 0
        self.playing = False
        self.playback_speed = 1.0
        
        # Cache frames for better performance
        self.frame_cache = {}
        self._cache_frames()
        
        # Setup UI controls
        self._setup_ui()
    
    def _calibrate_tracker(self, generic_hand_model: HandModel):
        """
        Runs the calibration step to find the mean hand scale.
        Adapted from _track_sequence_and_calibrate.
        """
        n_calibration_samples = 30 # Use 30 samples like in the original script
        predicted_scale_samples = []
        
        num_frames_to_iterate = min(n_calibration_samples, len(self.stream))
        if num_frames_to_iterate == 0:
            logger.error("Stream is empty, cannot calibrate.")
            return

        print(f"  Running calibration on {num_frames_to_iterate} frames...")
        
        try:
            for frame_idx in range(num_frames_to_iterate):
                # We can't use the iterator if we want to reset it later
                # So we manually get items.
                input_frame, gt_tracking = self.stream[frame_idx]
                
                gt_hand_model = self.stream._hand_pose_labels.hand_model
                crop_cameras = self.tracker.gen_crop_cameras(
                    [view.camera for view in input_frame.views],
                    self.stream._hand_pose_labels.camera_angles,
                    gt_hand_model,
                    gt_tracking,
                    min_num_crops=2,
                )
                res = self.tracker.track_frame_and_calibrate_scale(input_frame, crop_cameras)
                for hand_idx in res.hand_poses.keys():
                    predicted_scale_samples.append(res.predicted_scales[hand_idx])

            if not predicted_scale_samples:
                 logger.warning("No samples collected for scale calibration! Using generic model.")
                 self.calibrated_hand_model = generic_hand_model
                 return

            mean_scale = np.mean(predicted_scale_samples)
            logger.info(f"  Calibrated mean scale: {mean_scale} with {len(predicted_scale_samples)} samples")
            self.calibrated_hand_model = scaled_hand_model(
                generic_hand_model, mean_scale
            )
        except Exception as e:
            logger.error(f"Error during calibration: {e}. Falling back to generic model.")
            import traceback
            traceback.print_exc()
            self.calibrated_hand_model = generic_hand_model

    def _cache_frames(self):
        """Cache all frames for faster access."""
        print("Caching frames for faster visualization...")
        try:
            for idx, (input_frame, gt_tracking) in enumerate(self.stream):
                self.frame_cache[idx] = (input_frame, gt_tracking)
                if (idx + 1) % 50 == 0:
                    print(f"  Cached {idx + 1}/{len(self.stream)} frames...")
            print(f"✓ Cached {len(self.frame_cache)} frames")
        except Exception as e:
            logger.error(f"Error caching frames: {e}")
            import traceback
            traceback.print_exc()
        
    def _setup_ui(self):
        """Setup viser UI controls."""
        # Frame slider
        self.frame_slider = self.server.add_gui_slider(
            "Frame",
            min=0,
            max=len(self.stream) - 1,
            initial_value=0,
            step=1,
        )
        
        # Playback controls
        self.play_button = self.server.add_gui_button("Play/Pause")
        self.speed_slider = self.server.add_gui_slider(
            "Speed",
            min=0.1,
            max=5.0,
            initial_value=1.0,
            step=0.1,
        )
        
        # Display options
        self.show_gt_checkbox = self.server.add_gui_checkbox(
            "Show Ground Truth",
            initial_value=True,
        )
        self.show_pred_checkbox = self.server.add_gui_checkbox(
            "Show Predictions",
            # Enable predictions by default if tracker or predictions are loaded
            initial_value=(self.predictions is not None or self.tracker is not None),
        )
        self.show_cameras_checkbox = self.server.add_gui_checkbox(
            "Show Cameras",
            initial_value=self.show_cameras,
        )
        self.show_skeleton_checkbox = self.server.add_gui_checkbox(
            "Show Skeleton",
            initial_value=self.initial_show_skeleton,
        )
        self.show_keypoints_checkbox = self.server.add_gui_checkbox(
            "Show Keypoints",
            initial_value=self.initial_show_keypoints,
        )
        self.show_keypoint_indices_checkbox = self.server.add_gui_checkbox(
            "Show Keypoint Indices",
            initial_value=self.initial_show_keypoint_indices,
        )
        
        # Add separator
        self.server.add_gui_markdown("---\n**Appearance Settings**")
        
        # Point size
        self.point_size_slider = self.server.add_gui_slider(
            "Point Size",
            min=0.001,
            max=0.05,
            initial_value=0.01,
            step=0.001,
        )
        
        # Bone thickness
        self.bone_thickness_slider = self.server.add_gui_slider(
            "Bone Thickness",
            min=0.001,
            max=0.02,
            initial_value=0.005,
            step=0.001,
        )
        
        # Add legend
        self.server.add_gui_markdown(
            "---\n**Legend**\n"
            "- 🟢 Green = Left Hand\n"
            "- 🔵 Blue = Right Hand\n"
            "- Dimmer = Ground Truth\n"
            "- Brighter = Predictions"
        )
        
        # Setup callbacks
        @self.frame_slider.on_update
        def _on_frame_change(event: viser.GuiEvent):
            self.frame_idx = event.target.value
            self.update_visualization()
        
        @self.play_button.on_click
        def _on_play_pause(event: viser.GuiEvent):
            self.playing = not self.playing
        
        @self.speed_slider.on_update
        def _on_speed_change(event: viser.GuiEvent):
            self.playback_speed = event.target.value
        
        # Update callbacks for display options
        for checkbox in [self.show_gt_checkbox, self.show_pred_checkbox, 
                        self.show_cameras_checkbox, self.show_skeleton_checkbox,
                        self.show_keypoints_checkbox, self.show_keypoint_indices_checkbox]:
            checkbox.on_update(lambda _: self.update_visualization())
        
        for slider in [self.point_size_slider, self.bone_thickness_slider]:
            slider.on_update(lambda _: self.update_visualization())
    
    def draw_hand_skeleton(self, landmarks_3d: np.ndarray, hand_idx: int, 
                          prefix: str = "gt", color_modifier: tuple = (1.0, 1.0, 1.0)):
        """
        Draw hand skeleton in 3D.
        
        Args:
            landmarks_3d: (21, 3) array of 3D keypoints in world coordinates
            hand_idx: Hand index (0 or 1)
            prefix: Prefix for viser node names ("gt" or "pred")
            color_modifier: RGB color modifier
        """
        if len(landmarks_3d) != NUM_LANDMARKS_PER_HAND:
            print(f"  Warning: Expected {NUM_LANDMARKS_PER_HAND} landmarks, got {len(landmarks_3d)}")
            return
        
        base_color = HAND_COLORS[hand_idx]
        color = tuple(c * m for c, m in zip(base_color, color_modifier))
        
        # Determine scale based on coordinate magnitude
        coord_magnitude = np.abs(landmarks_3d).max()
        if coord_magnitude > 10:  # Likely in millimeters
            point_radius = 5.0  # 5mm spheres
            line_width = 2.0
        else:  # In meters
            point_radius = 0.01  # 1cm spheres
            line_width = 0.005
        
        # Use slider values
        point_radius = self.point_size_slider.value
        line_width = self.bone_thickness_slider.value

        # Draw keypoints
        if self.show_keypoints_checkbox.value:
            self.server.add_point_cloud(
                name=f"/{prefix}/hand_{hand_idx}/keypoints",
                points=landmarks_3d,
                colors=np.tile(color, (NUM_LANDMARKS_PER_HAND, 1)),
                point_size=point_radius,
            )
                
            # Add keypoint index labels if enabled
            if self.show_keypoint_indices_checkbox.value:
                for idx, point in enumerate(landmarks_3d):
                    label_offset = point_radius * 2.0
                    self.server.add_label(
                        name=f"/{prefix}/hand_{hand_idx}/label_{idx}",
                        text=str(idx),
                        position=tuple((point + np.array([label_offset, label_offset, label_offset])).tolist()),
                    )
        
        # Draw bones
        if self.show_skeleton_checkbox.value:
            bone_segments = []
            for (start_idx, end_idx) in HAND_CONNECTIONS:
                if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                    bone_segments.append(
                        [landmarks_3d[start_idx], landmarks_3d[end_idx]]
                    )
            
            if bone_segments:
                self.server.add_spline_catmull_rom(
                    name=f"/{prefix}/hand_{hand_idx}/skeleton",
                    positions=np.array(bone_segments),
                    color=color,
                    line_width=line_width,
                    segments_per_spline=2, # Straight lines
                )
    
    def draw_cameras(self, input_frame):
        """Draw camera frustums."""
        if not self.show_cameras_checkbox.value:
            return
        
        for cam_idx, view in enumerate(input_frame.views):
            camera = view.camera
            
            # Get camera pose (camera to world transform)
            if hasattr(camera, 'camera_to_world_xf'):
                c2w = camera.camera_to_world_xf
            else:
                # Create identity transform if not available
                c2w = np.eye(4)
            
            # Extract position and orientation
            position = c2w[:3, 3]
            rotation_matrix = c2w[:3, :3]
            
            # print(f"  Camera {cam_idx} position: {position}")
            
            # Determine scale based on position magnitude
            pos_magnitude = np.abs(position).max()
            if pos_magnitude > 10:  # Likely in millimeters
                axis_length = 50.0  # 50mm axes
            else:  # In meters
                axis_length = 0.1  # 10cm axes

            # Add camera frustum
            frustum_color = (1.0, 0.5, 0.0) if cam_idx == 0 else (1.0, 0.0, 0.5)
            
            self.server.add_frame(
                name=f"/cameras/camera_{cam_idx}",
                position=position,
                wxyz=viser.transforms.SO3.from_matrix(rotation_matrix).wxyz,
                axes_length=axis_length,
                axes_radius=axis_length * 0.05
            )
            
            # Add camera label
            self.server.add_label(
                name=f"/cameras/camera_{cam_idx}_label",
                text=f"Camera {cam_idx}",
                position=tuple(position),
            )
    
    def update_visualization(self):
        """Update visualization for current frame."""
        # Clear previous frame
        self.server.reset_scene()
        
        # Get frame data from cache
        if self.frame_idx not in self.frame_cache:
            logger.warning(f"Frame {self.frame_idx} not in cache")
            return
        
        input_frame, gt_tracking = self.frame_cache[self.frame_idx]
        
        # print(f"Frame {self.frame_idx}: {len(gt_tracking)} hands detected")

        # Draw ground truth
        if self.show_gt_checkbox.value and gt_tracking:
            # Get hand model from stream
            hand_model = None
            if hasattr(self.stream, '_hand_pose_labels') and self.stream._hand_pose_labels is not None:
                hand_model = self.stream._hand_pose_labels.hand_model
            
            if hand_model is not None:
                # print(f"  Drawing hands with hand_model")
                for hand_idx, gt_hand_pose in gt_tracking.items():
                    try:
                        landmarks_world = landmarks_from_hand_pose(
                            hand_model, gt_hand_pose, hand_idx
                        )
                        # print(f"  Hand {hand_idx}: {landmarks_world.shape} landmarks")
                        # print(f"    Range: [{landmarks_world.min():.3f}, {landmarks_world.max():.3f}]")
                        
                        self.draw_hand_skeleton(
                            landmarks_world, hand_idx, 
                            prefix="gt", 
                            color_modifier=(0.7, 0.7, 0.7)  # Dimmed
                        )
                    except Exception as e:
                        logger.error(f"Could not draw GT for hand {hand_idx}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"  Warning: hand_model is None, cannot draw keypoints")
        
        # Draw predictions
        if self.show_pred_checkbox.value:
            
            # --- Path 1: Load from pre-computed predictions file ---
            if self.predictions:
                # Note: The predictions file might be from your first script
                # Let's check for 'tracked_keypoints'
                if 'tracked_keypoints' in self.predictions:
                    valid_tracking = self.predictions['valid_tracking'][:, self.frame_idx]
                    for hand_idx, is_valid in enumerate(valid_tracking):
                        if is_valid:
                            pred_keypoints_3d = self.predictions['tracked_keypoints'][hand_idx, self.frame_idx]
                            self.draw_hand_skeleton(
                                pred_keypoints_3d, hand_idx,
                                prefix="pred",
                                color_modifier=(1.0, 1.0, 1.0) # Bright
                            )
                # Fallback for 'tracked_hands' format
                elif 'tracked_hands' in self.predictions:
                    if not hasattr(self.stream, '_hand_pose_labels'):
                         logger.warning("Need hand model from stream to visualize 'tracked_hands' predictions.")
                    else:
                        hand_model = self.stream._hand_pose_labels.hand_model
                        frame_predictions = self.predictions['tracked_hands'][self.frame_idx]
                        for hand_idx, pred_pose in frame_predictions.items():
                            pred_keypoints_3d = landmarks_from_hand_pose(
                                hand_model, pred_pose, hand_idx
                            )
                            self.draw_hand_skeleton(
                                pred_keypoints_3d, hand_idx,
                                prefix="pred",
                                color_modifier=(1.0, 1.0, 1.0) # Bright
                            )

            # --- Path 2: Run live tracking (CORRECTED) ---
            elif self.tracker is not None:
                if self.calibrated_hand_model is None:
                    logger.error(f"Frame {self.frame_idx}: Tracker is loaded but calibrated_hand_model is missing. Cannot predict.")
                else:
                    try:
                        # Get GT info needed for cropping
                        gt_hand_model = self.stream._hand_pose_labels.hand_model
                        camera_angles = self.stream._hand_pose_labels.camera_angles
                        
                        # 1. Generate crop cameras (like in working script)
                        crop_cameras = self.tracker.gen_crop_cameras(
                            [view.camera for view in input_frame.views],
                            camera_angles,
                            gt_hand_model,
                            gt_tracking,
                            min_num_crops=1,
                        )
                        
                        # 2. Run the correct tracking function (like in working script)
                        res = self.tracker.track_frame(
                            input_frame, 
                            self.calibrated_hand_model, 
                            crop_cameras
                        )

                        # 3. Convert poses to keypoints (like in working script)
                        for hand_idx, pred_pose in res.hand_poses.items():
                            pred_keypoints_3d = landmarks_from_hand_pose(
                                self.calibrated_hand_model, pred_pose, hand_idx
                            )
                            self.draw_hand_skeleton(
                                pred_keypoints_3d, hand_idx,
                                prefix="pred",
                                color_modifier=(1.0, 1.0, 1.0) # Bright
                            )
                    except Exception as e:
                        logger.error(f"Could not run tracker: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Draw cameras
        if self.show_cameras_checkbox.value:
            self.draw_cameras(input_frame)
        
        # Add coordinate frame at origin
        axes_scale = 0.1 # 10cm
        # Check coordinates to guess scale
        if self.show_gt_checkbox.value and gt_tracking:
             hand_model = self.stream._hand_pose_labels.hand_model
             landmarks_world = landmarks_from_hand_pose(
                    hand_model, next(iter(gt_tracking.values())), 0
             )
             if np.abs(landmarks_world).max() > 10:
                 axes_scale = 50.0 # 50mm

        self.server.add_frame(
            name="/world_origin",
            axes_length=axes_scale,
            axes_radius=axes_scale * 0.05,
        )
        
        # Add grid for reference
        grid_size = axes_scale * 10
        grid_spacing = axes_scale
        
        grid_points = []
        for i in np.arange(-grid_size, grid_size + 1e-5, grid_spacing):
            grid_points.append([[i, -grid_size, 0], [i, grid_size, 0]])
            grid_points.append([[-grid_size, i, 0], [grid_size, i, 0]])
        
        self.server.add_spline_catmull_rom(
            name="/grid",
            positions=np.array(grid_points),
            color=(0.3, 0.3, 0.3),
            line_width=axes_scale * 0.01,
            segments_per_spline=2, # Straight lines
        )
    
    def run(self):
        """Run interactive visualization loop."""
        print("\n" + "="*60)
        print("3D Hand Keypoint Visualization")
        print("="*60)
        print(f"Total frames: {len(self.stream)}")
        print("\nControls:")
        print("  - Use the web interface to navigate")
        print("  - Slider: Change frame")
        print("  - Play/Pause: Animate through frames")
        print("  - Checkboxes: Toggle visualization elements")
        print("  - Mouse: Rotate, pan, zoom the 3D view")
        print("\nPress Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Initial visualization
        self.update_visualization()
        
        try:
            # Main loop
            last_frame_time = time.time()
            while True:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
                # Handle playback
                if self.playing:
                    current_time = time.time()
                    dt = current_time - last_frame_time
                    
                    # Advance frame based on speed
                    if dt >= (1.0 / (30.0 * self.playback_speed)):
                        self.frame_idx = (self.frame_idx + 1) % len(self.stream)
                        self.frame_slider.value = self.frame_idx
                        self.update_visualization()
                        last_frame_time = current_time
        
        except KeyboardInterrupt:
            print("\n\nShutting down visualization...")
            # self.server.stop() # No longer needed, script just exits


def main():
    parser = argparse.ArgumentParser(
        description='3D visualization of hand keypoints using viser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Original UmeTrack MP4 format (shows ground truth)
    python visualize_3d_keypoints.py \\
        --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4
    
    # Run live tracking (requires model and generic hand model)
    python visualize_3d_keypoints.py \\
        --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \\
        --model pretrained_models/pretrained_weights.torch \\
        --generic-hand-model dataset/generic_hand_model.json
    
    # With pre-computed predictions from eval_results file
    python visualize_3d_keypoints.py \\
        --video UmeTrack_data/raw_data/real/hand_hand/training/user_00/recording_00.mp4 \\
        --predictions tmp/eval_results_unknown_skeleton/real/hand_hand/training/user_00/recording_00.npy
    
    # ZED image sequences with live tracking
    python visualize_3d_keypoints.py \\
        --left-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_left \\
        --right-dir ~/Documents/ZED/processed/HD2K_SN39914083_18-44-59_right \\
        --json ~/Documents/ZED/zed_stereo_intr.json \\
        --model pretrained_models/pretrained_weights.torch \\
        --generic-hand-model dataset/generic_hand_model.json
        """
    )
    
    # Input format - flexible options
    parser.add_argument('--video', 
                       help='Path to MP4 video (original UmeTrack format)')
    parser.add_argument('--image-sequence', nargs=3, metavar=('LEFT_DIR', 'RIGHT_DIR', 'JSON'),
                       help='Image sequence: left_dir right_dir json_path')
    
    # Individual arguments (alternative to --image-sequence)
    parser.add_argument('--left-dir',
                       help='Directory with left camera images (use with --right-dir and --json)')
    parser.add_argument('--right-dir',
                       help='Directory with right camera images (use with --left-dir and --json)')
    parser.add_argument('--json',
                       help='JSON file with camera intrinsics (use with --left-dir and --json)')
    
    # Common arguments
    parser.add_argument('--predictions', '-p',
                       help='Optional: .npy file with eval_results (e.g., tracked_keypoints)')
    parser.add_argument('--model', '-m',
                       help='Optional: Pretrained model path for live tracking')
    
    # --- NEW ARGUMENT ---
    parser.add_argument('--generic-hand-model', 
                       help='Path to generic_hand_model.json (REQUIRED for live tracking with --model)')
    
    # Display options
    parser.add_argument('--no-cameras', action='store_true',
                       help='Hide camera visualizations')
    parser.add_argument('--no-skeleton', action='store_true',
                       help='Hide skeleton connections (bones)')
    parser.add_argument('--no-keypoints', action='store_true',
                       help='Hide keypoint spheres')
    parser.add_argument('--show-indices', action='store_true',
                       help='Show keypoint indices (0-20) as labels')
    
    # Server options
    parser.add_argument('--port', type=int, default=8080,
                       help='Viser server port (default: 8080)')
    
    args = parser.parse_args()
    
    if not VISER_AVAILABLE:
        print("ERROR: viser not installed")
        print("Install with: pip install viser")
        return 1
    
    # Validate input arguments
    has_video = args.video is not None
    has_image_sequence = args.image_sequence is not None
    has_left_right_json = all([args.left_dir, args.right_dir, args.json])
    
    num_input_methods = sum([has_video, has_image_sequence, has_left_right_json])
    
    if num_input_methods == 0:
        print("Error: Must provide input data using one of:")
        print("  --video FILE")
        print("  --image-sequence LEFT_DIR RIGHT_DIR JSON")
        print("  --left-dir DIR --right-dir DIR --json FILE")
        parser.print_help()
        return 1
    
    if num_input_methods > 1:
        print("Error: Please use only one input method:")
        print(f"  --video: {has_video}")
        print(f"  --image-sequence: {has_image_sequence}")
        print(f"  --left-dir/--right-dir/--json: {has_left_right_json}")
        return 1
    
    # Validate tracking arguments
    if args.model and not args.generic_hand_model:
        parser.error("--generic-hand-model is required when using --model for live tracking.")
    
    if args.predictions and args.model:
        logger.warning("Both --predictions and --model provided. --predictions will be used, and live tracking will be disabled.")
        args.model = None # Disable live tracking

    # Determine input format and load stream
    stream = None
    
    # Check for video input
    if has_video:
        video_path = os.path.expanduser(args.video)
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return 1
        
        print("Loading MP4 video (UmeTrack format)...")
        print(f"  Video: {video_path}")
        
        stream = SyncedImagePoseStream(video_path)
        
        if len(stream) == 0:
            print("Error: No frames found in video")
            return 1
        
        print(f"  Total frames: {len(stream)}")
    
    # Check for image sequence input (compact format)
    elif has_image_sequence:
        if not IMAGE_SEQUENCE_AVAILABLE:
            print("ERROR: ImageSequencePoseStream not available")
            print("Make sure image_sequence_pose_stream.py is in the path")
            return 1
        
        left_dir, right_dir, json_path = args.image_sequence
        # (Expand user paths logic from original script) ...
        
        print("Loading ZED image sequence...")
        stream = ImageSequencePoseStream(left_dir, right_dir, json_path)
        # (Validation logic from original script) ...
    
    # Check for individual arguments format
    elif has_left_right_json:
        if not IMAGE_SEQUENCE_AVAILABLE:
            print("ERROR: ImageSequencePoseStream not available")
            print("Make sure image_sequence_pose_stream.py is in the path")
            return 1
        
        left_dir = os.path.expanduser(args.left_dir)
        right_dir = os.path.expanduser(args.right_dir)
        json_path = os.path.expanduser(args.json)
        
        # (Validation logic from original script) ...
        if not os.path.exists(left_dir): print(f"Error: Left directory not found: {left_dir}"); return 1
        if not os.path.exists(right_dir): print(f"Error: Right directory not found: {right_dir}"); return 1
        if not os.path.exists(json_path): print(f"Error: JSON file not found: {json_path}"); return 1

        print("Loading ZED image sequence...")
        stream = ImageSequencePoseStream(left_dir, right_dir, json_path)
        
        if len(stream) == 0:
            print("Error: No frames found in stream")
            return 1
        
        print(f"  Total frames: {len(stream)}")
    
    # Load predictions if provided
    predictions = None
    if args.predictions:
        pred_path = os.path.expanduser(args.predictions)
        if os.path.exists(pred_path):
            try:
                predictions = np.load(pred_path, allow_pickle=True).item()
                print(f"✓ Loaded predictions from: {pred_path}")
                if 'tracked_keypoints' in predictions:
                    print(f"  Found 'tracked_keypoints' for {predictions['tracked_keypoints'].shape[1]} frames")
                elif 'tracked_hands' in predictions:
                    print(f"  Found 'tracked_hands' for {len(predictions['tracked_hands'])} frames")
            except Exception as e:
                logger.error(f"Failed to load predictions: {e}")
                predictions = None
        else:
            print(f"Warning: Predictions file not found: {pred_path}")
    
    # Load model and tracker if requested
    model = None
    tracker = None
    generic_hand_model = None
    
    if args.model:
        if not TRACKER_AVAILABLE:
            print("Warning: Tracker modules not available, cannot perform live tracking.")
        else:
            model_path = os.path.expanduser(args.model)
            generic_hand_model_path = os.path.expanduser(args.generic_hand_model)
            
            if os.path.exists(model_path) and os.path.exists(generic_hand_model_path):
                try:
                    print(f"Loading model from: {model_path}")
                    model = load_pretrained_model(model_path)
                    model.eval()
                    opts = HandTrackerOpts()
                    tracker = HandTracker(model, opts)
                    print("✓ Tracker initialized")
                    
                    print(f"Loading generic hand model from: {generic_hand_model_path}")
                    generic_hand_model = load_hand_model_from_dict(_load_json(generic_hand_model_path))
                    print("✓ Generic hand model loaded")
                    
                except Exception as e:
                    logger.error(f"Failed to load model or hand model: {e}")
                    model = None
                    tracker = None
                    generic_hand_model = None
            else:
                if not os.path.exists(model_path):
                    print(f"Warning: Model file not found: {model_path}")
                if not os.path.exists(generic_hand_model_path):
                    print(f"Warning: Generic hand model file not found: {generic_hand_model_path}")

    
    # Create and run visualization
    viz = HandVisualization3D(
        stream, 
        show_cameras=not args.no_cameras,
        port=args.port,
        predictions=predictions, 
        model=model, 
        tracker=tracker,
        generic_hand_model=generic_hand_model,
        show_skeleton=not args.no_skeleton,
        show_keypoints=not args.no_keypoints,
        show_keypoint_indices=args.show_indices,
    )
    viz.run()
    
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())