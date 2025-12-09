# Copyright (c) Chuer Pan

import logging
import os
import sys
import fnmatch
import numpy as np
import lib.data_utils.fs as fs
from functools import partial
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.common.hand import HandModel, scaled_hand_model
from multiprocessing import Pool
from typing import Optional, Tuple, overload
import argparse
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame
from lib.tracker.video_pose_data import SyncedImagePoseStream, ImageSequencePoseStream, _load_json, load_hand_model_from_dict
from typing import Union

MM_TO_M = 0.001
M_TO_MM = 1000.0

try:
    from irt.utils.color_print import info_print
except ModuleNotFoundError:  # pragma: no cover - fallback for local runs without package install
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[4]
    _src_dir = _repo_root / "src"
    if _src_dir.exists():
        sys.path.insert(0, str(_src_dir))
    try:
        from irt.utils.color_print import info_print
    except ModuleNotFoundError:
        from irt.utils.color_print import info_print
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def _find_input_output_files(input_file: str, output_dir: str):
    input_full_path = [fs.join(input_file)]
    output_file_name = '_'.join(input_file.split("/")[-6:-1]+input_file.split("/")[-1].split(".")[0:1]) # 'raw_data_real_separate_hand_testing_user_19_recording_01'
    output_full_path = [fs.join(output_dir, output_file_name + ".npy")]
    return input_full_path, output_full_path

def _track_sequence_and_calibrate(
    image_pose_stream: Union[SyncedImagePoseStream, ImageSequencePoseStream],
    tracker: HandTracker,
    generic_hand_model: HandModel,
    n_calibration_samples: int,
):
    predicted_scale_samples = []
    
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        # Use BOTH cameras during calibration (required for scale estimation)
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            generic_hand_model,
            gt_tracking,
            min_num_crops=1,
        )
 
        res = tracker.track_frame_and_calibrate_scale(input_frame, crop_cameras)
        for hand_idx in res.hand_poses.keys():
            predicted_scale_samples.append(res.predicted_scales[hand_idx])
        if n_calibration_samples != 0 and len(predicted_scale_samples) >= n_calibration_samples:
            predicted_scale_samples = predicted_scale_samples[:n_calibration_samples]
            break

    assert len(predicted_scale_samples) > 0, "No samples collected for scale calibration!"
    mean_scale = np.mean(predicted_scale_samples)
    logger.info(f"Calibrated mean scale: {mean_scale} with {len(predicted_scale_samples)} samples")
    calibrated_hand_model = scaled_hand_model(
        generic_hand_model, mean_scale
    )
    return calibrated_hand_model


def _track_sequence(
    input_output: Tuple[str, str],
    model_path: str,
    generic_hand_model: HandModel,
    n_calibration_samples: int,
    override: bool = False,
    json_path: str = None,
) -> Optional[None]:
    try:
        data_path, output_path = input_output
        if not override and fs.exists(output_path):
            logger.info(f"Skipping '{data_path}' since output path '{output_path}' already exists")
            return None

        logger.info(f"Processing {data_path}...")
        model = load_pretrained_model(model_path)
        model.eval()
        
        if data_path.endswith(".mp4"):
            info_print(f"Loading image pose stream from {data_path}")
            image_pose_stream = SyncedImagePoseStream(data_path)
        elif os.path.isdir(data_path):
            # For directory input, try to detect ZED stereo format
            # Look for _left/_right pattern or check parent directory
            left_dir = None
            right_dir = None
            # Check if directory ends with _left
            if data_path.endswith("_left"):
                left_dir = data_path
                base_dir = data_path[:-5]  # Remove "_left"
                right_dir = base_dir + "_right"
                image_pose_stream = ImageSequencePoseStream(left_dir, right_dir, json_path = json_path, image_format = "png")
           
            # Find JSON file - check same directory, parent, or with base name
            base_path = left_dir.replace("_left", "")
            possible_json_paths = [
                os.path.join(os.path.dirname(left_dir), os.path.basename(base_path) + ".json"),
                os.path.join(left_dir, "..", os.path.basename(base_path) + ".json"),
                os.path.join(left_dir, "camera_params.json"),
                os.path.join(os.path.dirname(left_dir), "camera_params.json"),
            ]
            
            for json_candidate in possible_json_paths:
                json_candidate = os.path.abspath(json_candidate)  # Normalize path
                if os.path.exists(json_candidate):
                    json_path = json_candidate
                    break
        else:
            raise ValueError(f"Invalid input file: {data_path}. Must be .mp4 file or directory")
  
        # Skip processing if the video stream is invalid
        if len(image_pose_stream) == 0:
            logger.info(f"Skipping {data_path} due to invalid video file")
            return None 
        tracker = HandTracker(model, HandTrackerOpts())
        calibrated_hand_model = _track_sequence_and_calibrate(
            image_pose_stream, tracker, generic_hand_model, n_calibration_samples
        )

        # Reset the history and retrack using the calibrated skeleton.
        tracker.reset_history()
        tracked_keypoints = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
        tracked_keypoints_eye = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
        tracked_keypoints_2d = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 2])
        valid_tracking = np.zeros([NUM_HANDS, len(image_pose_stream)], dtype=bool)
        
        for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
            gt_hand_model = image_pose_stream._hand_pose_labels.hand_model

            crop_cameras = tracker.gen_crop_cameras(
                [view.camera for view in input_frame.views],
                image_pose_stream._hand_pose_labels.camera_angles,
                gt_hand_model,
                gt_tracking,
                min_num_crops=1,
            )
            res = tracker.track_frame(input_frame, calibrated_hand_model, crop_cameras)

            if not input_frame.views:
                logger.warning(f"No views in frame {frame_idx}, skipping...")
                continue
                
            # T_world_cam is the camera's pose in the world
            T_world_cam = input_frame.views[0].camera.camera_to_world_xf
            
            # T_cam_world is the transform from world points to camera points
            T_cam_world = np.linalg.inv(T_world_cam)
            
            # Define 180-degree rotation matrix around X-axis
            # This transforms from (+Y Up, +Z Back) to (+Y Down, +Z Fwd)
            R_Y_flip = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],  # Flips Y
                [0.0,  0.0,  1.0]   # Z remains forward
            ])
            # --- [END FIX] ---

            # Get camera for world_to_eye conversion and 2D projection
            camera = input_frame.views[1].camera # use the second left camera as that one usually have hands
            
            for hand_idx in res.hand_poses.keys():
                
                # 1. Get keypoints in UmeTrack's World Frame
                keypoints_world = landmarks_from_hand_pose(
                    calibrated_hand_model, res.hand_poses[hand_idx], hand_idx
                )

                # 2. Convert to eye coordinates using camera.world_to_eye() (like in visualize_keypoints.py)
                keypoints_eye = camera.world_to_eye(keypoints_world)
                
                # 3. Project to 2D image plane using camera.eye_to_window()
                keypoints_2d = camera.eye_to_window(keypoints_eye)

                
                # # 4. Convert to homogeneous coordinates for ZED frame transformation
                # num_keypoints = keypoints_world.shape[0]
                # ones = np.ones((num_keypoints, 1))
                # keypoints_world_homog = np.hstack((keypoints_world, ones))
                
                # # 5. Apply T_cam_world to move from World to Camera frame
                # keypoints_cam_homog = keypoints_world_homog @ T_cam_world.T
                
                # # 6. Convert back to (N, 3)
                # keypoints_cam = keypoints_cam_homog[:, :3]
                
                # # 7. Apply rotation to switch coordinate conventions
                # keypoints_zed_image_frame = keypoints_cam @ R_Y_flip
                
                # 8. Save all the keypoints
                tracked_keypoints[hand_idx, frame_idx] = keypoints_world 
                tracked_keypoints_eye[hand_idx, frame_idx] = keypoints_eye 
                tracked_keypoints_2d[hand_idx, frame_idx] = keypoints_2d
                
                valid_tracking[hand_idx, frame_idx] = True
 
        if not fs.exists(fs.dirname(output_path)):
            os.makedirs(fs.dirname(output_path))
        
        # Save in numpy format (.npy) which is compatible with visualize_3d_keypoints.py
        # np.save can handle dictionaries when allow_pickle=True
        results_dict = {
            "tracked_keypoints": tracked_keypoints,
            "valid_tracking": valid_tracking,
        }
        # Use numpy save format for compatibility with visualization script
        np.save(output_path, results_dict, allow_pickle=True)
        logger.info(f"Results saved at {output_path}")
        
        # Save 3D keypoints in eye coordinates as separate npy file
        output_path_eye = output_path.replace(".npy", "_eye_3d.npy")
        eye_results_dict = {
            "tracked_keypoints": tracked_keypoints_eye,
            "valid_tracking": valid_tracking,
        }
        np.save(output_path_eye, eye_results_dict, allow_pickle=True)
        logger.info(f"3D eye coordinates saved at {output_path_eye}")
        
        # Save 2D projected keypoints as separate npy file
        output_path_2d = output_path.replace(".npy", "_2d.npy")
        keypoints_2d_dict = {
            "tracked_keypoints": tracked_keypoints_2d,
            "valid_tracking": valid_tracking,
        }
        np.save(output_path_2d, keypoints_2d_dict, allow_pickle=True)
        logger.info(f"2D projected keypoints saved at {output_path_2d}")
        
        return None
    except Exception as e:
        logger.error(f"Error processing {input_output[0] if input_output else 'unknown'}: {e}", exc_info=True)
        raise

def main():
    root = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Run UmeTrack inference on ZED stereo data')
    parser.add_argument('--n-calibration-samples', type=int, default=30,
                        help='Number of calibration samples to use')
    parser.add_argument('--pool-size', type=int, default=1,
                        help='Number of parallel processes')
    
    default_model = os.path.join(root, "pretrained_models", "pretrained_weights.torch")
    parser.add_argument('--model-path', type=str, default=default_model,
                        help='Path to the pretrained model')
    default_hand_model = os.path.join(root, "UmeTrack_data", "raw_data", "real", "hand_hand", "training", "user_00", "recording_00.json")
    parser.add_argument('--generic-hand-model', type=str, default=default_hand_model,
                        help='Path to the generic hand model')
 
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input file')
    default_output = os.path.join(root, "zed_pred")
    parser.add_argument('--output-dir', type=str, default=default_output,
                        help='Path to the output directory')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Path to the JSON file with camera parameters, required for ZED stereo data, not required for .mp4 file from UmeTrack data')
    parser.add_argument('--visualize', action='store_true',
                        help='Automatically launch visualization after inference completes')
    parser.add_argument('--viz-port', type=int, default=8080,
                        help='Port for visualization server (default: 8080)')
    parser.add_argument('--override', action='store_true',
                        help='Override existing output files')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        return 1
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return 1
    if not os.path.exists(args.generic_hand_model):
        print(f"Error: Generic hand model not found: {args.generic_hand_model}")
        return 1

    generic_hand_model = load_hand_model_from_dict(_load_json(args.generic_hand_model)["hand_model"])
    input_paths, output_paths = _find_input_output_files(args.input_file, args.output_dir)
 

    track_fn = partial(
        _track_sequence,
        model_path=args.model_path,
        generic_hand_model=generic_hand_model,
        n_calibration_samples=args.n_calibration_samples,
        json_path=args.json_path,
        override=args.override
    )

    print(f"Starting tracking on {len(input_paths)} files using {args.pool_size} workers...")
    if args.pool_size == 1:
        for input_output in zip(input_paths, output_paths):
            track_fn(input_output)
    else:
        with Pool(args.pool_size) as p:
            p.map(track_fn, zip(input_paths, output_paths))
    print("Tracking complete!")
    
    # Launch visualization if requested
    if args.visualize and len(output_paths) > 0:
        import subprocess
        predictions_path = output_paths[0]  # Use first output file
        
        print(f"\n{'='*60}")
        print(f"Launching visualization with predictions from: {predictions_path}")
        print(f"{'='*60}\n")
        
        # Determine input type for visualization
        input_file = args.input_file
        viz_args = []
        
        if input_file.endswith(".mp4"):
            viz_args = ["--video", input_file]
        elif os.path.isdir(input_file) and input_file.endswith("_left"):
            # ZED stereo format
            base_dir = input_file[:-5]
            right_dir = base_dir + "_right"
            
            # Find JSON file
            json_file = args.json_path
            if not json_file:
                # Try to find it automatically
                base_path = input_file.replace("_left", "")
                possible_json_paths = [
                    os.path.join(os.path.dirname(input_file), os.path.basename(base_path) + ".json"),
                    os.path.join(input_file, "..", os.path.basename(base_path) + ".json"),
                    os.path.join(input_file, "camera_params.json"),
                    os.path.join(os.path.dirname(input_file), "camera_params.json"),
                ]
                for json_candidate in possible_json_paths:
                    json_candidate = os.path.abspath(json_candidate)
                    if os.path.exists(json_candidate):
                        json_file = json_candidate
                        break
            
            if json_file and os.path.exists(right_dir):
                viz_args = ["--left-dir", input_file, "--right-dir", right_dir, "--json", json_file]
            else:
                logger.warning(f"Could not find right directory or JSON file for visualization. "
                             f"Right dir: {right_dir}, JSON: {json_file}")
                logger.warning("Skipping visualization.")
                return 0
        
        if viz_args:
            # Build visualization command
            viz_script = os.path.join(root, "visualize_3d_keypoints.py")
            cmd = [
                sys.executable, viz_script,
                *viz_args,
                "--predictions", predictions_path,
                "--port", str(args.viz_port),
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Visualization failed: {e}")
                return 1
            except KeyboardInterrupt:
                logger.info("Visualization interrupted by user")
                return 0
        else:
            logger.warning("Could not determine input format for visualization. Skipping.")


if __name__ == '__main__':
    # This ensures the main() function is called when the script is run
    main()