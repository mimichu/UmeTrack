# Copyright (c) Chuer Pan

import logging
import os
import fnmatch
import pickle
import numpy as np
import lib.data_utils.fs as fs
from functools import partial
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.common.hand import HandModel, scaled_hand_model
from multiprocessing import Pool
from typing import Optional, Tuple
import argparse
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame
from lib.tracker.video_pose_data import SyncedImagePoseStream, _load_json, load_hand_model_from_dict

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def _find_input_output_files(input_file: str, output_dir: str):
    input_full_path = [fs.join(input_file)]
    output_full_path = [fs.join(output_dir, input_file.split("/")[-1].split(".")[0] + ".npy")]
    return input_full_path, output_full_path

def _track_sequence_and_calibrate(
    image_pose_stream: SyncedImagePoseStream,
    tracker: HandTracker,
    generic_hand_model: HandModel,
    n_calibration_samples: int,
):
    predicted_scale_samples = []
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        gt_hand_model = image_pose_stream._hand_pose_labels.hand_model
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            gt_hand_model,
            gt_tracking,
            min_num_crops=2,
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
) -> Optional[None]:
    try:
        data_path, output_path = input_output
        if not override and fs.exists(output_path):
            logger.info(f"Skipping '{data_path}' since output path '{output_path}' already exists")
            return None

        logger.info(f"Processing {data_path}...")
        model = load_pretrained_model(model_path)
        model.eval()

        image_pose_stream = SyncedImagePoseStream(data_path)
  
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

            for hand_idx in res.hand_poses.keys():
                tracked_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                    calibrated_hand_model, res.hand_poses[hand_idx], hand_idx
                )
                valid_tracking[hand_idx, frame_idx] = True

        if not fs.exists(fs.dirname(output_path)):
            os.makedirs(fs.dirname(output_path))
        with fs.open(output_path, "wb") as fp:
            pickle.dump(
                {
                    "tracked_keypoints": tracked_keypoints,
                    "valid_tracking": valid_tracking,
                },
                fp,
            )
        logger.info(f"Results saved at {output_path}")
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
        n_calibration_samples=args.n_calibration_samples
    )

    print(f"Starting tracking on {len(input_paths)} files using {args.pool_size} workers...")
    if args.pool_size == 1:
        for input_output in zip(input_paths, output_paths):
            track_fn(input_output)
    else:
        with Pool(args.pool_size) as p:
            p.map(track_fn, zip(input_paths, output_paths))
    print("Tracking complete!")


if __name__ == '__main__':
    # This ensures the main() function is called when the script is run
    main()