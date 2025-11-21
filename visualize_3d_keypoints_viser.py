import os
import numpy 
import argparse
import logging
import sys
from InteractionRetarget.src.vis.HandVis3D import HandVisualization3D

def main():
    parser = argparse.ArgumentParser(
        description='3D visualization of hand keypoints using viser',
        formatter_class=argparse)

    # Individual arguments (alternative to vi--image-sequence)
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
    default_hand_model = os.path.join(os.path.dirname(__file__), "dataset", "generic_hand_model.json")
    parser.add_argument('--generic-hand-model', type=str, default=default_hand_model,
                        help='Path to the generic hand model')
 
    # Server options
    parser.add_argument('--port', type=int, default=8080,
                       help='Viser server port (default: 8080)')
    
    args = parser.parse_args()
    
    
    viz = HandVisualization3D()
    viz.visualize_hand()
 
    
    return 0
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())