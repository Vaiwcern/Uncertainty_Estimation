import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
import argparse
from datetime import datetime

from predict_function import predict_and_save_results
from custom_dataset.DatasetController import DatasetController

def parse_args(): 
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of the dataset to be used. Options: 'RT' or 'Mass'.")

    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the dataset directory.")
    
    parser.add_argument('--save_path', type=str, required=True,
        help="Path to the directory where results would be saved.")
    
    parser.add_argument('--iterative', type=int, required=True,
        help="Number of iterative passes for each sample.")

    parser.add_argument('--samples', type=int, required=True,
        help="Number of stochastic samples per input (e.g., for MC dropout).")
    
    parser.add_argument('--eval_type', type=str, required=True,
        help="Option: 'segmentation', 'uncertainty' or 'out-of-distribution'")
    
    parser.add_argument('--ood_dataset', type=str, required=False,
        help="Option: 'segmentation', 'uncertainty' or 'ood'")
    
    parser.add_argument('--ood_dataset_path', type=str, required=False,
        help="Option: 'segmentation', 'uncertainty' or 'ood'")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.eval_type == "out-of-distribution":
        if not args.ood_dataset or not args.ood_dataset_path:
            raise ValueError("Arguments '--ood_dataset' and '--ood_dataset_path' are required when eval_type is 'out-of-distribution'.")
