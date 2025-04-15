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
    
    parser.add_argument('--model_path', type=str, required=True,
        help="Path to the directory where checkpoint saved.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Path to the directory where predictions would be saved.")
    
    parser.add_argument('--epoch', type=int, required=True,
        help="The epoch of the checkpoint want to load.")

    parser.add_argument('--training_mode', action='store_true',
        help="Enable training mode during prediction (dropout, BN). Default: False")
    
    parser.add_argument('--batch_size', type=int, required=True,
        help="Batch size for prediction (per step).")

    parser.add_argument('--iterative', type=int, required=True,
        help="Number of iterative passes for each sample.")

    parser.add_argument('--samples', type=int, required=True,
        help="Number of stochastic samples per input (e.g., for MC dropout).")
    
    parser.add_argument('--gpus', type=str, required=True,
            help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    # === Set devices ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    # === Log predicting process ===
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    train_log_dir = os.path.join(args.save_path, "predicts_logs")
    os.makedirs(train_log_dir, exist_ok=True)

    log_file_path = os.path.join(train_log_dir, f"predict__{timestamp}.log")

    log_file = open(log_file_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file


    # === Load Dataset ===
    if args.dataset == "RT":
        data_wrapper = DatasetController.get_roadtracer_test_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=False,
        )
    elif args.dataset == "Mass":
        data_wrapper = DatasetController.get_massachusetts_test_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=False,
        )
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    # ====== Predict ======
    predict_and_save_results(
        model_path=args.model_path,
        epoch=args.epoch,
        data_wrapper=data_wrapper,
        save_path=args.save_path,
        training=args.training_mode,
        iterative=args.iterative,
        samples=args.samples
    )
