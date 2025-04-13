import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from custom_dataset import DatasetController
from training_function import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')

    parser.add_argument('--model', type=str, required=True,
        help="Model type: 'iterative' or 'vanila'.")

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of the dataset to be used. Example: 'RT', 'Mass'.")

    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the dataset directory.")

    parser.add_argument('--dropout_rate', type=float, required=True,
        help="Dropout rate to prevent overfitting. Example: 0.5.")

    parser.add_argument('--use_batchnorm', type=bool, required=True,
        help="Whether to use Batch Normalization. (True/False)")

    parser.add_argument('--image_channel', type=int, required=True,
        help="Number of channels in original samples. E.g., 3 for RGB.")

    parser.add_argument('--add_channel', type=bool, required=True,
        help="Whether to add an extra channel during preprocessing. (True/False)")

    parser.add_argument('--batch_size', type=int, required=True,
        help="Training batch size. Common values: 8, 16, 32, etc.")

    parser.add_argument('--learning_rate', type=float, required=True,
        help="Learning rate for the optimizer. Example: 0.001.")

    parser.add_argument('--num_epoch', type=int, required=True,
        help="Total number of training epochs.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Directory to save model checkpoints.")

    parser.add_argument('--save_per_epoch', type=int, required=True,
        help="Save model weights every N epochs. Example: 5.")

    parser.add_argument('--loss_function', type=int, required=True,
        help="focal")
        
    parser.add_argument('--gpus', type=str, required=True,
        help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # === Choose gpus ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # === Load Dataset ===
    if args.dataset == "RT":
        data_wrapper = DatasetController.get_roadtracer_train_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=args.add_channel 
        )
    elif args.dataset == "Mass":
        data_wrapper = DatasetController.get_massachusetts_train_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=args.add_channel 
        )
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # === Train === 
    train(
        model=args.model,
        train_dataset_wrapper=data_wrapper,
        use_batchnorm=args.use_batchnorm,
        dropout_rate=args.dropout_rate,
        input_channels=args.input_channel,
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        save_path=args.save_path,
        loss_function = args.loss_fucntion,
        save_per_epoch=args.save_per_epoch
    )

