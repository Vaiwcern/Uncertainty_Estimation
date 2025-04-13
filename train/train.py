import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from datetime import datetime
import yaml

from custom_dataset.DatasetController import DatasetController
from training_function import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')

    parser.add_argument('--model', type=str, required=True,
        help="Model type. Options: 'iterative' or 'vanila'.")

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of the dataset to be used. Options: 'RT' or 'Mass'.")

    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the dataset directory.")

    parser.add_argument('--dropout_rate', type=float, required=False, default=0.1,
        help="Dropout rate to prevent overfitting. Default: 0.1.")

    parser.add_argument('--use_batchnorm', action='store_true', required=False, default=False,
        help="Whether to use Batch Normalization. Default: False")

    parser.add_argument('--image_channel', type=int, required=False, default=3,
        help="Number of channels in original samples. E.g., 3 for RGB, 1 for grarscale. Default: 3")

    parser.add_argument('--add_channel', action='store_true', required=False, default=False,
        help="Whether to add an extra channel during preprocessing. Default: False")

    parser.add_argument('--batch_size', type=int, required=True,
        help="Training batch size. Common values: 8, 16, 32, etc.")

    parser.add_argument('--learning_rate', type=float, required=False, default=0.001,
        help="Learning rate for the optimizer. Default: 0.001.")

    parser.add_argument('--num_epoch', type=int, required=True,
        help="Total number of training epochs.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Directory to save model checkpoints.")

    parser.add_argument('--save_per_epoch', type=int, required=False, default=5,
        help="Save model weights every N epochs. Default: 5.")

    parser.add_argument('--loss_function', type=str, required=True, default='focal',
        help="Loss function to use during training. Options: 'focal', or 'iou'. "
         "Default: 'focal'.")

    parser.add_argument('--buffer_size', type=int, required=False, default=None,
        help="Buffer size for shuffling dataset. "
         "If not specified, it will default to the total number of training images (perfect shuffle). "
         "For large datasets, you can use a smaller value to reduce memory usage.")
        
    parser.add_argument('--gpus', type=str, required=True,
        help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    # === Set devices ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    # === Log setting ===
    os.makedirs(args.save_path, exist_ok=True)
    setting_log_path = os.path.join(args.save_path, "setting.yaml")

    with open(setting_log_path, "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)


    # === Log training process ===
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    train_log_dir = os.path.join(args.save_path, "train_logs")
    os.makedirs(train_log_dir, exist_ok=True)

    log_file_path = os.path.join(train_log_dir, f"training__{timestamp}.log")

    log_file = open(log_file_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file


    # === Load Dataset ===
    if args.dataset == "RT":
        data_wrapper = DatasetController.get_roadtracer_train_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=args.add_channel,
            buffer_size = args.buffer_size
        )
    elif args.dataset == "Mass":
        data_wrapper = DatasetController.get_massachusetts_train_wrapper(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            add_channel=args.add_channel,
            buffer_size = args.buffer_size
        )
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    # === Train === 
    in_channels = args.image_channel + (1 if args.add_channel else 0)
    train(
        model=args.model,
        train_dataset_wrapper=data_wrapper,
        input_channels=in_channels,
        num_epoch=args.num_epoch,
        save_path=args.save_path,
        loss_function = args.loss_function,
        use_batchnorm=args.use_batchnorm,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        save_per_epoch=args.save_per_epoch,
    )

