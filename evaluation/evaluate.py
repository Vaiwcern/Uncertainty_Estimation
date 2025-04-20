import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import argparse
import yaml

from custom_dataset.DatasetController import DatasetController
from evaluation_function import segmentation_evaluation, uncertainty_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Unet model.')

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of the dataset. Options: 'RT' or 'Mass'.")

    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the dataset directory.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Directory to save evaluation results.")

    parser.add_argument('--prediction_path', type=str, required=True,
        help="Directory that saved predictions.")

    parser.add_argument('--iterative', type=int, required=True,
        help="Number of iterative refinement passes.")

    parser.add_argument('--samples', type=int, required=True,
        help="Number of stochastic samples (e.g., for MC dropout).")

    parser.add_argument('--eval_type', type=str, required=True,
        help="Evaluation type. Options: 'segmentation', 'uncertainty', 'out-of-distribution'")

    parser.add_argument('--ood_dataset', type=str, required=False,
        help="Name of OOD dataset if eval_type is 'out-of-distribution'.")

    parser.add_argument('--ood_dataset_path', type=str, required=False,
        help="Path to OOD dataset if eval_type is 'out-of-distribution'.")

    parser.add_argument('--relaxed_ccq', action='store_true',
        help="Use relaxed CCQ metric (with slack).")

    parser.add_argument('--n_rows', type=int, required=False,
        help="Cut images into n_rows x n_cols for uncertainty estimation.")

    parser.add_argument('--n_cols', type=int, required=False,
        help="Cut images into n_rows x n_cols for uncertainty estimation.")

    parser.add_argument('--gpus', type=str, required=True,
            help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")

    parser.add_argument('--num_workers', type=int, required=False,
            help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.eval_type == "out-of-distribution":
        if not args.ood_dataset or not args.ood_dataset_path:
            raise ValueError("Arguments '--ood_dataset' and '--ood_dataset_path' are required for OOD evaluation.")
        
    if args.eval_type == "uncertainty":
        if not args.n_rows or not args.n_cols:
            raise ValueError("Arguments '--n_rows' and '--n_cols' are required for uncertainty estimation.")
        

    # === Set devices ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    # === Log setting ===
    os.makedirs(args.save_path, exist_ok=True)
    setting_log_path = os.path.join(args.save_path, f"setting_{args.eval_type}.yaml")

    with open(setting_log_path, "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # === Load Dataset ===
    if args.dataset == "RT":
        data_wrapper = DatasetController.get_roadtracer_test_wrapper(
            dataset_path=args.dataset_path,
            batch_size=1,
            add_channel=False,
        )
    elif args.dataset == "Mass":
        data_wrapper = DatasetController.get_massachusetts_test_wrapper(
            dataset_path=args.dataset_path,
            batch_size=1,
            add_channel=False,
        )
    elif args.dataset == "Drive":
        data_wrapper = DatasetController.get_drive_test_wrapper(
            dataset_path=args.dataset_path,
            batch_size=1,
            add_channel=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # === Segmentation Evaluation ===
    if args.eval_type == 'segmentation':
        segmentation_evaluation(
            data_wrapper=data_wrapper,
            iterative=args.iterative,
            samples=args.samples,
            pred_dir=args.prediction_path,
            relax=args.relaxed_ccq,
            save_path=args.save_path,
            num_workers = args.num_workers
        )

    elif args.eval_type == 'uncertainty':
        uncertainty_evaluation(
                    data_wrapper=data_wrapper,
                    iterative=args.iterative,
                    samples=args.samples,
                    pred_dir=args.prediction_path,
                    save_path=args.save_path,
                    num_workers = args.num_workers,
                    n_rows = args.n_rows,
                    n_cols= args.n_cols, 
                )
    elif args.eval_type == 'out-of-distribution':
        print("OOD evaluation is not yet implemented.")

    else:
        raise ValueError(f"Unsupported evaluation type: {args.eval_type}")
