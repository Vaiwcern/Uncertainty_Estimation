import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import imageio.v3 as imageio

from evaluation_utils import load_model_from_folder, load_data_wrapper_from_folder
from evaluation_function import predict, build_distributed_predict_step

# ====== PREDICTION STEP ======
@tf.function
def distributed_predict_step(batch):
    def step_fn(images, masks, filenames):
        images = tf.cast(images, tf.float16)  # Nếu bạn dùng mixed precision
        preds = model(images, training=False)
        return preds, filenames
    images, masks, filenames = batch
    return strategy.run(step_fn, args=(images, masks, filenames))

def parse_args(): 
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')

    parser.add_argument('--model', type=str, required=True,
        help="Model type. Options: 'iterative' or 'vanila'.")

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of the dataset to be used. Options: 'RT' or 'Mass'.")

    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the dataset directory.")
    
    parser.add_argument('--model_path`', type=str, required=True,
        help="Path to the directory where checkpoint saved.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Path to the directory where checkpoint saved.")
    
    parser.add_argument('--epoch', type=int, required=True,
        help="The epoch of the checkpoint want to load.")

    parser.add_argument('--batch_size', type=int, required=True,
        help="")

    parser.add_argument('--gpus', type=str, required=True,
        help="Comma-separated list of GPU device IDs to use. Example: '0,1'.")

if __name__ == "__main__":
    args = parse_args()

    # === Set devices ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    # === Load model ===
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model, config = load_model_from_folder(args.model_path, epoch=args.epoch)

    # === hehe === 
    distributed_predict_step = build_distributed_predict_step(
        model=model,
        strategy=strategy,
        training=False,
        iterative=3,
        samples=5
    )


    # === Load Dataset ===
    data_wrapper = load_data_wrapper_from_folder(args.model_path, args.dataset, args.batch_size)


    # ====== Predict ======
    predict(data_wrapper)

    # === Save predictions ===
    for pred, filename in zip(all_preds, all_filenames):
        # Convert TensorFlow bytes to str
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")
        elif hasattr(filename, "numpy"):
            filename = filename.numpy().decode("utf-8")

        # Hậu xử lý
        pred_mask = (pred[..., 0] >= 0.5).astype(np.uint8) * 255
        save_path = os.path.join(save_dir, filename)
        imageio.imwrite(save_path, pred_mask)

    print(f"✅ Saved {len(all_preds)} predictions to {save_dir}")
