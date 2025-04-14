import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tqdm import tqdm
import math
import yaml
import imageio.v3 as imageio
import numpy as np

from model.unet import IterativeUnet, VanilaUnet
from custom_dataset.DatasetController import DatasetController

def load_yaml_config(path): 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Setting file not found at: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

def load_model_from_folder(folder_path: str, epoch: int):
    # 1. Load YAML config
    setting_path = os.path.join(folder_path, "setting.yaml")
    config = load_yaml_config(setting_path)

    # 2. Extract required model settings
    required_keys = ["model", "image_channel", "dropout_rate", "use_batchnorm"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in setting.yaml.")

    # Sau khi đã kiểm tra đầy đủ, gán giá trị
    model_type = config["model"]
    image_channel = config["image_channel"]
    dropout_rate = config["dropout_rate"]
    use_batchnorm = config["use_batchnorm"]

    # 3. Build model
    if model_type == "iterative":
        model = IterativeUnet(image_channel=image_channel, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    elif model_type == "vanila":
        model = VanilaUnet(image_channel=image_channel, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}' found in setting.yaml")

    # 4. Load weights
    weight_filename = f"model_epoch_{epoch}.weights.h5"
    weight_path = os.path.join(folder_path, weight_filename)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at: {weight_path}")

    try:
        model.load_weights(weight_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from '{weight_path}'. Error: {str(e)}")

    print(f"✅ Model '{model_type}' successfully loaded from epoch {epoch} ({weight_path})")
    return model, config

def build_distributed_predict_step(
    model,
    strategy,
    training: bool = False,
    iterative: int = 1,
    samples: int = 1
):
    @tf.function
    def distributed_predict_step(batch):
        def step_fn(images, masks, filenames):
            images = tf.cast(images, tf.float16)
            results_per_sample = []

            for _ in tf.range(samples):
                per_iter_outputs = []

                x = tf.cast(images, tf.float16)  
                if iterative > 1:
                    zero_channel = tf.zeros_like(x[..., :1], dtype=tf.float16)

                for _ in tf.range(iterative):
                    if iterative > 1:
                        images_input = tf.concat([x, zero_channel], axis=-1)
                    else:
                        images_input = x 

                    y_pred = model(images_input, training=training)
                    per_iter_outputs.append(y_pred)

                    if iterative > 1:
                        zero_channel = tf.cast(y_pred, tf.float16)

                results_per_sample.append(per_iter_outputs)

            return results_per_sample, masks, filenames

        images, masks, filenames = batch
        return strategy.run(step_fn, args=(images, masks, filenames))

    return distributed_predict_step

def save_all_predictions(all_preds, all_filenames, all_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i, (pred_samples, file_name, mask) in enumerate(zip(all_preds, all_filenames, all_masks)):
        # Remove file extension
        base_name = os.path.splitext(file_name)[0]

        # Save mask (no threshold)
        mask_image = (mask[..., 0] * 255).astype("uint8")  # Keep as grayscale float
        mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        imageio.imwrite(mask_path, mask_image)

        # Save predictions (no threshold)
        for sample_idx, iter_preds in enumerate(pred_samples):
            for iter_idx, pred in enumerate(iter_preds):
                pred_image = (pred[..., 0] * 255).astype("uint8")  # raw float → 0–255
                out_name = f"{base_name}_sample_{sample_idx}_iter{iter_idx}.png"
                out_path = os.path.join(save_dir, out_name)
                imageio.imwrite(out_path, pred_image)

    print(f"✅ Saved all predictions and masks (without threshold) to: {save_dir}")

def predict(data_wrapper, strategy, distributed_predict_step):
    val_dataset = data_wrapper.dataset

    num_images = len(data_wrapper.image_files)
    steps_per_epoch = math.ceil(num_images / data_wrapper.batch_size)
    print("Total images:", num_images)
    print("Steps per epoch:", steps_per_epoch)

    all_filenames = []
    all_masks = []
    all_preds = []

    for batch in tqdm(val_dataset.take(steps_per_epoch)):
        preds_nested, masks, filenames = distributed_predict_step(batch)

        # Gather predictions, filenames, masks from all replicas
        gathered_preds = strategy.gather(preds_nested, axis=0).numpy()           # shape: [batch, samples, iter, H, W, 1]
        gathered_filenames = strategy.gather(filenames, axis=0).numpy()
        gathered_masks = strategy.gather(masks, axis=0).numpy()

        all_preds.extend(gathered_preds)
        all_filenames.extend([f.decode("utf-8") if isinstance(f, bytes) else f for f in gathered_filenames])
        all_masks.extend(gathered_masks)

    return all_preds, all_filenames, all_masks

def predict_and_save_results(
    model_path: str,
    epoch: int,
    data_wrapper,
    save_path: str,
    training: bool,
    iterative: int,
    samples: int
) -> None:
    
    # === Load model ===
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model, config = load_model_from_folder(folder_path=model_path, epoch=epoch)

    print("========= Model config =========")
    print(config)

    # === Build distributed prediction function ===
    distributed_predict_step = build_distributed_predict_step(
        model=model,
        strategy=strategy,
        training=training,
        iterative=iterative,
        samples=samples
    )

    # === Predict ===
    all_preds, all_filenames, all_masks = predict(data_wrapper, strategy, distributed_predict_step)

    save_all_predictions(all_preds, all_filenames, all_masks, save_path)
