import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tqdm import tqdm
import math
import yaml
import imageio.v3 as imageio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import warnings

from model.unet import IterativeUnet, VanilaUnet

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
    required_keys = ["model", "image_channel", "dropout_rate", "use_batchnorm", "add_channel"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in setting.yaml.")

    # Sau khi đã kiểm tra đầy đủ, gán giá trị
    model_type = config["model"]
    image_channel = config["image_channel"]
    dropout_rate = config["dropout_rate"]
    use_batchnorm = config["use_batchnorm"]
    add_channel = config["add_channel"]

    # 3. Build model
    input_channels = image_channel + (1 if add_channel else 0)
    if model_type == "iterative":
        model = IterativeUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    elif model_type == "vanila":
        model = VanilaUnet(input_channels=input_channels, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}' found in setting.yaml")
    model.build((1, 256, 256, input_channels))


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
    # ✅ In ra model config
    print()
    print("========== Model Config ==========")
    print(f"Model type     : {model_type}")
    print(f"Input channels : {input_channels}")
    print(f"Dropout rate   : {dropout_rate}")
    print(f"Use BatchNorm  : {use_batchnorm}")
    print("==================================")
    print()

    return model

def build_distributed_predict_step(
    model,
    strategy,
    training: bool = False,
    iterative: int = 1,
    samples: int = 1
):
    @tf.function(reduce_retracing=True)
    def distributed_predict_step(batch):
        def step_fn(images, masks, filenames):
            images = tf.cast(images, tf.float16)  # mixed precision
            batch_size = tf.shape(images)[0]

            # Kết quả shape: [samples, batch, iterative, H, W, 1]
            result_array = tf.TensorArray(dtype=tf.float32, size=samples)

            for s in tf.range(samples):
                iter_array = tf.TensorArray(dtype=tf.float32, size=iterative)
                zero_channel = tf.zeros_like(images[..., :1], dtype=tf.float16)

                for i in tf.range(iterative):
                    if iterative > 1:
                        input_images = tf.concat([images, zero_channel], axis=-1)
                    else:
                        input_images = images

                    y_pred = model(input_images, training=training)
                    iter_array = iter_array.write(i, tf.cast(y_pred, tf.float32))  # ensure float32 loss & output

                    if iterative > 1:
                        zero_channel = tf.cast(y_pred, tf.float16)

                # [iterative, batch, H, W, 1] → [batch, iterative, H, W, 1]
                iter_stack = tf.transpose(iter_array.stack(), [1, 0, 2, 3, 4])
                result_array = result_array.write(s, iter_stack)

            # [samples, batch, iterative, H, W, 1] → [batch, samples, iterative, H, W, 1]
            final_stack = tf.transpose(result_array.stack(), [1, 0, 2, 3, 4, 5])

            return final_stack, masks, filenames

        images, masks, filenames = batch
        return strategy.run(step_fn, args=(images, masks, filenames))

    return distributed_predict_step

def _save_one_sample(pred_samples, file_name, mask, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]

    # Save mask
    mask_image = np.clip(mask[..., 0], 0.0, 1.0) * 255
    mask_image = mask_image.astype("uint8")
    mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
    imageio.imwrite(mask_path, mask_image)

    # Save predictions
    for sample_idx, iter_preds in enumerate(pred_samples):
        for iter_idx, pred in enumerate(iter_preds):
            pred_slice = pred[..., 0]

            if not np.all((0.0 <= pred_slice) & (pred_slice <= 1.0)):
                raise ValueError(f"🚨 Pixel value out of [0, 1] range in sample {sample_idx}, iter {iter_idx}")

            if not np.any(pred_slice > 0.5):
                warnings.warn(
                    f"⚠️ All predicted pixels <= 0.5 in {base_name} sample {sample_idx} iter {iter_idx}",
                    stacklevel=1
                )

            pred_image = (pred_slice * 255).astype("uint8")
            out_name = f"{base_name}_sample_{sample_idx}_iter{iter_idx}.png"
            out_path = os.path.join(save_dir, out_name)
            imageio.imwrite(out_path, pred_image)


def save_all_predictions(all_preds, all_filenames, all_masks, save_dir, num_workers=8):
    os.makedirs(save_dir, exist_ok=True)

    assert len(all_preds) == len(all_filenames) == len(all_masks), \
        "Mismatch in number of predictions, filenames, or masks!"

    print(" === Predictions saving ... ===")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = [
            executor.submit(_save_one_sample, pred, fname, mask, save_dir)
            for pred, fname, mask in zip(all_preds, all_filenames, all_masks)
        ]

        for future in tqdm(tasks, desc="Saving predictions"):
            future.result()

    print(f"✅ Saved all predictions and masks to: {save_dir}")


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

        # ✅ Gather predictions safely (if preds_nested is nested structure)
        gathered_preds = tf.nest.map_structure(lambda x: strategy.gather(x, axis=0), preds_nested)
        gathered_preds = gathered_preds.numpy()  # shape: [batch, samples, iterative, H, W, 1]

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
    # === Logging thông tin ===
    print("🚀 Starting distributed prediction")
    print(f"📁 Model checkpoint folder   : {model_path}")
    print(f"📦 Output save path          : {save_path}")
    print(f"🔁 Iterative steps           : {iterative}")
    print(f"🎲 Samples per input         : {samples}")
    print()

    # === Đảm bảo thư mục tồn tại ===
    os.makedirs(save_path, exist_ok=True)

    # === Load mô hình với strategy ===
    strategy = tf.distribute.MirroredStrategy()
    print(f"🧠 Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = load_model_from_folder(folder_path=model_path, epoch=epoch)

    # === Build distributed prediction function ===
    distributed_predict_step = build_distributed_predict_step(
        model=model,
        strategy=strategy,
        training=training,
        iterative=iterative,
        samples=samples
    )

    # === Run prediction ===
    all_preds, all_filenames, all_masks = predict(
        data_wrapper,
        strategy,
        distributed_predict_step
    )

    # === Save results ===
    save_all_predictions(
        all_preds=all_preds,
        all_filenames=all_filenames,
        all_masks=all_masks,
        save_dir=save_path
    )
