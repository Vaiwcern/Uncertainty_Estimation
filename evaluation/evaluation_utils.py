import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import tensorflow as tf

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

def load_data_wrapper_from_folder(folder_path: str, dataset: str, batch_size: int):
    # 1. Load YAML config
    setting_path = os.path.join(folder_path, "setting.yaml")
    config = load_yaml_config(setting_path)

    # 2. Extract required model settings
    required_keys = ["dataset_path", "add_channel"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in setting.yaml.")

    # Sau khi đã kiểm tra đầy đủ, gán giá trị
    dataset_path = config["dataset_path"]
    add_channel = config["add_channel"]

    if dataset == "RT":
        data_wrapper = DatasetController.get_roadtracer_train_wrapper(
            dataset_path=dataset_path,
            batch_size=batch_size,
            add_channel=add_channel,
        )
    elif dataset == "Mass":
        data_wrapper = DatasetController.get_massachusetts_train_wrapper(
            dataset_path=dataset_path,
            batch_size=batch_size,
            add_channel=add_channel,
        )
    else: 
        raise ValueError(f"Unsupported dataset: {dataset}")

    