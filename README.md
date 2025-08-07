# 🔍 Uncertainty Estimation for Semantic Segmentation

This repository provides a complete pipeline for training, predicting, and evaluating uncertainty estimation methods for semantic segmentation tasks. The project is implemented using TensorFlow and supports multiple methods including:

- **Vanilla U-Net**
- **Iterative U-Net**
- **Monte Carlo (MC) Dropout**
- **Ensemble Models**
- **Iterative + Stochastic Sampling**

---

## 🧠 Key Features

- ✅ Supports **semantic segmentation** with uncertainty quantification.
- 🔁 Iterative feedback loop in model (Iterative U-Net).
- 🎲 Dropout-based stochastic predictions.
- 👯‍♂️ Ensemble support for robust uncertainty estimation.
- 📊 Comprehensive evaluation: segmentation + uncertainty metrics.
- 💾 Configurable CLI-based interface for training, prediction, and evaluation.

---

## 📂 Supported Datasets

- 🛣️ **RoadTracer (RT)**
- 🗺️ **Massachusetts Roads (Mass)**
- 👁️ **DRIVE (retinal blood vessels)**

All datasets are loaded via the `DatasetController` using pre-wrapped `tf.data.Dataset`.

---

## 🏗️ Model Architectures

| Model           | Description |
|----------------|-------------|
| **Vanilla U-Net** | Classic encoder-decoder U-Net with optional Dropout and BatchNorm |
| **Iterative U-Net** | U-Net variant that refines predictions across multiple feedback iterations |

Models are dynamically selected via CLI and fully support:

- Custom input channels
- Dropout
- Batch Normalization
- Iterative feedback
- Mixed precision inference

---

## 🚀 Getting Started

### 1. 🏋️ Train a Model

```bash
python train.py \
  --model iterative \
  --dataset RT \
  --dataset_path /path/to/roadtracer \
  --dropout_rate 0.3 \
  --use_batchnorm \
  --image_channel 3 \
  --add_channel \
  --batch_size 8 \
  --learning_rate 0.001 \
  --num_epoch 100 \
  --save_path ./checkpoints/iterative_rt \
  --loss_function dice_focal \
  --save_per_epoch 10 \
  --buffer_size 1000 \
  --gpus 0
