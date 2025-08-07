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

## ⚙️ Environment Setup

To set up your development environment for this project, follow the steps below:

### 📥 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 🐍 2. Create a Virtual Environment

It is recommended to use Python 3.8+.

```bash
python -m venv venv
```

### ▶️ 3. Activate the Virtual Environment

- On **Linux/macOS**:

```bash
source venv/bin/activate
```

- On **Windows**:

```bash
venv\Scripts\activate.bat
```

### 📦 4. Install Dependencies

Install all required packages using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ✅ Tip: Using a virtual environment helps avoid dependency conflicts with other projects.

## 🏋️‍♂️ Training Usage

To train a model, use the following command:

```bash
python train.py \
  --model iterative \
  --dataset RT \
  --dataset_path ./data/roadtracer \
  --batch_size 8 \
  --num_epoch 100 \
  --save_path ./checkpoints/iterative_rt \
  --dropout_rate 0.3 \
  --use_batchnorm \
  --image_channel 3 \
  --add_channel \
  --learning_rate 0.001 \
  --loss_function dice_focal \
  --save_per_epoch 10 \
  --buffer_size 1000 \
  --gpus 0
```

### 🧾 Training Arguments

| Argument           | Description |
|--------------------|-------------|
| `--model`          | Model type to use: `vanila` or `iterative` |
| `--dataset`        | Dataset name: `RT`, `Mass`, or `Drive` |
| `--dataset_path`   | Path to the dataset directory |
| `--batch_size`     | Number of samples per training batch |
| `--num_epoch`      | Number of training epochs |
| `--save_path`      | Directory to save model checkpoints |
| `--dropout_rate`   | Dropout rate for regularization (e.g., 0.3) |
| `--use_batchnorm`  | Include Batch Normalization if specified |
| `--image_channel`  | Number of channels in input image (usually 3 for RGB) |
| `--add_channel`    | Add a feedback channel for iterative models |
| `--learning_rate`  | Initial learning rate |
| `--loss_function`  | Loss function to use: `focal`, `iou`, `bce`, `dice`, `dice_bce`, `dice_focal` |
| `--save_per_epoch` | Save model every N epochs |
| `--buffer_size`    | Buffer size for shuffling the dataset |
| `--gpus`           | Comma-separated GPU device IDs (e.g., `0`, `0,1`) |


---

## 🔮 Prediction Usage

Run inference using a trained model with dropout and/or iterative refinement:

```bash
python predict.py \
  --dataset RT \
  --dataset_path ./data/roadtracer \
  --model_path ./checkpoints/iterative_rt \
  --save_path ./predictions/iterative_rt \
  --epoch 90 \
  --training_mode \
  --batch_size 1 \
  --iterative 3 \
  --samples 10 \
  --gpus 0
```

### 🧾 Prediction Arguments

| Argument            | Description |
|---------------------|-------------|
| `--dataset`         | Dataset name: `RT`, `Mass`, or `Drive` |
| `--dataset_path`    | Path to test dataset |
| `--model_path`      | Directory where model checkpoint is saved |
| `--save_path`       | Path to save prediction outputs |
| `--epoch`           | Which epoch to load checkpoint from |
| `--training_mode`   | Enable dropout during inference |
| `--batch_size`      | Number of samples per prediction step |
| `--iterative`       | Number of iterative refinement passes |
| `--samples`         | Number of stochastic samples per input |
| `--gpus`            | GPU device IDs to use (e.g., `0`) |


---

## 📊 Evaluation Usage

Evaluate predictions using segmentation or uncertainty metrics:

```bash
python evaluate.py \
  --dataset RT \
  --dataset_path ./data/roadtracer \
  --prediction_path ./predictions/iterative_rt \
  --save_path ./evaluation/iterative_rt \
  --eval_type uncertainty \
  --iterative 3 \
  --samples 10 \
  --n_rows 2 \
  --n_cols 2 \
  --relaxed_ccq \
  --gpus 0
```

### 🧾 Evaluation Arguments

| Argument              | Description |
|-----------------------|-------------|
| `--dataset`           | Dataset name: `RT`, `Mass`, or `Drive` |
| `--dataset_path`      | Path to dataset |
| `--prediction_path`   | Directory containing saved predictions |
| `--save_path`         | Path to save evaluation results |
| `--eval_type`         | Type of evaluation: `segmentation`, `uncertainty`, or `out-of-distribution` |
| `--iterative`         | Number of iterative passes used during prediction |
| `--samples`           | Number of stochastic samples used |
| `--n_rows`            | Number of rows for region-based uncertainty cropping |
| `--n_cols`            | Number of columns for region-based uncertainty cropping |
| `--relaxed_ccq`       | Use relaxed CCQ evaluation metric with slack |
| `--ood_dataset`       | OOD dataset name (only for `out-of-distribution` eval) |
| `--ood_dataset_path`  | Path to OOD dataset |
| `--gpus`              | GPU device IDs to use |
| `--num_workers`       | Number of worker threads (optional) |
| `--ensembles`         | Number of ensemble models (optional) |
