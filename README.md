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
