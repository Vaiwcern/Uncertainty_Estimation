import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import matplotlib.pyplot as plt

from evaluation import compute_ccq, compute_ccq_normal, corr, rAULC

NUM_ITERATION = 3
SAVE_PATH ="/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/eval/RTdata_mc_dropout_ver2"
OUTPUT_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_mc_dropout_model_ver2/predict_epoch_55/"
IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop/imagery_test"
MASK_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop/masks_thick"

os.makedirs(SAVE_PATH, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_TEST_PATH) if f.endswith(".png")]
num_image = len(image_files)
print("Total images:", num_image)

import numpy as np

def split_and_mean(array, num_rows=2, num_cols=2):
    """
    Chia mảng 2D thành nhiều phần bằng nhau và tính mean của từng phần.

    Parameters:
        array (np.ndarray): Mảng đầu vào 2D.
        num_rows (int): Số hàng muốn chia.
        num_cols (int): Số cột muốn chia.

    Returns:
        crops (List[np.ndarray]): Danh sách các mảng con.
        means (List[float]): Danh sách các giá trị mean của từng crop.
    """
    h, w = array.shape
    assert h % num_rows == 0 and w % num_cols == 0, "Kích thước không chia hết!"

    crop_h = h // num_rows
    crop_w = w // num_cols

    means = []

    for i in range(num_rows):
        for j in range(num_cols):
            crop = array[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            means.append(np.mean(crop))

    return means

def evaluate_single_image(image_name):
    try:
        name_without_ext = os.path.splitext(image_name)[0]
        mask_name = f"{'_'.join(Path(image_name).stem.split('_')[:-4])}_osm_{'_'.join(Path(image_name).stem.split('_')[4:])}.png"
        mask_path = os.path.join(MASK_PATH, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            return 0, 0, 0, 0
        mask = (mask >= 128).astype(np.uint8)

        outputs = []
        for i in range(5):
            output_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_output_{i}.png") 
            img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] File not found or unreadable: {output_path}")
                return 0, 0, 0, 0
            img = img.astype(np.float32) / 255.0
            outputs.append(img)

        # Cal uncertainty ouput
        uncertainty = np.var(outputs, axis=0)
        uncs = split_and_mean(uncertainty, num_rows=2, num_cols=2)

        # Cal error
        # pred = np.mean(outputs, axis=0)
        pred = np.mean(outputs, axis=0)
        error = np.abs(pred - mask) ** 2
        errors = split_and_mean(error, num_rows=2, num_cols=2)

        return uncs, errors

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0


def plot_unc_vs_error(x, y, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.4)
    plt.xlabel("Uncertainty (sqrt)")
    plt.ylabel("Prediction Error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 


if __name__ == "__main__":  
    uncertainties = []
    errors = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(evaluate_single_image, image_files), total=num_image))

    for unc_crops, error_crops in results:
        uncertainties += unc_crops
        errors += error_crops

    uncertainties = np.array(uncertainties)
    errors = np.array(errors)

    print("Std vs abs")
    print(corr(uncertainties**0.5, errors**0.5))
    print(rAULC(uncertainties**0.5, errors**0.5))

    print("-----------------------------")

    print("Std vs mse")
    print(corr(uncertainties**0.5, errors))
    print(rAULC(uncertainties**0.5, errors))

    print("-----------------------------")

    print("Var vs abs")
    print(corr(uncertainties, errors**0.5))
    print(rAULC(uncertainties, errors**0.5))

    print("-----------------------------")


    print("Var vs mse")
    print(corr(uncertainties, errors))
    print(rAULC(uncertainties, errors))

    print("-----------------------------")

    def plot_corr_rAULC(x, y, title, filename):
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, s=10, alpha=0.4)
        plt.xlabel("Uncertainty")
        plt.ylabel("Error")
        plt.title(f"{title}\nCorr={corr(x, y):.4f} | rAULC={rAULC(x, y):.4f}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, filename))
        plt.close()

    plot_pairs = [
        (uncertainties**0.5, errors**0.5, "Std vs Abs", "std_vs_abs.png"),
        (uncertainties**0.5, errors, "Std vs MSE", "std_vs_mse.png"),
        (uncertainties, errors**0.5, "Var vs Abs", "var_vs_abs.png"),
        (uncertainties, errors, "Var vs MSE", "var_vs_mse.png")
    ]

    for x, y, title, filename in plot_pairs:
        plot_corr_rAULC(x, y, title, filename)

    print("All plots saved to folder.")



    