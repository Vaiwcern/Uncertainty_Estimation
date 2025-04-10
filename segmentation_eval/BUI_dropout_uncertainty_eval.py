import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
from plot import plot_unc_vs_error, plot_corr_rAULC
import pandas as pd

from evaluation import compute_ccq, compute_ccq_normal, corr, rAULC
from evaluation import get_uncertainty_by_var, get_uncertainty_by_std
from evaluation import get_error_by_abs, get_error_by_mse

NUM_ITERATION = 3
SAVE_PATH ="/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/segmentation_eval/BUI256_DROP"
OUTPUT_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/model/BUI256_DROP/predict_epoch_100/"
IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/BUI_256/test/image"
MASK_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/BUI_256/test/mask"

os.makedirs(SAVE_PATH, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_TEST_PATH) if f.endswith(".png")]
num_image = len(image_files)
print("Total images:", num_image)

def evaluate_single_image(image_name):
    try:
        name_without_ext = os.path.splitext(image_name)[0]
        mask_name = Path(image_name).name
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

        pred = np.mean(outputs, axis=0)

        var_unc = get_uncertainty_by_var(outputs, axis=0, num_rows=2, num_cols=2)
        std_unc = get_uncertainty_by_std(outputs, axis=0, num_rows=2, num_cols=2)

        abs_err = get_error_by_abs(pred, mask, num_rows=2, num_cols=2)
        mse_err = get_error_by_mse(pred, mask, num_rows=2, num_cols=2)

        return var_unc, std_unc, abs_err, mse_err

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0

if __name__ == "__main__":  
    var_uncertainties = []
    std_uncertainties = []
    abs_errors = []
    mse_errors = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(evaluate_single_image, image_files), total=num_image))

    for var_unc, std_unc, abs_err, mse_err in results:
        var_uncertainties += var_unc
        std_uncertainties += std_unc

        abs_errors += abs_err
        mse_errors += mse_err

    var_uncertainties = np.array(var_uncertainties)
    std_uncertainties = np.array(std_uncertainties)
    abs_errors = np.array(abs_errors)
    mse_errors = np.array(mse_errors)

    print("Std vs abs")
    print("Corr", corr(std_uncertainties, abs_errors))
    print("rAULR", rAULC(std_uncertainties, abs_errors))
    print("-----------------------------")

    
    print("Std vs mse")
    print("Corr", corr(std_uncertainties, mse_errors))
    print("rAULR", rAULC(std_uncertainties, mse_errors))
    print("-----------------------------")


    print("Var vs abs")
    print("Corr", corr(var_uncertainties, abs_errors))
    print("rAULR", rAULC(var_uncertainties, abs_errors))
    print("-----------------------------")


    print("Var vs mse")
    print("Corr", corr(var_uncertainties, mse_errors))
    print("rAULR", rAULC(var_uncertainties, mse_errors))
    print("-----------------------------")

    results = [
        {
            "Metric": "Std vs abs",
            "Corr": corr(std_uncertainties, abs_errors),
            "rAULR": rAULC(std_uncertainties, abs_errors)
        },
        {
            "Metric": "Std vs mse",
            "Corr": corr(std_uncertainties, mse_errors),
            "rAULR": rAULC(std_uncertainties, mse_errors)
        },
        {
            "Metric": "Var vs abs",
            "Corr": corr(var_uncertainties, abs_errors),
            "rAULR": rAULC(var_uncertainties, abs_errors)
        },
        {
            "Metric": "Var vs mse",
            "Corr": corr(var_uncertainties, mse_errors),
            "rAULR": rAULC(var_uncertainties, mse_errors)
        }
    ]

    # Chuyển thành DataFrame và lưu CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_PATH, "uncertainty_result.csv"), index=False)

    plot_pairs = [
        (std_uncertainties, abs_errors, "Std vs Abs", "std_vs_abs.png"),
        (std_uncertainties, mse_errors, "Std vs MSE", "std_vs_mse.png"),
        (var_uncertainties, abs_errors, "Var vs Abs", "var_vs_abs.png"),
        (var_uncertainties, mse_errors, "Var vs MSE", "var_vs_mse.png")
    ]

    for x, y, title, filename in plot_pairs:
        plot_corr_rAULC(x, y, title, filename, SAVE_PATH)

    print("All plots saved to folder.")



    