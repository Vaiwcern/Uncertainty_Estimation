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

from evaluation import compute_ccq, compute_ccq_normal, corr, rAULC
from evaluation import get_uncertainty_by_var, get_uncertainty_by_std
from evaluation import get_error_by_abs, get_error_by_mse

NUM_ITERATION = 3
SAVE_PATH ="/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/eval/RTdata_dropout"
OUTPUT_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_dropout_model/predict_epoch_55/"
IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop/imagery_test"
MASK_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop/masks_thick"

os.makedirs(SAVE_PATH, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_TEST_PATH) if f.endswith(".png")]
num_image = len(image_files)
print("Total images:", num_image)

import numpy as np

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

        variances = []
        for i in range(5):
            output_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_output_{i}.png") 
            img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] File not found or unreadable: {output_path}")
                return 0, 0, 0, 0
            img = img.astype(np.float32) / 255.0
            variances.append(img)

        pred = np.mean(variances, axis=0)

        return variances, pred, mask

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

    for variances, pred, mask in results:
        var_uncertainties += get_uncertainty_by_var(variances, num_rows=2, num_cols=2)
        std_uncertainties += get_uncertainty_by_std(variances, num_rows=2, num_cols=2)

        abs_errors += get_error_by_abs(pred, mask, num_rows=2, num_cols=2)
        mse_errors += get_error_by_mse(pred, mask, num_rows=2, num_cols=2)

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

    plot_pairs = [
        (std_uncertainties, abs_errors, "Std vs Abs", "std_vs_abs.png"),
        (std_uncertainties, mse_errors, "Std vs MSE", "std_vs_mse.png"),
        (var_uncertainties, abs_errors, "Var vs Abs", "var_vs_abs.png"),
        (var_uncertainties, mse_errors, "Var vs MSE", "var_vs_mse.png")
    ]

    for x, y, title, filename in plot_pairs:
        plot_corr_rAULC(x, y, title, filename)

    print("All plots saved to folder.")



    