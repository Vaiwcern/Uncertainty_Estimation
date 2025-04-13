import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
from plot import plot_unc_vs_error, plot_corr_rAULC, plot_uncertainty_bar_chart
import pandas as pd

from evaluation import compute_ccq, compute_ccq_normal, corr, rAULC
from evaluation import get_uncertainty_by_var, get_uncertainty_by_std
from evaluation import get_error_by_abs, get_error_by_mse
from evaluation import cal_roc_auc, cal_pr_auc, min_max_normalize

NUM_ITERATION = 3

def evaluate_single_image(args):
    image_name, OUTPUT_PATH, num_rows, num_cols = args

    try:
        name_without_ext = os.path.splitext(image_name)[0]

        outputs = []
        for i in range(NUM_ITERATION):
            output_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_output_{i}.png") 
            img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] File not found or unreadable: {output_path}")
                return 0, 0, 0, 0
            img = img.astype(np.float32) / 255.0
            outputs.append(img)

        var_unc = get_uncertainty_by_var(outputs, axis=0, num_rows=num_rows, num_cols=num_cols)
        std_unc = get_uncertainty_by_std(outputs, axis=0, num_rows=num_rows, num_cols=num_cols)

        return var_unc, std_unc

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0

if __name__ == "__main__":  

    EPOCH = 55
    SAVE_PATH = f"/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/segmentation_eval/RTdata_iterative_{EPOCH}"
    RT_OUTPUT_PATH = f"/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_iterative_model/predict_epoch_{EPOCH}/"
    MASS_OUTPUT_PATH = f"/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_iterative_model/predict_epoch_{EPOCH}_mass/"

    RT_IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop/imagery_test"
    MASS_IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/Massachusetts_Crop/tiff/test"

    os.makedirs(SAVE_PATH, exist_ok=True)   

    RT_image_files = [f for f in os.listdir(RT_IMAGE_TEST_PATH) if f.endswith(".png")]
    num_RT_image = len(RT_image_files)
    print("Total RT images:", num_RT_image)

    Mass_image_files = [f for f in os.listdir(MASS_IMAGE_TEST_PATH) if f.endswith(".tif")]
    num_Mass_image = len(Mass_image_files)
    print("Total Mass images:", num_Mass_image)

    RT_var_unc = []
    RT_std_unc = []
    Mass_var_unc = []
    Mass_std_unc = []

    rt_args = [(img, RT_OUTPUT_PATH, 2, 2) for img in RT_image_files]
    mass_args = [(img, MASS_OUTPUT_PATH, 1, 1) for img in Mass_image_files]

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(evaluate_single_image, mass_args), total=num_Mass_image))
    for var_unc, std_unc in results:
        Mass_var_unc += var_unc
        Mass_std_unc += std_unc

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(evaluate_single_image, rt_args), total=num_RT_image))
    for var_unc, std_unc in results:
        RT_var_unc += var_unc
        RT_std_unc += std_unc

    RT_var_unc = np.array(RT_var_unc)
    RT_std_unc = np.array(RT_std_unc)
    Mass_var_unc = np.array(Mass_var_unc)
    Mass_std_unc = np.array(Mass_std_unc)

    RT_label = np.zeros_like(RT_var_unc)
    Mass_label = np.ones_like(Mass_var_unc)

    print(f"RT len: {len(RT_var_unc)}")
    print(f"Mass len: {len(Mass_std_unc)}")

    data_list = [RT_var_unc, RT_std_unc, Mass_var_unc, Mass_std_unc]
    titles = ['RT_var_unc', 'RT_std_unc', 'Mass_var_unc', 'Mass_std_unc']

    var_uncertainties = np.concatenate((RT_var_unc, Mass_var_unc))
    std_uncertainties = np.concatenate((RT_std_unc, Mass_std_unc))
    label = np.concatenate((RT_label, Mass_label))

    var_uncertainties = min_max_normalize(var_uncertainties)
    std_uncertainties = min_max_normalize(std_uncertainties)
    
    path_1 = os.path.join(SAVE_PATH, f"chart_var_uncertainties.png")
    plot_uncertainty_bar_chart(var_uncertainties, f"chart_var_uncertainties.png", path_1)

    path_2 = os.path.join(SAVE_PATH, f"chart_std_uncertainties.png")
    plot_uncertainty_bar_chart(std_uncertainties, f"chart_std_uncertainties.png", path_2)

    print(var_uncertainties)
    print(std_uncertainties)
    print(label)

    var_roc_auc = cal_roc_auc(label, var_uncertainties)
    var_pr_auc = cal_pr_auc(label, var_uncertainties)

    std_roc_auc = cal_roc_auc(label, std_uncertainties)
    std_pr_auc = cal_pr_auc(label, std_uncertainties)

    # In kết quả
    print(f"var_roc_auc: {var_roc_auc}")
    print(f"var_pr_auc: {var_pr_auc}")
    print(f"std_roc_auc: {std_roc_auc}")
    print(f"std_pr_auc: {std_pr_auc}")

    # Lưu vào CSV
    csv_path = os.path.join(SAVE_PATH, "ood_result.csv")

    df = pd.DataFrame([{
        "var_roc_auc": var_roc_auc,
        "var_pr_auc": var_pr_auc,
        "std_roc_auc": std_roc_auc,
        "std_pr_auc": std_pr_auc
    }])

    df.to_csv(csv_path, index=False)








    