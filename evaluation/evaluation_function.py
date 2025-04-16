import os
import csv
import numpy as np
import imageio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from metric import compute_ccq, compute_ccq_normal, corr, rAULC
from metric import get_uncertainty_by_var, get_uncertainty_by_std
from metric import get_error_by_abs, get_error_by_mse, compute_ece

from plot import plot_corr_rAULC

def get_pred_n_mask(base_name, pred_dir, iterative, samples): 
    # --- Load mask ---
    mask_path = os.path.join(pred_dir, f"{base_name}_mask.png")
    mask = imageio.imread(mask_path)
    if mask is None:
        print(f"[ERROR] Cannot read mask: {mask_path}")
        return 0, 0

    mask = (mask >= 128).astype(np.uint8)

    # --- Load all predictions: all iterations for each sample ---
    final_preds = []  # shape: [samples, iterative, H, W]
    for s in range(samples):
        sample_preds = []
        for it in range(iterative):
            pred_filename = f"{base_name}_sample_{s}_iter{it}.png"
            pred_path = os.path.join(pred_dir, pred_filename)

            if not os.path.exists(pred_path):
                print(f"[ERROR] Missing prediction: {pred_path}")
                return 0, 0

            pred_img = imageio.imread(pred_path).astype(np.float32) / 255.0
            sample_preds.append(pred_img)

        final_preds.append(sample_preds)

    final_preds = np.array(final_preds, dtype=np.float32)  # shape: [samples, iterative, H, W]
    return final_preds, mask


def segmentation_evaluate_single_image(args):
    image_name, iterative, samples, pred_dir, relax = args
    try:
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        final_preds, mask = get_pred_n_mask(base_name, pred_dir, iterative, samples)

        # --- Average samples at the last iteration ---
        last_iter = iterative - 1
        pred = np.mean(final_preds[:, last_iter, :, :], axis=0)

        # --- Compute metrics ---
        if relax:
            corr, comp, qual, f1 = compute_ccq(pred, mask, threshold=0.5, slack=5)
        else:
            corr, comp, qual, f1 = compute_ccq_normal(pred, mask, threshold=0.5, slack=5)
        return corr, comp, qual, f1

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0

def uncertainty_evaluate_single_image(args):
    image_name, iterative, samples, pred_dir = args
    try:
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        final_preds, mask = get_pred_n_mask(base_name, pred_dir, iterative, samples)

        # --- Warning if samples duplicate
        for i in range(samples):
            for j in range(i + 1, samples):
                if np.array_equal(final_preds[i], final_preds[j]):
                    print(f"⚠️ Warning: sample {i} và sample {j} giống hệt nhau!")

        # --- Average samples at the last iteration ---
        last_iter = iterative - 1
        pred = np.mean(final_preds[:, last_iter, :, :], axis=0)

        if final_preds.shape[0] == 1:
            preds = final_preds[0, :, :, :]  # shape: (iterative, H, W)
        else:
            preds = final_preds[:, last_iter, :, :]  # shape: (samples, H, W)

        var_unc = get_uncertainty_by_var(preds, axis=0, num_rows=2, num_cols=2)
        std_unc = get_uncertainty_by_std(preds, axis=0, num_rows=2, num_cols=2)

        abs_err = get_error_by_abs(pred, mask, num_rows=2, num_cols=2)
        mse_err = get_error_by_mse(pred, mask, num_rows=2, num_cols=2)

        return var_unc, std_unc, abs_err, mse_err

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0


def segmentation_evaluation(
        data_wrapper, 
        iterative: int, 
        samples: int, 
        pred_dir: str, 
        relax: bool, 
        save_path: str, 
        num_workers: int):
    
    correctness = 0
    completeness = 0
    quality = 0
    f1_score = 0

    num_images = len(data_wrapper.image_files)
    args_list = [(img, iterative, samples, pred_dir, relax) for img in data_wrapper.image_files]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(segmentation_evaluate_single_image, args_list), total=num_images))

    for corr, comp, qual, f1 in results:
        correctness += corr
        completeness += comp
        quality += qual
        f1_score += f1

    print("\n===> AVERAGE CCQ RESULT <===")
    print(f"Correctness:  {correctness / num_images:.4f}")
    print(f"Completeness: {completeness / num_images:.4f}")
    print(f"Quality:      {quality / num_images:.4f}")
    print(f"F1 Score:     {f1_score / num_images:.4f}")

    save_file = os.path.join(save_path, "seg_avg_results.csv")

    with open(save_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Correctness (Precision)",
            "Completeness (Recall)",
            "Quality (IoU)",
            "F1 Score"
        ])
        writer.writerow([
            round(correctness / num_images, 4),
            round(completeness / num_images, 4),
            round(quality / num_images, 4),
            round(f1_score / num_images, 4)
        ])

    print(f"\n✅ Results saved to: {save_file}")

def uncertainty_evaluation(
        data_wrapper, 
        iterative: int, 
        samples: int, 
        pred_dir: str, 
        save_path: str, 
        num_workers: int):
    
    var_uncertainties = []
    std_uncertainties = []
    abs_errors = []
    mse_errors = []

    num_images = len(data_wrapper.image_files)
    args_list = [(img, iterative, samples, pred_dir) for img in data_wrapper.image_files]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(uncertainty_evaluate_single_image, args_list), total=num_images))

    for var_unc, std_unc, abs_err, mse_err in results:
        var_uncertainties += var_unc
        std_uncertainties += std_unc

        abs_errors += abs_err
        mse_errors += mse_err

    var_uncertainties = np.array(var_uncertainties)
    std_uncertainties = np.array(std_uncertainties)
    abs_errors = np.array(abs_errors)
    mse_errors = np.array(mse_errors)

    print("var_uncertainty", var_uncertainties)
    print("std_uncertainty", std_uncertainties)
    print("abs_errors", abs_errors)
    print("mse_errors", mse_errors)


    print("Std vs abs")
    print("Corr", corr(std_uncertainties, abs_errors))
    print("rAULC", rAULC(std_uncertainties, abs_errors))
    print("ECE", compute_ece(std_uncertainties, abs_errors))
    print("-----------------------------")

    
    print("Std vs mse")
    print("Corr", corr(std_uncertainties, mse_errors))
    print("rAULC", rAULC(std_uncertainties, mse_errors))
    print("ECE", compute_ece(std_uncertainties, mse_errors))
    print("-----------------------------")


    print("Var vs abs")
    print("Corr", corr(var_uncertainties, abs_errors))
    print("rAULC", rAULC(var_uncertainties, abs_errors))
    print("ECE", compute_ece(var_uncertainties, abs_errors))
    print("-----------------------------")


    print("Var vs mse")
    print("Corr", corr(var_uncertainties, mse_errors))
    print("rAULR", rAULC(var_uncertainties, mse_errors))
    print("ECE", compute_ece(var_uncertainties, mse_errors))
    print("-----------------------------")

    results = [
        {
            "Metric": "Std vs abs",
            "Corr": corr(std_uncertainties, abs_errors),
            "rAULR": rAULC(std_uncertainties, abs_errors),
            "ECE": compute_ece(std_uncertainties, abs_errors)
        },
        {
            "Metric": "Std vs mse",
            "Corr": corr(std_uncertainties, mse_errors),
            "rAULR": rAULC(std_uncertainties, mse_errors),
            "ECE": compute_ece(std_uncertainties, mse_errors)
        },
        {
            "Metric": "Var vs abs",
            "Corr": corr(var_uncertainties, abs_errors),
            "rAULR": rAULC(var_uncertainties, abs_errors),
            "ECE": compute_ece(var_uncertainties, abs_errors)
        },
        {
            "Metric": "Var vs mse",
            "Corr": corr(var_uncertainties, mse_errors),
            "rAULR": rAULC(var_uncertainties, mse_errors),
            "ECE": compute_ece(var_uncertainties, mse_errors)
        }
    ]

    # Chuyển thành DataFrame và lưu CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, "uncertainty_result.csv"), index=False)

    plot_pairs = [
        (std_uncertainties, abs_errors, "Std vs Abs", "std_vs_abs.png"),
        (std_uncertainties, mse_errors, "Std vs MSE", "std_vs_mse.png"),
        (var_uncertainties, abs_errors, "Var vs Abs", "var_vs_abs.png"),
        (var_uncertainties, mse_errors, "Var vs MSE", "var_vs_mse.png")
    ]

    for x, y, title, filename in plot_pairs:
        plot_corr_rAULC(x, y, title, filename, save_path)

    print("All plots saved to folder.")