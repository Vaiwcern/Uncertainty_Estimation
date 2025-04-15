import os
import csv
import numpy as np
import imageio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from metric import compute_ccq, compute_ccq_normal

def segmentation_evaluate_single_image(args):
    image_name, iterative, samples, pred_dir, relax = args
    try:
        base_name = os.path.splitext(os.path.basename(image_name))[0]

        # --- Load mask ---
        # print(pred_dir)
        # print(base_name)
        mask_path = os.path.join(pred_dir, f"{base_name}_mask.png")
        mask = imageio.imread(mask_path)
        if mask is None:
            print(f"[ERROR] Cannot read mask: {mask_path}")
            return 0, 0, 0, 0
        mask = (mask >= 128).astype(np.uint8)

        # --- Load predictions: pick the last iterative of each samples ---
        final_preds = []
        for s in range(samples):
            last_iter = iterative - 1  # last iterative
            pred_filename = f"{base_name}_sample_{s}_iter{last_iter}.png"
            pred_path = os.path.join(pred_dir, pred_filename)
            pred_img = imageio.imread(pred_path)

            if pred_img is None:
                print(f"[ERROR] Cannot read prediction: {pred_path}")
                return 0, 0, 0, 0

            pred_img = pred_img.astype(np.float32) / 255.0
            final_preds.append(pred_img)

        # --- Average samples ---
        pred = np.mean(final_preds, axis=0)

        # --- Compute metrics ---
        if relax:
            corr, comp, qual, f1 = compute_ccq(pred, mask, threshold=0.5, slack=5)
        else:
            corr, comp, qual, f1 = compute_ccq_normal(pred, mask, threshold=0.5, slack=5)
        return corr, comp, qual, f1

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
