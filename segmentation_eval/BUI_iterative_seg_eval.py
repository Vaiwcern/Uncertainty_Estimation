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


from evaluation import compute_ccq, compute_ccq_normal

NUM_ITERATION = 3
SAVE_PATH ="/home/ltnghia02/MEDICAL_ITERATIVE/Uncertainty_Estimation/segmentation_eval/BUI_iter"
OUTPUT_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/model/BUI_iter/predict_epoch_100/"
IMAGE_TEST_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/BUI_256/test/image"
MASK_PATH = "/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/BUI_256/test/mask"

os.makedirs(SAVE_PATH, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_TEST_PATH) if f.endswith(".png")]
num_image = len(image_files)
print("Total images:", num_image)

def get_pred(outputs):
    avg_pred = np.mean(outputs, axis=0)
    return avg_pred

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
        grads = []
        for i in range(NUM_ITERATION):
            output_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_output_{i}.png") 
            img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] File not found or unreadable: {output_path}")
                return 0, 0, 0, 0
            img = img.astype(np.float32) / 255.0
            outputs.append(img)

            # grad_path = os.path.join(OUTPUT_PATH, f"{name_without_ext}_grad_{i}.png") 
            # img = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
            # if img is None:
            #     print(f"[ERROR] File not found or unreadable: {grad_path}")
            #     return 0, 0, 0, 0
            # img = img.astype(np.float32) / 255.0
            # grads.append(img)

        # pred = np.mean(outputs, axis=0)
        pred = outputs[-1]
        corr, comp, qual, f1 = compute_ccq(pred, mask, threshold=0.5, slack=5)
        # corr, comp, qual, f1 = compute_ccq_normal(pred, mask, threshold=0.5)
        return corr, comp, qual, f1

    except Exception as e:
        print(f"[ERROR] Exception while processing {image_name}: {e}")
        return 0, 0, 0, 0


if __name__ == "__main__":
    correctness = 0
    completeness = 0
    quality = 0
    f1_score = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(evaluate_single_image, image_files), total=num_image))

    for corr, comp, qual, f1 in results:
        correctness += corr
        completeness += comp
        quality += qual
        f1_score += f1

    print("\n===> AVERAGE CCQ RESULT <===")
    print(f"Correctness:  {correctness / num_image:.4f}")
    print(f"Completeness: {completeness / num_image:.4f}")
    print(f"Quality:      {quality / num_image:.4f}")
    print(f"F1 Score:     {f1_score / num_image:.4f}")

    save_file = os.path.join(SAVE_PATH, "seg_avg_results.csv")

    with open(save_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header with chú thích
        writer.writerow([
            "Correctness (Precision)",
            "Completeness (Recall)",
            "Quality (IoU)",
            "F1 Score"
        ])
        # 1 hàng giá trị
        writer.writerow([
            round(correctness / num_image, 4),
            round(completeness / num_image, 4),
            round(quality / num_image, 4),
            round(f1_score / num_image, 4)
        ])

    print(f"\n✅ Results saved to: {save_file}")
