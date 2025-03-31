import matplotlib.pyplot as plt
from evaluation import compute_ccq, compute_ccq_normal, corr, rAULC
import os
import numpy as np

def plot_uncertainty_bar_chart(array: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(max(12, len(array) * 0.1), 4))  # tự động giãn theo số lượng phần tử
    plt.bar(range(len(array)), array, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Uncertainty')
    plt.ylim(0, 1)
    plt.grid(axis='y')

    # Nếu số lượng phần tử nhỏ, in giá trị lên từng cột
    if len(array) <= 30:
        for i, val in enumerate(array):
            plt.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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


def plot_corr_rAULC(x, y, title, filename, SAVE_PATH):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=0.4)
    plt.xlabel("Uncertainty")
    plt.ylabel("Error")
    plt.title(f"{title}\nCorr={corr(x, y):.4f} | rAULC={rAULC(x, y):.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, filename))
    plt.close()
