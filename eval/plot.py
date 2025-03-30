import matplotlib.pyplot as plt

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
