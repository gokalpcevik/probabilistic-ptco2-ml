import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing.o2_dataset import OxygenDataset

# Global style
plt.style.use("seaborn-v0_8-muted")
plt.rcParams.update({
    "font.size": 15,
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "legend.frameon": False,
})

def plot_dataset_lifetime_vs_po2(data_dir="data/processed/"):
    full_dataset = OxygenDataset(data_dir, split="train", normalize=False)
    val_dataset = OxygenDataset(data_dir, split="val", normalize=False)
    test_dataset = OxygenDataset(data_dir, split="test", normalize=False)

    all_features = np.vstack([
        full_dataset.features,
        val_dataset.features,
        test_dataset.features
    ])
    all_targets = np.concatenate([
        full_dataset.targets,
        val_dataset.targets,
        test_dataset.targets
    ])

    lifetimes = all_features[:, 0]  # Lifetime in μs
    po2 = all_targets               # pO2 in cmHg

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        lifetimes, po2,
        s=30,
        alpha=0.7,
        c=po2,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5
    )

    # Labels & Title
    ax.set_xlabel("Luminescence Lifetime (μs)")
    ax.set_ylabel("pO₂ (cmHg)")
    ax.set_title("Entire Dataset Distribution: Lifetime vs pO₂")

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("pO₂ (cmHg)", rotation=270, labelpad=15)

    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_dataset_lifetime_vs_po2()
