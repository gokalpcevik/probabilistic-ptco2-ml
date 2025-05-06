import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP, stern_volmer_predict
from src.data_processing.o2_dataset import OxygenDataset

plt.style.use("seaborn-v0_8-muted")
plt.rcParams.update(
    {
        "font.size": 13,
        "font.family": "sans-serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "figure.dpi": 120,
        "legend.frameon": False,
    }
)

def _load_normalization_stats():
    """Try multiple known filenames to remain backward‑compatible."""
    candidate_files = [
        "models/normalization_stats_homo.npz",
        "models/normalization_stats_physics.npz",
        "models/normalization_stats_two_stage.npz",
    ]
    for fname in candidate_files:
        try:
            return np.load(fname)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No normalization stats file found in models/")


# -----------------------------------------------------------------------------
# Load models & test data
# -----------------------------------------------------------------------------

def load_models_and_data(batch_size: int = 1):
    stats = _load_normalization_stats()
    min_vals = stats["min"]
    max_vals = stats["max"]
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # avoid divide‑by‑zero


    random.seed()
    data_seed = random.randint(0,2**30)    
    test_dataset = OxygenDataset("data/processed/", split="test", normalize=True, random_seed=data_seed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Load trained weights (two‑stage) ---
    regressor = pO2Regressor(input_dim=3)
    regressor.load_state_dict(torch.load("models/regressor_stage2.pth", map_location="cpu"))
    regressor.eval()

    physics = SternVolmerMLP(input_dim=2)
    physics.load_state_dict(torch.load("models/physics_stage2.pth", map_location="cpu"))
    physics.eval()

    return regressor, physics, test_loader, min_vals, max_vals, ranges


# -----------------------------------------------------------------------------
# Testing routine
# -----------------------------------------------------------------------------

def test_two_stage(save_fig: str | None = None):
    """Evaluate two‑stage model on the held‑out test split."""
    regressor, physics, test_loader, min_vals, max_vals, ranges = load_models_and_data()

    true_po2, pred_po2 = [], []
    tau0_list, ksv_list, lifetime_meas = [], [], []

    with torch.no_grad():
        for features, target_po2 in test_loader:
            lifetime_norm, temp_norm, pulses_norm = features[:, 0], features[:, 1], features[:, 2]

            lifetime = lifetime_norm * ranges[0] + min_vals[0]
            po2_hat = regressor(features)

            physics_in = torch.stack([temp_norm, pulses_norm], dim=1)
            tau0, log_ksv, _ = physics(physics_in).T
            ksv = torch.exp(log_ksv)

            true_po2.append(target_po2.item())
            pred_po2.append(po2_hat.item())
            tau0_list.append(tau0.item())
            ksv_list.append(ksv.item())
            lifetime_meas.append(lifetime.item())

    true_po2 = np.array(true_po2)
    pred_po2 = np.array(pred_po2)

    # ---------------- Metrics ----------------
    mse = np.mean((true_po2 - pred_po2) ** 2)
    mae = np.mean(np.abs(true_po2 - pred_po2))
    r2 = 1.0 - np.sum((true_po2 - pred_po2) ** 2) / np.sum((true_po2 - np.mean(true_po2)) ** 2)

    stats_text = (
        f"$R^2$:  {r2:.4f}\n"
        f"MSE:   {mse:.3f}\n"
        f"MAE:   {mae:.3f}"
    )

    # ---------------- Plot ----------------
    plt.figure(figsize=(7, 7))
    plt.scatter(true_po2, pred_po2, alpha=0.75, s=40, edgecolors="k", linewidths=0.5, label="Predictions")

    max_lim = max(np.max(true_po2), np.max(pred_po2)) * 1.05
    plt.plot([0, max_lim], [0, max_lim], "--", color="gray", label="Ideal (y = x)")

    plt.xlabel("True pO₂ (cmHg)")
    plt.ylabel("Predicted pO₂ (cmHg)")
    plt.title("Two‑Stage Model • True vs Predicted pO₂")

    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=13,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='gray', alpha=0.9)
    )

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300)
        print(f"Figure saved to {save_fig}")
    else:
        plt.show()

    return true_po2, pred_po2, tau0_list, ksv_list, lifetime_meas


if __name__ == "__main__":
    test_two_stage()
