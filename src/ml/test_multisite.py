import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.ml.training.train_multisite import (
    MultiSiteQuenchingMLP,
    multi_site_quenching_predict,
)
from src.data_processing.o2_dataset import OxygenDataset


def load_model_and_data():
    """Load normalisation stats, test split and the trained multi‑site model."""
    stats = np.load("models/normalization_stats_multi_site.npz")
    min_vals = stats["min"]
    max_vals = stats["max"]
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # avoid divide‑by‑zero for constant columns

    test_dataset = OxygenDataset("data/processed/", split="test", normalize=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MultiSiteQuenchingMLP(input_dim=2)
    model.load_state_dict(torch.load("models/multi_site_quenching_mlp.pth"))
    model.eval()

    return model, test_loader, min_vals, max_vals, ranges


def test_multi_site():
    model, test_loader, min_vals, max_vals, ranges = load_model_and_data()

    all_true_po2 = []
    all_pred_po2 = []
    all_pred_tau0 = []
    all_pred_ksv1 = []
    all_pred_ksv2 = []
    all_pred_f1 = []
    all_measured_lifetime = []

    with torch.no_grad():
        for idx, (features, true_po2) in enumerate(test_loader, start=1):
            lifetime_norm = features[:, 0]
            temperature_norm = features[:, 1]
            pulses_norm = features[:, 2]

            lifetime = lifetime_norm * (max_vals[0] - min_vals[0]) + min_vals[0]

            physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)
            model_out = model(physics_inputs)
            tau0 = model_out[:, 0]
            log_ksv1 = model_out[:, 1]
            log_ksv2 = model_out[:, 2]
            f1 = model_out[:, 3]
            offset = model_out[:, 4]

            pred_po2 = multi_site_quenching_predict(
                tau0, log_ksv1, log_ksv2, f1, lifetime, offset
            )

            all_true_po2.append(true_po2.item())
            all_pred_po2.append(pred_po2.item())
            all_pred_tau0.append(tau0.item())
            all_pred_ksv1.append(math.exp(log_ksv1.item()))
            all_pred_ksv2.append(math.exp(log_ksv2.item()))
            all_pred_f1.append(f1.item())
            all_measured_lifetime.append(lifetime.item())

            print(
                f"Sample {idx:03d}: True pO₂ = {true_po2.item():6.3f} cmHg | "
                f"Pred = {pred_po2.item():6.3f} cmHg | "
                f"f₁ = {f1.item():.3f} | "
                f"Ksv₁ = {math.exp(log_ksv1.item()):7.3f} | "
                f"Ksv₂ = {math.exp(log_ksv2.item()):7.3f} | "
                f"Tau0 = {tau0.item():7.3f}"
            )

    true_po2 = np.array(all_true_po2)
    pred_po2 = np.array(all_pred_po2)

    mse = np.mean((true_po2 - pred_po2) ** 2)
    r2 = 1.0 - np.sum((true_po2 - pred_po2) ** 2) / np.sum(
        (true_po2 - np.mean(true_po2)) ** 2
    )

    print("\n────────────────────────────────────────")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 : {r2:.4f}")

    # Scatter plot -----------------------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.scatter(true_po2, pred_po2, alpha=0.7)
    mn, mx = true_po2.min(), true_po2.max()
    plt.plot([mn, mx], [mn, mx], "--", color="gray")
    plt.xlabel("True pO2 (cmHg)")
    plt.ylabel("Predicted pO2 (cmHg)")
    plt.title("True vs Predicted pO2 – Multi‑site Model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (
        true_po2,
        pred_po2,
        np.array(all_pred_tau0),
        np.array(all_pred_ksv1),
        np.array(all_pred_ksv2),
        np.array(all_pred_f1),
        np.array(all_measured_lifetime),
    )


if __name__ == "__main__":
    test_multi_site()
