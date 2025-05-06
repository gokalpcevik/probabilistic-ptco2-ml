import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP, stern_volmer_predict
from src.data_processing.o2_dataset import OxygenDataset

plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({
    "font.size": 13,
    "font.family": "sans-serif",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "figure.dpi": 120,
    "legend.frameon": False
})

def load_models_and_data():
    stats = np.load("models/normalization_stats_homo.npz")
    min_vals = stats["min"]
    max_vals = stats["max"]
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # avoid divide-by-zero

    test_dataset = OxygenDataset(
        "data/processed/", split="test", normalize=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    regression_model = pO2Regressor(input_dim=3)
    regression_model.load_state_dict(torch.load("models/regression_mlp.pth"))
    regression_model.eval()

    physics_model = SternVolmerMLP(input_dim=2)
    physics_model.load_state_dict(torch.load("models/physics_mlp.pth"))
    physics_model.eval()

    return regression_model, physics_model, test_loader, min_vals, max_vals, ranges

def test_joint():
    regression_model, physics_model, test_loader, min_vals, max_vals, ranges = load_models_and_data()

    all_true_po2 = []
    all_pred_po2 = []
    all_pred_tau0 = []
    all_pred_ksv = []
    all_measured_lifetime = []

    with torch.no_grad():
        for features, true_po2 in test_loader:
            lifetime_norm = features[:, 0]
            temperature_norm = features[:, 1]
            pulses_norm = features[:, 2]

            lifetime = lifetime_norm * (max_vals[0] - min_vals[0]) + min_vals[0]

            pred_po2 = regression_model(features)

            physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)
            physics_out = physics_model(physics_inputs)
            tau0 = physics_out[:, 0]
            log_ksv = physics_out[:, 1]

            all_true_po2.append(true_po2.item())
            all_pred_po2.append(pred_po2.item())
            all_pred_tau0.append(tau0.item())
            all_pred_ksv.append(torch.exp(log_ksv).item())
            all_measured_lifetime.append(lifetime.item())

    all_true_po2 = np.array(all_true_po2) 
    all_pred_po2 = np.array(all_pred_po2) 
    all_pred_tau0 = np.array(all_pred_tau0)
    all_pred_ksv = np.array(all_pred_ksv)
    all_measured_lifetime = np.array(all_measured_lifetime)

    mse = np.mean((all_true_po2 - all_pred_po2) ** 2)
    r2 = 1.0 - np.sum((all_true_po2 - all_pred_po2) ** 2) / np.sum((all_true_po2 - np.mean(all_true_po2)) ** 2)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R^2:  {r2:.4f}")

    plt.figure(figsize=(7,7))
    plt.scatter(all_true_po2, all_pred_po2, alpha=0.7)
    plt.plot([0, max(all_true_po2)], [0, max(all_true_po2)], '--', color='gray')
    plt.xlabel("True pO2 (cmHg)")
    plt.ylabel("Predicted pO2 (cmHg)")
    plt.title("True vs Predicted pO2 (Joint Network)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_true_po2, all_pred_po2, all_pred_tau0, all_pred_ksv, all_measured_lifetime

if __name__ == "__main__":
    test_joint()