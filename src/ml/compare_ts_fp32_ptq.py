import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.ml.pO2regressor import QuantizablepO2Regressor
from src.ml.training.train_sternvolmer import QuantizableSternVolmerMLP
from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP
from src.data_processing.o2_dataset import OxygenDataset
from torch.utils.data import DataLoader

plt.style.use("seaborn-v0_8-muted")
plt.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "lines.linewidth": 2.5,
    "legend.frameon": False,
    "legend.loc": "lower right",
})

stats = np.load("models/normalization_stats_homo.npz")
min_vals = stats["min"]
max_vals = stats["max"]
ranges = max_vals - min_vals
ranges[ranges == 0.0] = 1.0

test_dataset = OxygenDataset("data/processed/", split="test", normalize=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

reg_fp32 = pO2Regressor(input_dim=3)
reg_fp32.load_state_dict(torch.load("models/regressor_stage2.pth"))
reg_fp32.eval()

reg_ptq = QuantizablepO2Regressor(input_dim=3)
reg_ptq = torch.load("models/quantized_regressor_stage2.pth", weights_only=False)
reg_ptq.eval()

true_po2, pred_fp32, pred_ptq = [], [], []

with torch.no_grad():
    for features, target in test_loader:
        po2_fp32 = reg_fp32(features)
        po2_ptq = reg_ptq(features)
        true_po2.append(target.item())
        pred_fp32.append(po2_fp32.item())
        pred_ptq.append(po2_ptq.item())

true_po2 = np.array(true_po2)
pred_fp32 = np.array(pred_fp32)
pred_ptq = np.array(pred_ptq)

def compare_models(true, pred_fp32, pred_ptq):
    mse_fp32 = mean_squared_error(true, pred_fp32)
    r2_fp32 = r2_score(true, pred_fp32)
    mae_fp32 = mean_absolute_error(true, pred_fp32)

    mse_ptq = mean_squared_error(true, pred_ptq)
    r2_ptq = r2_score(true, pred_ptq)
    mae_ptq = mean_absolute_error(true, pred_ptq)

    delta = np.abs(pred_fp32 - pred_ptq)
    within_1 = np.mean(delta < 1.0) * 100
    within_0_1 = np.mean(delta < 0.1) * 100

    stats_text = (
        f"FP32:\n"
        f"  R²   = {r2_fp32:.4f}\n"
        f"  MAE  = {mae_fp32:.3f}\n"
        f"  MSE  = {mse_fp32:.3f}\n\n"
        f"PTQ:\n"
        f"  R²   = {r2_ptq:.4f}\n"
        f"  MAE  = {mae_ptq:.3f}\n"
        f"  MSE  = {mse_ptq:.3f}\n\n"
        f"Agreement:\n"
        f"  <1.0 cmHg  = {within_1:.1f}%\n"
        f"  <0.1 cmHg = {within_0_1:.1f}%"
    )

    return stats_text

text_box = compare_models(true_po2, pred_fp32, pred_ptq)

# Plot comparison
plt.figure(figsize=(6.5, 6.5))
plt.scatter(pred_fp32, pred_ptq, alpha=0.7, s=30, edgecolors='k', label="Predictions")
plt.plot([min(pred_fp32), max(pred_fp32)], [min(pred_fp32), max(pred_fp32)], '--', color='gray', label="Ideal (y = x)")

plt.xlabel("FP32 pO2 [cmHg]")
plt.ylabel("Quantized pO2 [cmHg]")
plt.title("FP32 vs PTQ Predictions")

plt.text(0.05, 0.95, text_box, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='gray', alpha=0.9))

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
