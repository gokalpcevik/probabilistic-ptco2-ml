import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.data_processing.o2_dataset import OxygenDataset
from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP
from src.ml.training.train_common import (
    monotonicity_penalty,
    arrhenius_penalty,
    physics_loss_fn,
    smoothness_penalty,
)

# --- Losses ---
def physics_loss_fn(tau0, log_ksv, pred_po2, lifetime_obs, eps=1e-6):
    ksv = torch.exp(log_ksv)
    # lifetime_pred_ratio = 1.0 / (1.0 + ksv * pred_po2)
    # measured_tau_ratio  = lifetime_obs / tau0.clamp(min=eps)
    # return F.mse_loss(lifetime_pred_ratio, measured_tau_ratio)
    lifetime_pred_ratio = 1.0 / (1.0 + ksv * pred_po2)
    measured_tau_ratio  = lifetime_obs / tau0.clamp(min=eps)

    resid  = lifetime_pred_ratio - measured_tau_ratio
    weight = (1.0 + ksv * pred_po2).detach() ** 2
    return torch.mean(weight * resid**2)

# --- Training Loop ---
def train_joint(
    data_dir="./data/processed/",
    save_dir="./models/",
    epochs=200,
    batch_size=256,
    lr=4e-4,
    weight_decay=0.01,
    lambda_phys=0.0,
    lambda_mono=0.15,
    lambda_arrh=0.05,
    lambda_smooth=0.1,
    use_monotonicity_prior=False,
    use_arrhenius_prior=False,
    use_smoothness_regularization=False,
):
    # --- Data ---
    train_dataset = OxygenDataset(data_dir, split="train")
    val_dataset = OxygenDataset(data_dir, split="val")

    np.savez(
        f"{save_dir}/normalization_stats_homo.npz",
        min=train_dataset.min,
        max=train_dataset.max,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Models ---
    regression_mlp = pO2Regressor(input_dim=3)
    physics_mlp = SternVolmerMLP(input_dim=2)

    all_params = list(regression_mlp.parameters()) + list(physics_mlp.parameters())
    optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    # --- Load normalization stats
    min_vals = train_dataset.min
    max_vals = train_dataset.max
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0.0] = 1.0  # Prevent divide-by-zero

    print(f"Total parameters: {sum(p.numel() for p in all_params)}")
    print(f"{'Epoch':>6} | {'Data Loss':>12} | {'Val Loss':>10}")
    print("-" * 50)

    for epoch in range(epochs):
        regression_mlp.train()
        physics_mlp.train()

        total_data_loss = 0.0
        total_val_loss = 0.0
        total_phys_loss = 0.0
        total_mono_penalty = 0.0
        total_arrh_penalty = 0.0
        total_smooth_penalty = 0.0

        for features, targets in train_loader:
            optimizer.zero_grad()

            lifetime_norm = features[:, 0]
            temperature_norm = features[:, 1]
            pulses_norm = features[:, 2]

            # --- De-normalize lifetime and temperature
            lifetime = lifetime_norm * range_vals[0] + min_vals[0]
            temperature = temperature_norm * range_vals[1] + min_vals[1]

            pred_po2 = regression_mlp(features)
            physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)
            tau0_logksv = physics_mlp(physics_inputs)
            tau0 = tau0_logksv[:, 0]
            log_ksv = tau0_logksv[:, 1]

            # --- Compute loss components
            data_loss = F.mse_loss(pred_po2, targets)
            phys_loss = physics_loss_fn(tau0, log_ksv, pred_po2, lifetime)

            total_loss = data_loss + lambda_phys * phys_loss

            if use_monotonicity_prior:
                total_loss += lambda_mono * monotonicity_penalty(temperature, tau0, log_ksv)

            if use_arrhenius_prior:
                total_loss += lambda_arrh * arrhenius_penalty(temperature, tau0)

            if use_smoothness_regularization:
                total_loss += lambda_smooth * smoothness_penalty(features, pred_po2)

            total_loss.backward()
            optimizer.step()

            # --- Accumulate weighted averages
            batch_size_now = features.size(0)
            total_data_loss += data_loss.item() * batch_size_now
            total_phys_loss += phys_loss.item() * batch_size_now

        # --- Validation
        regression_mlp.eval()
        physics_mlp.eval()
        with torch.no_grad():
            for features, targets in val_loader:
                pred_po2 = regression_mlp(features)
                val_loss = F.mse_loss(pred_po2, targets)
                total_val_loss += val_loss.item() * features.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_data_loss = total_data_loss / len(train_dataset)
        avg_phys_loss = total_phys_loss / len(train_dataset)
        print(
            f"Epoch {epoch+1:03d} | "
            f"Data Loss: {avg_data_loss:.5f} | "
            f"Physics Loss: {avg_phys_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f}"
        )

    torch.save(regression_mlp.state_dict(), f"{save_dir}/regression_mlp.pth")
    torch.save(physics_mlp.state_dict(), f"{save_dir}/physics_mlp.pth")
    return regression_mlp, physics_mlp

if __name__ == "__main__":
    train_joint()
