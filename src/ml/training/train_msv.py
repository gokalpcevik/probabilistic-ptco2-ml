import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from src.data_processing.o2_dataset import OxygenDataset
from src.ml.training.train_common import (
    monotonicity_penalty,
    arrhenius_penalty
)

class ModifiedSternVolmerMLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # tau0, log_ksv, log_ks, offset
        )
        
        self.min_tau0 = 25.0
        self.max_tau0 = 30.0
        self.min_logksv = np.log(0.01)
        self.max_logksv = np.log(100.0)
        self.min_logks = np.log(0.01)
        self.max_logks = np.log(100.0)

    def forward(self, x):
        out = self.net(x)
        tau0_unbounded = out[:, 0]
        logksv_unbounded = out[:, 1]
        logks_unbounded = out[:, 2]
        
        tau0 = self.min_tau0 + (self.max_tau0 - self.min_tau0) * torch.sigmoid(tau0_unbounded)
        logksv = self.min_logksv + (self.max_logksv - self.min_logksv) * torch.sigmoid(logksv_unbounded)
        logks = self.min_logks + (self.max_logks - self.min_logks) * torch.sigmoid(logks_unbounded)
        
        return torch.stack([tau0, logksv, logks, out[:, 3]], dim=1)

def modified_stern_volmer_predict(tau0, log_ksv, log_ks, lifetime, offset, eps=1e-6, max_iter=10):
    ksv = torch.exp(log_ksv).clamp(min=1e-2)
    ks = torch.exp(log_ks).clamp(min=1e-2)
    lifetime = lifetime.clamp(min=0.1)
    
    ratio = tau0 / lifetime
    
    # Initialize pO2 guess (use the approximation as starting point)
    po2 = (ratio - 1.0) / (ksv + ks + eps)
    
    # Newton-Raphson solver
    for _ in range(max_iter):
        # Compute f(pO2) = (1 + K_SV * pO2)(1 + K_S * pO2) - tau0/tau
        f = (1 + ksv * po2) * (1 + ks * po2) - ratio
        # Compute derivative df/dpO2 = K_SV * (1 + K_S * pO2) + K_S * (1 + K_SV * pO2)
        df = ksv * (1 + ks * po2) + ks * (1 + ksv * po2)
        # Update pO2: pO2 = pO2 - f / df
        po2 = po2 - f / (df + eps)
        po2 = torch.clamp(po2, min=0.0)
    
    po2 = po2 + offset
    
    return po2

# --- Main Training Function ---
def train_modified_sv(
    data_dir="./data/processed/",
    save_dir="./models/",
    epochs=200,
    batch_size=256,
    lr=1e-4,
    weight_decay=0.01,
    lambda_mono=0.1,
    lambda_arrh=0.01,
    use_monotonicity_prior=True,
    use_arrhenius_prior=True,
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = OxygenDataset(data_dir, split="train")
    val_dataset = OxygenDataset(data_dir, split="val")

    np.savez(f"{save_dir}/normalization_stats_modified_sv.npz", min=train_dataset.min, max=train_dataset.max)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ModifiedSternVolmerMLP(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Load normalization stats
    min_vals = train_dataset.min
    max_vals = train_dataset.max
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0.0] = 1.0  # Prevent divide-by-zero

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10}")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_data_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()

            lifetime_norm = features[:, 0]
            temperature_norm = features[:, 1]
            pulses_norm = features[:, 2]

            # --- De-normalize lifetime ---
            lifetime = lifetime_norm * range_vals[0] + min_vals[0]

            physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)
            model_out = model(physics_inputs)
            tau0 = model_out[:, 0]
            log_ksv = model_out[:, 1]
            log_ks = model_out[:, 2]
            offset = model_out[:, 3]

            pred_po2 = modified_stern_volmer_predict(tau0, log_ksv, log_ks, lifetime, offset)

            data_loss = F.mse_loss(pred_po2, targets)
            total_loss = data_loss

            if use_monotonicity_prior:
                temp_raw = temperature_norm * range_vals[1] + min_vals[1]  # De-normalize temperature
                total_loss += lambda_mono * monotonicity_penalty(temp_raw, tau0, log_ksv, log_ks)

            if use_arrhenius_prior:
                temp_raw = temperature_norm * range_vals[1] + min_vals[1]
                total_loss += lambda_arrh * arrhenius_penalty(temp_raw, tau0)

            total_loss.backward()
            optimizer.step()

            total_data_loss += data_loss.item() * features.size(0)

        avg_train_loss = total_data_loss / len(train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                lifetime_norm = features[:, 0]
                temperature_norm = features[:, 1]
                pulses_norm = features[:, 2]

                lifetime = lifetime_norm * range_vals[0] + min_vals[0]

                physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)
                model_out = model(physics_inputs)
                tau0 = model_out[:, 0]
                log_ksv = model_out[:, 1]
                log_ks = model_out[:, 2]
                offset = model_out[:, 3]

                pred_po2 = modified_stern_volmer_predict(tau0, log_ksv, log_ks, lifetime, offset)

                val_loss = F.mse_loss(pred_po2, targets)
                total_val_loss += val_loss.item() * features.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)

        print(f"{epoch+1:6d} | {avg_train_loss:12.5f} | {avg_val_loss:10.5f}")

    torch.save(model.state_dict(), f"{save_dir}/modified_sv_mlp.pth")

    return model

if __name__ == "__main__":
    best_model = train_modified_sv(
        data_dir="./data/processed/",
        save_dir="./models/",
        epochs=400,
        lr=4e-4,
        lambda_mono=0.1,
        lambda_arrh=0.02,
        use_monotonicity_prior=False,
        use_arrhenius_prior=False,
    )