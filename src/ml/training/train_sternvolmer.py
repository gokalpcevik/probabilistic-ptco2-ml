import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
from torch.utils.data import DataLoader
import numpy as np
from src.data_processing.o2_dataset import OxygenDataset
import os
from src.ml.training.train_common import (
    monotonicity_penalty,
    arrhenius_penalty,
)

class SternVolmerMLP(nn.Module):
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
            nn.Linear(16, 3),  # tau0, log_ksv, offset
        )
        
        self.min_tau0 = 25.0
        self.max_tau0 = 30.0
        
        self.min_logksv = np.log(0.01)
        self.max_logksv = np.log(100.0)

    def forward(self, x):
        out = self.net(x)
        tau0_unbounded = out[:, 0]
        logksv_unbounded = out[:, 1]

        tau0 = self.min_tau0 + (self.max_tau0 - self.min_tau0) * torch.sigmoid(tau0_unbounded)
        logksv = self.min_logksv + (self.max_logksv - self.min_logksv) * torch.sigmoid(logksv_unbounded)

        return torch.stack([tau0, logksv, out[:, 2]], dim=1)
    

class QuantizableSternVolmerMLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # tau0, log_ksv, offset
        )
        
        self.min_tau0 = 25.0
        self.max_tau0 = 30.0
        
        self.min_logksv = np.log(0.01)
        self.max_logksv = np.log(100.0)

    def forward(self, x):
        x = self.quant(x)
        out = self.net(x)
        out = self.dequant(out)
        tau0_unbounded = out[:, 0]
        logksv_unbounded = out[:, 1]

        tau0 = self.min_tau0 + (self.max_tau0 - self.min_tau0) * torch.sigmoid(tau0_unbounded)
        logksv = self.min_logksv + (self.max_logksv - self.min_logksv) * torch.sigmoid(logksv_unbounded)

        return torch.stack([tau0, logksv, out[:, 2]], dim=1)
    

def stern_volmer_predict(tau0, log_ksv, lifetime, offset, eps=1e-6):
    ksv = torch.exp(log_ksv)
    lifetime = lifetime.clamp(min=0.1)  
    ksv = ksv.clamp(min=1e-2)            
    po2 = (tau0 / lifetime - 1.0) / ksv + offset
    return po2

# --- Main Training Function ---
def train_physics_only(
    data_dir="./data/processed/",
    save_dir="./models/",
    epochs=200,
    batch_size=256,
    lr=1e-4,
    weight_decay=0.01,
    lambda_mono=0.5,
    lambda_arrh=0.02,
    use_monotonicity_prior=True,
    use_arrhenius_prior=True,
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = OxygenDataset(data_dir, split="train")
    val_dataset = OxygenDataset(data_dir, split="val")

    np.savez(f"{save_dir}/normalization_stats_physics.npz", min=train_dataset.min, max=train_dataset.max)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SternVolmerMLP(input_dim=2)
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
            offset = model_out[:, 2]

            pred_po2 = stern_volmer_predict(tau0, log_ksv, lifetime, offset)

            data_loss = F.mse_loss(pred_po2, targets)
            total_loss = data_loss

            if use_monotonicity_prior:
                temp_raw = temperature_norm * range_vals[1] + min_vals[1]  # De-normalize temperature for physical prior
                total_loss += lambda_mono * monotonicity_penalty(temp_raw, tau0, log_ksv)

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
                offset = model_out[:, 2]

                pred_po2 = stern_volmer_predict(tau0, log_ksv, lifetime, offset)

                val_loss = F.mse_loss(pred_po2, targets)
                total_val_loss += val_loss.item() * features.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)

        print(f"{epoch+1:6d} | {avg_train_loss:12.5f} | {avg_val_loss:10.5f}")

    torch.save(model.state_dict(), f"{save_dir}/physics_only_mlp.pth")

    return model

if __name__ == "__main__":
    best_model = train_physics_only(
    data_dir="./data/processed/",
    save_dir="./models/",
    epochs=400,
    lr=1e-3,
    lambda_mono=0.5,
    lambda_arrh=0.5,
    use_monotonicity_prior=False,
    use_arrhenius_prior=False,
)
