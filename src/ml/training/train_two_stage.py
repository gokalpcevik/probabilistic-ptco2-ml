import argparse
import os
from typing import Tuple
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_processing.o2_dataset import OxygenDataset
from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP, train_physics_only

from src.ml.training.train_common import (
    monotonicity_penalty,
    arrhenius_penalty,
    smoothness_penalty,
)


def physics_loss_fn(
    tau0: torch.Tensor,
    log_ksv: torch.Tensor,
    pred_po2: torch.Tensor,
    lifetime_obs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ksv = torch.exp(log_ksv)
    lifetime_pred_ratio = 1.0 / (1.0 + ksv * pred_po2)
    measured_tau_ratio = lifetime_obs / tau0.clamp(min=eps)
    measured_tau_ratio = measured_tau_ratio.clamp(min=eps)
    return F.mse_loss(lifetime_pred_ratio, measured_tau_ratio)


from torch.utils.data import TensorDataset, DataLoader


def _train_regressor_alignment(
    physics_ckpt: str,
    data_dir: str = "./data/processed/",
    save_dir: str = "./models/",
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    lambda_phys: float = 0.0,
    lambda_mono: float = 0.0,
    lambda_arrh: float = 0.0,
    lambda_smooth: float = 0.0,
    unfreeze_at_pct: float = 0.9,
    physics_lr_scale: float = 0.1,
    n_synth: int = 0,  # <<< how many synthetic samples to add
    min_factor: float = 1.10,  # <<< start 10 % above max real pO2
    max_factor: float = 3.5,  # <<< up to 350 % above
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Stage-2 training loop with physics-aligned synthetic augmentation."""
    os.makedirs(save_dir, exist_ok=True)

    # ── REAL DATASETS ──────────────────────────────────────────────────────
    train_dataset_real = OxygenDataset(data_dir, split="train")
    val_dataset = OxygenDataset(data_dir, split="val")

    # ── MODELS ─────────────────────────────────────────────────────────────
    reg_mlp = pO2Regressor(input_dim=3).to(device)
    phys_mlp = SternVolmerMLP(input_dim=2).to(device)
    phys_mlp.load_state_dict(torch.load(physics_ckpt, map_location=device))

    # Freeze physics model initially
    for p in phys_mlp.parameters():
        p.requires_grad = False

    optim_groups = [
        {"params": reg_mlp.parameters(), "lr": lr},
        {"params": [], "lr": lr * physics_lr_scale},  # will be filled when unfrozen
    ]
    optimizer = optim.Adam(optim_groups, weight_decay=weight_decay)

    phys_mlp.eval()  # inference only
    rng = np.random.default_rng()

    # statistics for normalising lifetime later
    life_min, life_rng = train_dataset_real.min[0], train_dataset_real.range[0]

    # draw random (T,pulses) pairs from *normalised* real features
    idx = rng.integers(0, len(train_dataset_real), size=n_synth)
    temp_norm = train_dataset_real.features[idx, 1].astype(np.float32)
    pulses_norm = train_dataset_real.features[idx, 2].astype(np.float32)

    # choose synthetic targets above real maximum pO2
    po2_max = train_dataset_real.targets.max()
    po2_synth = rng.uniform(
        po2_max * min_factor, po2_max * max_factor, size=n_synth
    ).astype(np.float32)

    # predict tau_0 & Ksv -> compute synthetic lifetime
    with torch.no_grad():
        inp = torch.tensor(
            np.stack([temp_norm, pulses_norm], axis=1),
            dtype=torch.float32,
            device=device,
        )
        tau0_logksv = phys_mlp(inp)
        tau0 = tau0_logksv[:, 0].cpu().numpy()
        ksv = np.exp(tau0_logksv[:, 1].cpu().numpy())
        offset = tau0_logksv[:, 2].cpu().numpy()

    lifetime = tau0 / (1.0 + ksv * (po2_synth - offset))
    life_norm = (lifetime - life_min) / life_rng

    X_synth = np.stack([life_norm, temp_norm, pulses_norm], axis=1).astype(np.float32)
    y_synth = po2_synth

    # ── CONCATENATE REAL + SYNTHETIC ──────────────────────────────────────
    X_aug = np.vstack([train_dataset_real.features, X_synth])
    y_aug = np.hstack([train_dataset_real.targets, y_synth])

    train_dataset = TensorDataset(
        torch.tensor(X_aug, dtype=torch.float32),
        torch.tensor(y_aug, dtype=torch.float32),
    )

    # ── DATALOADERS ───────────────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # save real normalisation stats for test scripts
    np.savez(
        f"{save_dir}/normalization_stats_homo.npz",
        min=train_dataset_real.min,
        max=train_dataset_real.max,
    )

    # cached ranges for physics loss
    min_vals, range_vals = train_dataset_real.min, train_dataset_real.range
    range_vals[range_vals == 0.0] = 1.0

    print(f"-> Stage 2 with synthetic augmentation ({n_synth} points)")
    total_params = sum(p.numel() for p in reg_mlp.parameters()) + sum(
        p.numel() for p in phys_mlp.parameters()
    )
    print(f"   Total parameters (incl. frozen): {total_params}")

    # ── TRAINING LOOP
    for epoch in range(1, epochs + 1):

        # unfreeze physics net late in training
        if epoch == int(unfreeze_at_pct * epochs):
            for p in phys_mlp.parameters():
                p.requires_grad = True
            optim_groups[1]["params"] = phys_mlp.parameters()
            print(f"[Epoch {epoch}] Unfroze physics MLP (lr ×{physics_lr_scale}).")

        reg_mlp.train()
        phys_mlp.train()
        running = {"data": 0, "phys": 0, "mono": 0, "arrh": 0, "smooth": 0}

        for feat, tgt in train_loader:
            feat, tgt = feat.to(device), tgt.to(device)
            optimizer.zero_grad()

            ln, tn, pn = feat[:, 0], feat[:, 1], feat[:, 2]
            lifetime = ln * range_vals[0] + min_vals[0]
            temp_raw = tn * range_vals[1] + min_vals[1]

            pred_po2 = reg_mlp(feat)
            tau0_logksv = phys_mlp(torch.stack([tn, pn], dim=1))
            tau0, logk = tau0_logksv[:, 0], tau0_logksv[:, 1]

            # losses ------------------------------------------------------
            data_l = F.mse_loss(pred_po2, tgt)
            phys_l = physics_loss_fn(tau0, logk, pred_po2, lifetime)

            total_l = data_l + lambda_phys * phys_l

            if lambda_mono > 0.0:
                total_l += lambda_mono * monotonicity_penalty(temp_raw, tau0, logk)
            if lambda_arrh > 0.0:
                total_l += lambda_arrh * arrhenius_penalty(temp_raw, tau0)
            if lambda_smooth > 0.0:
                total_l += lambda_smooth * smoothness_penalty(feat, pred_po2)

            total_l.backward()
            optimizer.step()

            bs = feat.size(0)
            running["data"] += data_l.item() * bs
            running["phys"] += phys_l.item() * bs

        # -------- validation data loss only -----------------------------
        reg_mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vf, vt in val_loader:
                vf, vt = vf.to(device), vt.to(device)
                val_loss += F.mse_loss(reg_mlp(vf), vt).item() * vf.size(0)
        val_loss /= len(val_dataset)

        n_train = len(train_dataset)
        print(
            f"Epoch {epoch:03d} | "
            f"data {running['data']/n_train:.5f} | "
            f"phys {running['phys']/n_train:.5f} | "
            f"val {val_loss:.5f}"
        )

    # ── SAVE CHECKPOINTS ─────────────────────────────────────────────────
    torch.save(reg_mlp.state_dict(), f"{save_dir}/regressor_stage2.pth")
    torch.save(phys_mlp.state_dict(), f"{save_dir}/physics_stage2.pth")
    return reg_mlp, phys_mlp

def main():
    parser = argparse.ArgumentParser("Two‑stage physics‑aligned training")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Run only Stage 1, Stage 2, or both (default)",
    )

    # Stage 1 hyper‑parameters (we simply expose epochs & priors; rest use defaults from original script)
    parser.add_argument("--stage1_epochs", type=int, default=200)
    parser.add_argument("--stage1_use_mono", action="store_true")
    parser.add_argument("--stage1_use_arrh", action="store_true")

    # Stage 2 hyper‑parameters
    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument("--lambda_phys", type=float, default=50.0)
    parser.add_argument("--unfreeze_pct", type=float, default=0.9)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Paths
    data_dir = "./data/processed/"
    save_dir = "./models/"
    physics_ckpt = f"{save_dir}/physics_only_mlp.pth"

    # Stage 1
    if args.stage in ("1", "both"):
        train_physics_only(
            data_dir=data_dir,
            save_dir=save_dir,
            epochs=args.stage1_epochs,
            use_monotonicity_prior=args.stage1_use_mono,
            use_arrhenius_prior=args.stage1_use_arrh,
        )

    if args.stage == "1":
        return

    if not os.path.isfile(physics_ckpt):
        raise FileNotFoundError(
            "Physics checkpoint not found. Run Stage 1 first or specify a valid --stage=both."
        )

    # Stage 2 
    _train_regressor_alignment(
        physics_ckpt=physics_ckpt,
        data_dir=data_dir,
        save_dir=save_dir,
        epochs=args.stage2_epochs,
        lambda_phys=args.lambda_phys,
        unfreeze_at_pct=args.unfreeze_pct,
        device=args.device,
    )


if __name__ == "__main__":
    main()
