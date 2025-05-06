import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ml.pO2regressor import pO2Regressor
from src.ml.training.train_sternvolmer import SternVolmerMLP, stern_volmer_predict
from src.ml.training.train_msv import ModifiedSternVolmerMLP, modified_stern_volmer_predict
from src.ml.training.train_multisite import MultiSiteQuenchingMLP, multi_site_quenching_predict

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
    "legend.loc": "upper right",
})


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

_STATS_CANDIDATES = [
    "models/normalization_stats_two_stage.npz",
    "models/normalization_stats_homo.npz",
    "models/normalization_stats_physics.npz",
    "models/normalization_stats_modified_sv.npz",
    "models/normalization_stats_multi_site.npz",
]

def _load_stats():
    for path in _STATS_CANDIDATES:
        if os.path.exists(path):
            stats = np.load(path)
            return stats["min"], stats["max"], path
    raise FileNotFoundError("Could not locate any normalization stats .npz file under models/.")

def _safe_load(model_cls, weight_path, **kwargs):
    if not os.path.exists(weight_path):
        print(f"[WARN] {weight_path} not found – skipping …")
        return None
    model = model_cls(**kwargs)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_models():
    models = {}

    physics_only = _safe_load(SternVolmerMLP, "models/physics_only_mlp.pth", input_dim=2)
    if physics_only is not None:
        models["Stern-Volmer"] = (None, physics_only)

    reg_joint = _safe_load(pO2Regressor, "models/regression_mlp.pth", input_dim=3)
    phys_joint = _safe_load(SternVolmerMLP, "models/physics_mlp.pth", input_dim=2)
    if reg_joint is not None and phys_joint is not None:
        models["Joint"] = (reg_joint, phys_joint)

    reg_two = _safe_load(pO2Regressor, "models/regressor_stage2.pth", input_dim=3)
    phys_two = _safe_load(SternVolmerMLP, "models/physics_stage2.pth", input_dim=2)
    if reg_two is not None and phys_two is not None:
        models["Two-Stage"] = (reg_two, phys_two)

    modified_sv = _safe_load(ModifiedSternVolmerMLP, "models/modified_sv_mlp.pth", input_dim=2)
    if modified_sv is not None:
        models["Modified-SV"] = (None, modified_sv)

    multi_site = _safe_load(MultiSiteQuenchingMLP, "models/multi_site_quenching_mlp.pth", input_dim=2)
    if multi_site is not None:
        models["Multi-Site"] = (None, multi_site)

    if not models:
        raise RuntimeError("No models could be loaded – please train at least one variant first.")
    return models

# -----------------------------------------------------------------------------
# Stern-Volmer sweep
# -----------------------------------------------------------------------------

def stern_volmer_sweep():
    min_vals, max_vals, stats_file = _load_stats()
    models = load_models()

    print(f"Loaded normalization stats from: {stats_file}")
    print(f"Models found: {', '.join(models.keys())}")

    TEMP_C_LIST = [30.0]
    PULSES_LOG = np.log1p(1e1)

    lifetime_values = np.linspace(7.0, 30.0, 300)

    curves = {}

    for TEMP_C in TEMP_C_LIST:
        temp_norm = (TEMP_C - min_vals[1]) / (max_vals[1] - min_vals[1])
        pulses_norm = (PULSES_LOG - min_vals[2]) / (max_vals[2] - min_vals[2])

        physics_const_in = torch.tensor([[temp_norm, pulses_norm]], dtype=torch.float32)

        for name, (regressor, physics) in models.items():
            x_vals, y_vals = [], []
            for lifetime in lifetime_values:
                lifetime_norm = (lifetime - min_vals[0]) / (max_vals[0] - min_vals[0])
                lifetime_t = torch.tensor([lifetime], dtype=torch.float32)
                reg_in = torch.tensor([[lifetime_norm, temp_norm, pulses_norm]], dtype=torch.float32)

                with torch.no_grad():
                    model_out = physics(physics_const_in)
                    tau0 = model_out[:, 0]

                    if name == "Modified-SV":
                        log_ksv = model_out[:, 1]
                        log_ks = model_out[:, 2]
                        offset = model_out[:, 3]
                        pO2_pred = modified_stern_volmer_predict(tau0, log_ksv, log_ks, lifetime_t, offset).item()
                    elif name == "Multi-Site":
                        log_ksv1 = model_out[:, 1]
                        log_ksv2 = model_out[:, 2]
                        f1 = model_out[:, 3]
                        offset = model_out[:, 4]
                        pO2_pred = multi_site_quenching_predict(tau0, log_ksv1, log_ksv2, f1, lifetime_t, offset).item()
                    elif regressor is None:
                        log_ksv = model_out[:, 1]
                        offset = model_out[:, 2]
                        pO2_pred = stern_volmer_predict(tau0, log_ksv, lifetime_t, offset).item()
                    else:
                        pO2_pred = regressor(reg_in).item()

                    x_vals.append(pO2_pred)
                    y_vals.append(tau0.item() / lifetime - 1.0)

            curves[f"{name} ({TEMP_C:.0f}°C)"] = (x_vals, y_vals)

    plt.figure(figsize=(10, 7))

    model_styles = {
        "Stern-Volmer": "-",
        "Joint": "--",
        "Two-Stage": ":",
        "Modified-SV": "-.",
        "Multi-Site": "--",  
    }

    model_colors = {
        "Stern-Volmer": "#1f77b4",  # blue
        "Joint": "#ff7f0e",         # orange
        "Two-Stage": "#2ca02c",     # green
        "Modified-SV": "#d62728",   # red
        "Multi-Site": "#9467bd",    # purple
    }

    for label, (x, y) in curves.items():
        model_name = label.split(" (")[0]
        style = model_styles.get(model_name, "-")
        color = model_colors.get(model_name, "black")
        plt.plot(x, y, linestyle=style, color=color, label=label)

    plt.xlabel("Predicted pO₂ (cmHg)", labelpad=12)
    plt.ylabel(r"$\frac{\tau_0}{\tau} - 1$", labelpad=12)
    plt.title("Stern–Volmer Plot Comparison Across Models", pad=16)
    plt.legend(ncol=1, fancybox=True)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlim(left=0)  
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    stern_volmer_sweep()