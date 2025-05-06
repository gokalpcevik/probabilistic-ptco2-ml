# train_common.py

import torch
import torch.nn.functional as F
import numpy as np

def monotonicity_penalty(temperature, *outputs, eps=1e-6):
    """
    Generalized monotonicity penalty for one or more outputs.
    Each output is expected to be a tensor aligned with the temperature tensor.
    """
    sorted_temp, indices = torch.sort(temperature)
    penalties = []

    for out in outputs:
        sorted_out = out[indices]
        delta_temp = sorted_temp[1:] - sorted_temp[:-1]
        delta_temp = torch.where(delta_temp.abs() < eps, torch.full_like(delta_temp, eps), delta_temp)

        grad_out = (sorted_out[1:] - sorted_out[:-1]) / delta_temp
        penalty = torch.mean(F.relu(grad_out))
        penalties.append(penalty)

    return sum(penalties)

def arrhenius_penalty(temperature_C, tau0, eps=1e-6):
    """
    Arrhenius penalty enforcing linear relationship between log(tau0) and 1/T.
    """
    temperature_K = temperature_C + 273.15
    inv_temp = 1.0 / (temperature_K + eps)
    log_tau0 = torch.log(tau0.clamp(min=eps))

    A = inv_temp.unsqueeze(1)
    B = log_tau0.unsqueeze(1)

    solution = torch.linalg.lstsq(A, B).solution.squeeze()
    slope = solution

    log_tau0_pred = slope * inv_temp
    penalty = F.mse_loss(log_tau0_pred, log_tau0)
    return penalty

def physics_loss_fn(tau0, log_ksv, pred_po2, lifetime_obs, eps=1e-6, scaled=False):
    """
    Basic or scaled physics loss comparing predicted and observed lifetime ratios.
    If scaled=True, applies scaling by (1+Ksv*pO2)^2.
    """
    ksv = torch.exp(log_ksv)
    lifetime_pred_ratio = 1.0 / (1.0 + ksv * pred_po2)
    measured_tau_ratio  = lifetime_obs / tau0.clamp(min=eps)

    if scaled:
        resid  = lifetime_pred_ratio - measured_tau_ratio
        weight = (1.0 + ksv * pred_po2).detach() ** 2
        return torch.mean(weight * resid**2)
    else:
        measured_tau_ratio = measured_tau_ratio.clamp(min=eps)
        return F.mse_loss(lifetime_pred_ratio, measured_tau_ratio)

def smoothness_penalty(features, pred_po2):
    """
    Smoothness penalty on lifetime -> pO2 mapping.
    """
    lifetime = features[:, 0].detach().requires_grad_(True)

    try:
        grads = torch.autograd.grad(
            outputs=pred_po2,
            inputs=lifetime,
            grad_outputs=torch.ones_like(pred_po2),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        penalty = torch.mean(grads**2)
    except Exception:
        penalty = torch.tensor(0.0, device=features.device)

    return penalty

def load_data_normalization(dataset):
    """
    Given an OxygenDataset, returns (min_vals, max_vals, range_vals) with divide-by-zero protection.
    """
    min_vals = dataset.min
    max_vals = dataset.max
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0.0] = 1.0  # prevent divide-by-zero
    return min_vals, max_vals, range_vals

def print_training_header():
    """
    Prints standard header for training loops.
    """
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10}")
    print("-" * 50)