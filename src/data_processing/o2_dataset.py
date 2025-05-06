# o2_dataset.py

import numpy as np
import json
import os
import torch
from scipy.optimize import curve_fit
from torch.utils.data import Dataset
from torch import tensor

# --- Utility Functions ---
def mono_exponential(t, A, B, tau):
    return A + B * np.exp(-t / tau)

def fit_lifetime(decay_curve, time_step_us = 2.0):
    t = np.arange(len(decay_curve)) * time_step_us
    initial_guess = [np.min(decay_curve), np.max(decay_curve) - np.min(decay_curve), 20]
    try:
        popt, pcov = curve_fit(
            mono_exponential, t, decay_curve, p0=initial_guess, maxfev=5000
        )
        lifetime = popt[2]  # tau in microseconds
        residuals = decay_curve - mono_exponential(t, *popt)
        residual_var = np.var(residuals)
        msse = np.sum(residuals**2)
        r_squared = 1 - (msse / np.sum((decay_curve - np.mean(decay_curve)) ** 2))
    except Exception:
        lifetime, residual_var, msse, r_squared = 0.0, 1.0, 1.0, 0.0
    return lifetime, residual_var, msse, r_squared


class OxygenDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        normalize=True,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=45,
        jitter=False,           
        jitter_params=None,    
    ):
        self.samples = []
        self.jitter = jitter
        self.jitter_params = jitter_params or {
            "lifetime_std": 0.01,   
            "temperature_std": 0.02, 
            "pulses_std": 0.00       
        }

        for entry in os.listdir(data_dir):
            if entry.endswith(".json"):
                file_path = data_dir + entry
                with open(file_path, "r") as f:
                    data = json.load(f)
                    oxygen_pressure = data["po2"]
                    measurements = data["measurements"]
                    for m in measurements:
                        lifetime, cov_lifetime, msse, r2 = fit_lifetime(
                            m["decay_curve"]
                        )
                        features = [
                            lifetime,
                            m["temperature"],
                            np.log1p(m["pulses_applied"]),
                        ]
                        # print(f"Lifetime:{lifetime} - Temperature: {m["temperature"]}, Pulses:{np.log1p(m["pulses_applied"])}")
                        target = oxygen_pressure
                        self.samples.append((features, target))
                        
        features = np.array([s[0] for s in self.samples], dtype=np.float32)
        targets = np.array([s[1] for s in self.samples], dtype=np.float32)

        from sklearn.model_selection import train_test_split 
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, targets, test_size=test_ratio, random_state=random_seed
        )
        val_size_adjusted = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_seed
        )

        if split == "train":
            self.features = X_train
            self.targets = y_train
        elif split == "val":
            self.features = X_val
            self.targets = y_val
        elif split == "test":
            self.features = X_test
            self.targets = y_test
        else:
            raise ValueError(f"Invalid split: {split}")

        if normalize:
            self.min = X_train.min(axis=0)
            self.max = X_train.max(axis=0)
            self.range = self.max - self.min
            self.range[self.range == 0] = 1.0 
            self.features = (self.features - self.min) / self.range
        else:
            self.min = np.zeros(features.shape[1], dtype=np.float32)
            self.max = np.ones(features.shape[1], dtype=np.float32)
            self.range = np.ones(features.shape[1], dtype=np.float32)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = self.features[idx].copy() 
        target = self.targets[idx]

        if self.jitter:
            feature = self.apply_jitter(feature)

        return tensor(feature, dtype=torch.float32), tensor(target, dtype=torch.float32)

    def apply_jitter(self, feature):
        # Lifetime jitter (index 0)
        lifetime_jitter = np.random.normal(1.0, self.jitter_params["lifetime_std"])
        feature[0] *= lifetime_jitter

        # Temperature jitter (index 1)
        temp_range = self.max[1] - self.min[1]
        temperature_jitter = np.random.normal(0.0, self.jitter_params["temperature_std"] / temp_range)
        feature[1] += temperature_jitter

        # Pulses jitter (index 2)
        pulses_jitter = np.random.normal(1.0, self.jitter_params["pulses_std"])
        feature[2] *= pulses_jitter

        return feature
