import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

DATA_DIR = 'data'
TIMESTEP_US = 2
NUM_SAMPLES = 60

def load_decay_curves_and_temps(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    decay_curves = [m['decay_curve'] for m in data]
    temperatures = [m['temperature'] for m in data]
    return np.array(decay_curves), np.array(temperatures)

def mono_exponential(t, A, B, tau):
    return A + B * np.exp(t / -tau)

def main():
    files = sorted(os.listdir(DATA_DIR), key=lambda x: int(os.path.splitext(x)[0]))
    time_axis = np.arange(0, NUM_SAMPLES * TIMESTEP_US, TIMESTEP_US)

    oxygen_percentages = []
    lifetimes = []
    average_temperatures = []

    fig_decay, ax_decay = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))  # smooth colormap for different oxygen levels
    
    for idx, filename in enumerate(files):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(DATA_DIR, filename)
        decay_curves, temperatures = load_decay_curves_and_temps(filepath)

        mean_decay_curve = np.mean(decay_curves, axis=0)
        mean_temperature = np.mean(temperatures)

        initial_guess = [np.min(mean_decay_curve), np.max(mean_decay_curve) - np.min(mean_decay_curve), 20]

        try:
            popt, _ = curve_fit(mono_exponential, time_axis, mean_decay_curve, p0=initial_guess, maxfev=5000)
            A_fit, B_fit, tau_fit = popt
        except RuntimeError:
            print(f"Fit failed for {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        oxygen_percentage = float(base_name) / 100.0

        oxygen_percentages.append(oxygen_percentage)
        lifetimes.append(tau_fit)
        average_temperatures.append(mean_temperature)

        color = colors[idx]
        ax_decay.plot(time_axis, mean_decay_curve, 'o', markersize=4, color=color, label=f"{oxygen_percentage:.2f}% O₂ - Raw")
        ax_decay.plot(time_axis, mono_exponential(time_axis, *popt), '-', linewidth=2, color=color, label=f"{oxygen_percentage:.2f}% O₂ - Fit")

    ax_decay.set_xlabel('Time (µs)')
    ax_decay.set_ylabel('Luminescence Intensity (a.u.)')
    ax_decay.set_title('Luminescence Decay Curves and Fits')
    ax_decay.legend(fontsize=8, loc='best', ncol=2)
    ax_decay.grid(True, linestyle='--', alpha=0.5)
    fig_decay.tight_layout()

    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Temperature (°C)', color=color2)
    ax2.plot(oxygen_percentages, average_temperatures, 's--', color=color2, markersize=6, linewidth=2, label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle('Luminescence Lifetime and Temperature vs Oxygen Concentration', fontsize=17)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
