import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = 'data' 
TIMESTEP_US = 2    
NUM_SAMPLES = 60   

def load_decay_curves(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    decay_curves = [m['decay_curve'] for m in data]
    return np.array(decay_curves)

def main():
    files = sorted(os.listdir(DATA_DIR))

    time_axis = np.arange(0, NUM_SAMPLES * TIMESTEP_US, TIMESTEP_US)  # [0, 2, 4, ..., 118]

    plt.figure(figsize=(12, 8))

    for filename in files:
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(DATA_DIR, filename)
        decay_curves = load_decay_curves(filepath)

        mean_decay_curve = np.mean(decay_curves, axis=0)

        base_name = os.path.splitext(filename)[0]  # remove .json
        oxygen_percentage = float(base_name) / 100.0

        plt.plot(time_axis, mean_decay_curve, label=f"{oxygen_percentage:.2f}% O₂")

    plt.xlabel('Time (µs)', fontsize=14)
    plt.ylabel('Luminescence Intensity (a.u.)', fontsize=14)
    plt.title('Luminescence Decay Curves vs. Time', fontsize=16)
    plt.legend(title="Oxygen Level", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
