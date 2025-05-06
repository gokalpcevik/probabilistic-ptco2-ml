import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from collections import defaultdict

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

DATA_DIR = './data/processed/Field'

po2_vals = []
ptco2_preds = []

grouped = defaultdict(list)  # {po2: [ptco2_1, ptco2_2, ...]}

for filename in os.listdir(DATA_DIR):
    if not filename.endswith('.json'):
        continue

    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        data = json.load(f)

    true_po2 = data.get('po2')
    if true_po2 is None:
        continue

    for m in data['measurements']:
        pred = m.get('ptco2')
        if pred is not None:
            grouped[true_po2].append(pred)
            po2_vals.append(true_po2)
            ptco2_preds.append(pred)

po2_vals = np.array(po2_vals)
ptco2_preds = np.array(ptco2_preds)

r2 = r2_score(po2_vals, ptco2_preds)

mean_vals = []
std_devs = []
po2_sorted = sorted(grouped.keys())

for po2 in po2_sorted:
    preds = np.array(grouped[po2])
    mean_vals.append(np.mean(preds))
    std_devs.append(np.std(preds))

# Plot
plt.figure(figsize=(8, 6))

plt.scatter(po2_vals, ptco2_preds, alpha=0.6, s=30, edgecolors='k', label='Predictions')

plt.errorbar(po2_sorted, mean_vals, yerr=std_devs, fmt='o', capsize=4,
             color='red', label='Mean ± Std Dev')

# Ideal y=x line
min_val = min(min(po2_vals), min(ptco2_preds))
max_val = max(max(po2_vals), max(ptco2_preds))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='Ideal (y = x)')

r2_text = f"$R^2$ = {r2:.4f}"
plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Styling
plt.title('Predicted vs Ground Truth PtcO₂', fontsize=14)
plt.xlabel('Ground Truth pO₂ [cmHg]', fontsize=12)
plt.ylabel('Predicted PtcO₂ [cmHg]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
