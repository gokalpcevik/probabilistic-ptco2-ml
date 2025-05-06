"""
Export FP32 weights and biases from a trained PyTorch model
to a C header for MCU inference.
"""

import torch
import numpy as np
import os

from src.ml.pO2regressor import pO2Regressor 

input_dim = 3 
model = pO2Regressor(input_dim)
state_dict = torch.load("models/regressor_stage2.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("Loaded model:")
print(model)


stats = np.load("models/normalization_stats_homo.npz")
min_vals = stats["min"]
range_vals = stats["max"] - stats["min"]

assert min_vals.shape[0] == 3, "Expected 3 input features"


h = []
push = h.append

push("/* Auto-generated - DO NOT EDIT */")
push("/**")
push(" * Floating-point Neural Network Weights for pO2 Prediction")
push(" * Architecture: MLP with ReLU activations")
push(" * Format: Weights are stored row-major (output x input)")
push(" */")
push("")
push("#ifndef FP32_MODEL_WEIGHTS_H")
push("#define FP32_MODEL_WEIGHTS_H")
push("")
push("#include <stdint.h>")
push("")


push("/** Input normalization constants */")
push(
    "static const float INPUT_NORM_MIN[3]   = { " +
    ", ".join(f"{v:.8f}f" for v in min_vals) + " };"
)
push(
    "static const float INPUT_NORM_RANGE[3] = { " +
    ", ".join(f"{v:.8f}f" for v in range_vals) + " };"
)
push("")

linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
for i, layer in enumerate(linear_layers):
    w = layer.weight.detach().cpu().numpy()  # (out_features, in_features)
    b = layer.bias.detach().cpu().numpy()    # (out_features,)

    push(f"/** Layer {i}: Linear({w.shape[1]} → {w.shape[0]}) */")
    
    # Weights (flattened row-major)
    push(f"static const float L{i}_WEIGHTS[{w.size}] = {{")
    w_flat = w.flatten()
    rows = [
        "    " + ", ".join(f"{val:.8f}f" for val in w_flat[i:i+8])
        for i in range(0, len(w_flat), 8)
    ]
    push(",\n".join(rows))
    push("};")

    # Biases
    push(f"static const float L{i}_BIASES[{b.size}] = {{")
    b_rows = [
        "    " + ", ".join(f"{val:.8f}f" for val in b[i:i+8])
        for i in range(0, len(b), 8)
    ]
    push(",\n".join(b_rows))
    push("};")
    push("")
         
push("/** Normalize input values to [0,1] */")
push("static inline void normalize_input(const float raw[3], float norm[3]) {")
push("    for (int i = 0; i < 3; ++i) {")
push("        float val = raw[i];")
push("        if (val < INPUT_NORM_MIN[i]) val = INPUT_NORM_MIN[i];")
push("        float max_val = INPUT_NORM_MIN[i] + INPUT_NORM_RANGE[i];")
push("        if (val > max_val) val = max_val;")
push("        norm[i] = (val - INPUT_NORM_MIN[i]) / INPUT_NORM_RANGE[i];")
push("    }")
push("}")
push("")


push("#endif /* FP32_MODEL_WEIGHTS_H */\n")


os.makedirs("outputs/mcu", exist_ok=True)
out_path = "outputs/mcu/fp32_model_weights.h"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(h))

print(f"> Export complete → {out_path}")
