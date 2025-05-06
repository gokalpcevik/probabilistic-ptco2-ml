# debug_compare.py  ── run with  python debug_compare.py tau  temp  pulses
import sys, numpy as np, torch
from pathlib import Path

if len(sys.argv) != 4:
    print("usage:  python debug_compare.py  <tau µs>  <temp °C>  <pulses>")
    sys.exit(1)

tau_raw, temp_raw, pulses_raw = map(float, sys.argv[1:])

stats = np.load("models/normalization_stats_homo.npz")
x_min   = stats["min"]          # (3,)
x_range = stats["max"] - stats["min"]

def normalise(v_raw, v_min, v_rng):
    return (v_raw - v_min) / v_rng

x_norm = np.array([
    normalise(tau_raw,   x_min[0], x_range[0]),
    normalise(temp_raw,  x_min[1], x_range[1]),
    normalise(np.log1p(pulses_raw), x_min[2], x_range[2])  # log1p exactly as MCU
], dtype=np.float32)

model = torch.load("models/quantized_regressor_stage2.pth",
                   map_location="cpu",
                   weights_only=False).eval()

# Grab input‑tensor quant params from model.quant
INPUT_SCALE      = float(model.quant.scale)
INPUT_ZERO_POINT = int(model.quant.zero_point)

def quantise(f):
    return int(round(f / INPUT_SCALE)) + INPUT_ZERO_POINT

x_q = np.vectorize(quantise, otypes=[np.int32])(x_norm).astype(np.int8)

# Quantised uint8 storage (e.g. in PyTorch)
x_q_u8 = (x_q.astype(np.int16) + 128).astype(np.uint8)

# This is what MCU gets before subtracting INPUT_ZERO_POINT = -128
x_q_mcu = x_q

layer_outputs = {}       # layer‑idx → np.ndarray int8
def make_hook(name):
    def _hook(mod, inp, out):
        # out is still a quantised Tensor until it reaches DeQuantStub
        q = out.int_repr() if hasattr(out, "int_repr") else None
        if q is not None:
            layer_outputs[name] = q.cpu().numpy()
    return _hook

qlinears = [m for m in model.modules()
            if isinstance(m, torch.nn.quantized.Linear)]
for idx, m in enumerate(qlinears):
    m.register_forward_hook(make_hook(f"layer{idx}"))

# ─────────────────────────────────────────────
# 5.  Forward pass  ---------------------------
# ─────────────────────────────────────────────
with torch.inference_mode():
    y_pred = model(torch.tensor(x_norm).unsqueeze(0))  # float result

def dump(arr, tag, head=16):
    mn, mx = arr.min(), arr.max()
    mean   = arr.mean()
    head_s = ", ".join(map(str, arr.flatten()[:head]))
    extra  = ", ..." if arr.size > head else ""
    print(f"{tag}: min={mn:3d}  max={mx:3d}  mean={mean:6.2f}  first=[{head_s}{extra}]")

print("\n─── INPUT ─────────────────────────────────────")
print(f"raw    : τ={tau_raw:.3f}  T={temp_raw:.3f}  pulses={pulses_raw}")
print(f"norm   : {x_norm}")
print(f"quant (uint8): {x_q_u8}")   # what PyTorch stores (on-disk or export)
print(f"quant (int8) : {x_q}")      # what CMSIS-NN gets (after subtracting 128)


print("\n─── LAYERS ────────────────────────────────────")
for l in sorted(layer_outputs.keys(), key=lambda s: int(s[5:])):
    dump(layer_outputs[l], l)

print("\n─── OUTPUT ────────────────────────────────────")
print(f"y_pred (float de‑quant) = {y_pred.item():.6f}")
