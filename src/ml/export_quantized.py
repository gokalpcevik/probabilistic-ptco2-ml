import torch, numpy as np, textwrap, sys, math

INT32_MAX = 2**31 - 1


def quantize_multiplier(real):
    """Return (multiplier, shift) so that
    out = (val * multiplier) / (2**(-shift))       if shift < 0
        = (val * multiplier) * (2**shift)         if shift ≥ 0
    and |shift| ≤ 31, 0 ≤ multiplier < 2**31."""
    if real == 0.0:
        return 0, 0

    shift = 0
    while real < 0.5:
        real *= 2.0
        shift -= 1 

    while real >= 1.0:
        real *= 0.5
        shift += 1  

    multiplier = int(round(real * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1

    return multiplier, shift


model = torch.load(
    "models/quantized_regressor_stage2.pth", map_location="cpu", weights_only=False
).eval()

print("model.quant", model.quant)            # scale & zero_point for INPUT
for i,m in enumerate(model.modules()):
    if isinstance(m, torch.nn.quantized.Linear):
        print(f"layer{i}: w ZP unique →", 
              m.weight().q_per_channel_zero_points().unique()[:5])


# torch.no_grad()

stats = np.load("models/normalization_stats_homo.npz")
min_vals, max_vals = stats["min"], stats["max"]
range_vals = max_vals - min_vals

h = []
push = h.append
push("/* Auto-generated - DO NOT EDIT */")
push("#ifndef QUANTIZED_MODEL_WEIGHTS_H")
push("#define QUANTIZED_MODEL_WEIGHTS_H")
push("")
push("#include <stdint.h>")
push('#include "arm_nn_types.h"')
push("")

# ------------- input normalisation -----------------------------------
push("// Normalisation constants   x_norm = (x_raw-min) / range")
push(
    "static const float INPUT_MIN[3]   = { "
    + ", ".join(f"{v:.8f}f" for v in min_vals)
    + " };"
)
push(
    "static const float INPUT_RANGE[3] = { "
    + ", ".join(f"{v:.8f}f" for v in range_vals)
    + " };"
)
push("")

# ------------- input tensor quantisation -----------------------------
inp_scale = float(model.quant.scale)
print(f"Input scale: {model.quant.scale}f")
print(f"Input scale: {inp_scale}f")
inp_zp = int(model.quant.zero_point)

push("// Input tensor quantisation params (torch.quant.QuantStub)")
push(f"static const float   INPUT_SCALE      = {inp_scale}f;")
push(f"static const int32_t INPUT_ZERO_POINT = {inp_zp};")
push("")

# ------------- iterate through quantised Linear layers --------------
layer_id = 0
prev_zp = inp_zp  # needed for next layer's input_offset
prev_scale = inp_scale

linear_modules = [
    m for m in model.modules() if isinstance(m, torch.nn.quantized.Linear)
]
total_linear = len(linear_modules)

for m in linear_modules:
    # Fetch data tensors ------------------------------------------------
    Wq = m.weight()  # q_per_channel QInt8
    Bq = m.bias()  # int32 already scaled
    w_scales = Wq.q_per_channel_scales().numpy()  # (out_ch,)

    # ---- weights --------------------------------------------------------
    W_int8 = Wq.int_repr().numpy().astype(np.int32).flatten()
    B_int = Bq.detach().numpy().astype(np.int32)

    out_scale = float(m.scale)
    out_zp = int(m.zero_point)
    out_ch = Wq.size(0)

    # Per‑channel multipliers / shifts ---------------------------------
    mult_arr, shift_arr = [], []
    for ws in w_scales:
        real_mult = (prev_scale * ws) / out_scale
        m_i, s_i = quantize_multiplier(real_mult)
        mult_arr.append(m_i)
        shift_arr.append(s_i)

    # Activation bounds -------------------------------------------------
    if layer_id < total_linear - 1:  # hidden layers
        act_min, act_max = out_zp, 127  # use ZERO_POINT_i
    else:  # head
        act_min, act_max = -128, 127

    # ---------- emit C arrays -----------------------------------------
    push(f"// ── Layer {layer_id} ─────────────────────────────────────")
    push(
        f"__attribute__((aligned(16))) "
        f"static const int8_t WEIGHTS_{layer_id}[] = {{ "
        + ", ".join(map(str, W_int8))
        + " };"
    )
    push(
        f"__attribute__((aligned(4))) "
        f"static const int32_t BIASES_{layer_id}[] = {{ "
        + ", ".join(map(str, B_int))
        + " };"
    )

    push(f"static const float   SCALE_{layer_id}      = {out_scale:.10f}f;")
    push(f"static const int32_t ZERO_POINT_{layer_id} = {out_zp};")

    push(
        f"__attribute__((aligned(4))) "
        f"static const int32_t MULTIPLIER_{layer_id}[] = {{ "
        + ", ".join(map(str, mult_arr))
        + " };"
    )
    push(
        f"__attribute__((aligned(4))) "
        f"static const int32_t SHIFT_{layer_id}[]     = {{ "
        + ", ".join(map(str, shift_arr))
        + " };"
    )

    push(
        f"static const cmsis_nn_per_channel_quant_params "
        f"PER_CHANNEL_PARAMS_{layer_id} = {{"
    )
    push(f"    (int32_t*)MULTIPLIER_{layer_id},")
    push(f"    (int32_t*)SHIFT_{layer_id}")
    push(f"}};")

    # ---------- fc‑params ---------------------------------------------
    input_offset_expr = (
        f"-INPUT_ZERO_POINT" if layer_id == 0 else f"-ZERO_POINT_{layer_id-1}"
    )  # ★ FIX 3 ★

    push(f"static const cmsis_nn_fc_params FC_PARAMS_{layer_id} = {{")
    push(f"    {input_offset_expr},      /* input_offset  */")
    push(f"    0,                        /* filter_offset */")
    push(f"    ZERO_POINT_{layer_id},    /* output_offset */")
    push(f"    {{ {act_min}, {act_max} }}")  # activation bounds
    push("};\n")

    # ---------- prepare for next layer --------------------------------
    prev_scale = out_scale
    prev_zp = out_zp
    layer_id += 1

push("#endif /* QUANTIZED_MODEL_WEIGHTS_H */\n")

out_path = "outputs/mcu/quantized_model_weights.h"
with open(out_path, "w", encoding="utf-8") as f:  # Specify UTF-8 encoding
    f.write("\n".join(h))

print(f"Header file saved →  {out_path}")
