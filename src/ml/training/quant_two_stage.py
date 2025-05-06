import torch
from src.ml.pO2regressor import QuantizablepO2Regressor
from src.ml.training.train_sternvolmer import QuantizableSternVolmerMLP
from src.data_processing.o2_dataset import OxygenDataset
from torch.utils.data import DataLoader
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
import numpy as np

# Load trained models
regressor = QuantizablepO2Regressor(input_dim=3)
regressor.load_state_dict(torch.load("models/regressor_stage2.pth"))
regressor.eval()

physics = QuantizableSternVolmerMLP(input_dim=2)
physics.load_state_dict(torch.load("models/physics_stage2.pth"))
physics.eval()

# Automatically fuse Linear + ReLU pairs inside .net
fuse_list = []
for i in range(len(regressor.net) - 1):
    if isinstance(regressor.net[i], torch.nn.Linear) and isinstance(regressor.net[i + 1], torch.nn.ReLU):
        fuse_list.append([f"net.{i}", f"net.{i + 1}"])
torch.quantization.fuse_modules(regressor, fuse_list, inplace=True)

fuse_list = []
for i in range(len(physics.net) - 1):
    if isinstance(physics.net[i], torch.nn.Linear) and isinstance(physics.net[i + 1], torch.nn.ReLU):
        fuse_list.append([f"net.{i}", f"net.{i + 1}"])
torch.quantization.fuse_modules(physics, fuse_list, inplace=True)


regressor.qconfig = torch.quantization.get_default_qconfig("fbgemm")
physics.qconfig = torch.quantization.get_default_qconfig("fbgemm")


# sym_act   = MinMaxObserver.with_args(dtype=torch.qint8,  
#                                     qscheme=torch.per_tensor_symmetric)
# sym_weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
#                                               qscheme=torch.per_channel_symmetric)

# qconfig = QConfig(activation=sym_act, weight=sym_weight)

# regressor.qconfig = qconfig
# physics.qconfig = qconfig
print(regressor.qconfig)
# Prepare for calibration
torch.ao.quantization.prepare(regressor, inplace=True)
torch.ao.quantization.prepare(physics, inplace=True)

# Load test data for calibration
stats = np.load("models/normalization_stats_homo.npz")
min_vals = stats["min"]
max_vals = stats["max"]
ranges = max_vals - min_vals
ranges[ranges == 0.0] = 1.0

test_dataset = OxygenDataset("data/processed/", split="test", normalize=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Calibration (no gradients)
with torch.no_grad():
    for features, _ in test_loader:
        _ = regressor(features)
        physics_input = features[:, 1:3]
        _ = physics(physics_input)

# Convert to quantized models
quantized_regressor = torch.quantization.convert(regressor.eval(), inplace=False)
quantized_physics = torch.quantization.convert(physics.eval(), inplace=False)

# Save
torch.save(quantized_regressor, "models/quantized_regressor_stage2.pth")
torch.save(quantized_physics, "models/quantized_physics_stage2.pth")

print(">> Quantization complete.")
