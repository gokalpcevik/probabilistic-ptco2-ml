import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.quantization
import torch.nn.quantized as nnq
import numpy as np
from src.data_processing.o2_dataset import OxygenDataset

# --- Model ---
class pO2Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    

class QuantizablepO2Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
    def forward(self, x):
       x = self.quant(x)
       x = self.net(x)
       x = self.dequant(x)
       return x.squeeze(-1)
   
