import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

scaler_path = MODEL_DIR / "keystroke_scaler.joblib"
weights_path = MODEL_DIR / "keystroke_cnn_weights.pth"

class Keystroke1DCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super(Keystroke1DCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        # Use adaptive pooling to handle variable-length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool1d(7)
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7, 256), 
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x shape: (batch, length)
        x = x.unsqueeze(1)  # (batch, 1, length)
        x = self.conv_block(x)  # (batch, 64, length_reduced)
        x = self.adaptive_pool(x)  # (batch, 64, 7) - fixed size
        
        # Flatten for FC layer
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, 64*7=448)
        
        embedded = self.fc_block(x)
        return nn.functional.normalize(embedded, p=2, dim=1)

device = torch.device('cpu') 

print("Loading standardizer...")
local_scaler = joblib.load(scaler_path)

print("Loading Neural Network...")
local_model = Keystroke1DCNN(embedding_dim=64).to(device)

local_model.load_state_dict(torch.load(weights_path, map_location=device))

local_model.eval() 
print("Biometric Authentication Module Ready.")