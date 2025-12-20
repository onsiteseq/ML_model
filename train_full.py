import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# --- Блоки модели ---
class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels, channels//16), nn.ReLU(), nn.Linear(channels//16, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(x.mean(dim=-1)).view(b, c, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_c, out_c, 3, stride, 1), nn.BatchNorm1d(out_c), nn.ReLU(), nn.Conv1d(out_c, out_c, 3, 1, 1), nn.BatchNorm1d(out_c))
        self.skip = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride), nn.BatchNorm1d(out_c)) if stride != 1 or in_c != out_c else nn.Identity()
    def forward(self, x): return F.relu(self.conv(x) + self.skip(x))

class ImprovedRPPGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, 32, 7, 1, 3), nn.ReLU(), ResBlock(32, 64, 2), SEBlock(64), ResBlock(64, 128, 2), SEBlock(128), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 2))
    def forward(self, x): return self.net(x)

# --- Данные ---
class FastRPPGDataset(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sig = np.load(os.path.join('./data/signals', row['filename']))
        # Padding/Crop to 300
        if len(sig) > 300: sig = sig[:300]
        else: sig = np.pad(sig, ((0, max(0, 300-len(sig))), (0, 0)))
        # Normalize
        sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-6)
        return torch.tensor(sig, dtype=torch.float32).T, torch.tensor([row['sbp'], row['dbp']], dtype=torch.float32)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(FastRPPGDataset('train_labels.csv'), batch_size=64, shuffle=True)
    val_loader = DataLoader(FastRPPGDataset('val_labels.csv'), batch_size=64)
    model = ImprovedRPPGNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    for e in range(50):
        model.train()
        for s, l in train_loader:
            opt.zero_grad()
            crit(model(s.to(device)), l.to(device)).backward()
            opt.step()
        model.eval()
        val_mae = sum(crit(model(s.to(device)), l.to(device)).item() for s, l in val_loader) / len(val_loader)
        print(f"Epoch {e+1} | Val MAE: {val_mae:.4f}")