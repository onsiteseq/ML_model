import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import datetime

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
        # Путь к сигналам
        sig_path = os.path.join('./data/signals', row['filename'])
        sig = np.load(sig_path)
        
        # Подгонка под 300 кадров
        if len(sig) > 300: sig = sig[:300]
        else: sig = np.pad(sig, ((0, max(0, 300-len(sig))), (0, 0)))
        
        # Нормализация
        sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-6)
        return torch.tensor(sig, dtype=torch.float32).T, torch.tensor([row['sbp'], row['dbp']], dtype=torch.float32)

def log_message(message, log_file="training_log.txt"):
    """Печать в консоль и запись в файл."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_file = "training_log.txt"
    
    # Очистка лога перед началом
    with open(log_file, "w") as f:
        f.write(f"=== Training Started: {datetime.datetime.now()} ===\n")

    train_loader = DataLoader(FastRPPGDataset('train_labels.csv'), batch_size=64, shuffle=True)
    val_loader = DataLoader(FastRPPGDataset('val_labels.csv'), batch_size=64)
    
    model = ImprovedRPPGNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.L1Loss() # MAE Loss
    
    best_val_mae = float('inf')
    num_epochs = 50

    for e in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for s, l in train_loader:
            opt.zero_grad()
            pred = model(s.to(device))
            loss = crit(pred, l.to(device))
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_mae_total = 0.0
        val_sbp_mae = 0.0
        val_dbp_mae = 0.0
        
        with torch.no_grad():
            for s, l in val_loader:
                pred = model(s.to(device))
                target = l.to(device)
                
                # Общий MAE
                val_mae_total += crit(pred, target).item()
                
                # Раздельные метрики для SBP и DBP
                val_sbp_mae += F.l1_loss(pred[:, 0], target[:, 0]).item()
                val_dbp_mae += F.l1_loss(pred[:, 1], target[:, 1]).item()
        
        avg_val_mae = val_mae_total / len(val_loader)
        avg_sbp_mae = val_sbp_mae / len(val_loader)
        avg_dbp_mae = val_dbp_mae / len(val_loader)

        # Логирование результатов
        msg = (f"Epoch {e+1:02d}/{num_epochs} | "
               f"Train Loss: {avg_train_loss:.4f} | "
               f"Val MAE: {avg_val_mae:.4f} (SBP: {avg_sbp_mae:.2f}, DBP: {avg_dbp_mae:.2f})")
        log_message(msg, log_file)

        # Сохранение лучшей модели
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), 'best_rppg_model.pth')
            log_message(f"--> New best model saved with Val MAE: {avg_val_mae:.4f}", log_file)

    log_message(f"=== Training Finished: {datetime.datetime.now()} ===", log_file)