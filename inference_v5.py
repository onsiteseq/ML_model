import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# --- Архитектура v5 (должна точно совпадать с train_full_v5.py) ---
class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//16), 
            nn.ReLU(), 
            nn.Linear(channels//16, channels), 
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(x.mean(dim=-1)).view(b, c, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride, 1), 
            nn.BatchNorm1d(out_c), 
            nn.ReLU(), 
            nn.Dropout1d(dropout),
            nn.Conv1d(out_c, out_c, 3, 1, 1), 
            nn.BatchNorm1d(out_c)
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride), 
            nn.BatchNorm1d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()
    
    def forward(self, x): 
        return F.relu(self.conv(x) + self.skip(x))

class ImprovedRPPGNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 7, 1, 3), 
            nn.ReLU(), 
            ResBlock(64, 128, 2), 
            SEBlock(128),
            ResBlock(128, 256, 2), 
            SEBlock(256),
            ResBlock(256, 256, 2), 
            SEBlock(256),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(), 
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.net(x)

def run_inference(signal_path, model_path='best_rppg_model_v4.pth'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_size = 600

    # 1. Загрузка модели
    model = ImprovedRPPGNetV4().to(device)
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл весов {model_path} не найден!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Загрузка и предобработка сигнала (.npy)
    try:
        sig = np.load(signal_path)
    except Exception as e:
        print(f"Ошибка при загрузке файла {signal_path}: {e}")
        return

    # Подгонка под окно 600 кадров
    if len(sig) > window_size:
        sig = sig[:window_size]
    else:
        sig = np.pad(sig, ((0, max(0, window_size - len(sig))), (0, 0)))

    # Нормализация (такая же, как в Dataset)
    sig_mean = sig.mean(axis=0)
    sig_std = sig.std(axis=0) + 1e-6
    sig = (sig - sig_mean) / sig_std

    # Подготовка тензора (Batch, Channels, Length)
    input_tensor = torch.tensor(sig, dtype=torch.float32).T.unsqueeze(0).to(device)

    # 3. Предсказание
    with torch.no_grad():
        prediction = model(input_tensor)
        sbp, dbp = prediction[0].cpu().numpy()

    print(f"\n--- Результаты анализа ---")
    print(f"Файл: {os.path.basename(signal_path)}")
    print(f"Систолическое (SBP): {sbp:.1f} mmHg")
    print(f"Диастолическое (DBP): {dbp:.1f} mmHg")
    print(f"--------------------------\n")

if __name__ == "__main__":
    # Пример запуска: укажите путь к вашему .npy файлу
    test_file = './data/signals/MCD_front__mcd__1020__front__after__6.npy'
    
    if os.path.exists(test_file):
        run_inference(test_file)
    else:
        print("Пожалуйста, укажите существующий .npy файл в переменной test_file.")