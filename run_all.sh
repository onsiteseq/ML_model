#!/bin/bash
# run_all.sh

echo "--- Шаг 0: Установка зависимостей ---"
pip install torch torchvision torchaudio
pip install opencv-python pandas numpy scipy scikit-learn tqdm mediapipe

echo "--- Шаг 1: Предобработка видео (AVI -> NPY) ---"
# Извлекаем RGB-сигналы из видео низкого разрешения
python3 preprocess_to_npy.py

echo "--- Шаг 2: Разделение данных на Train/Val ---"
# Сопоставляем чанки с метаданными из mcd_rppg.csv и делим по фолдам
python3 split_data.py

echo "--- Шаг 3: Запуск обучения на V100 ---"
# Обучаем модель ResRPPG-Net предсказывать давление
python3 train_full.py