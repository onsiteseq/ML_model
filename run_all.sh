#!/bin/bash
# run_all.sh

echo "--- Шаг 0: Установка зависимостей ---"
# Используем python3 -m pip для гарантии установки в нужный Python
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install opencv-python pandas numpy scipy scikit-learn tqdm mediapipe

echo "--- Шаг 1: Предобработка видео (AVI -> NPY) ---"
python3 preprocess_to_npy.py

echo "--- Шаг 2: Разделение данных на Train/Val ---"
python3 split_data.py

echo "--- Шаг 3: Запуск обучения на V100 ---"
python3 train_full.py
