# Remote Blood Pressure Estimation (rPPG to SBP/DBP)

Проект по бесконтактному измерению артериального давления на основе технологии дистанционной фотоплетизмографии (rPPG) с использованием глубокого обучения.

## О кейсе (Case 2)
Данное исследование посвящено оценке артериального давления (SBP/DBP) путем анализа микроизменений цвета кожи лица на видео. 
* **Входные данные**: Видеопоток лица в формате AVI/MP4.
* **Целевые показатели**: Систолическое (SBP) и Диастолическое (DBP) артериальное давление в мм рт. ст..
* **Датасет**: MCD-rPPG (3600 записей, 600 испытуемых).

## Baseline (Эталонные модели)
В качестве базовых показателей используются результаты классических rPPG-архитектур на датасете MCD-rPPG:
* **TS-CAN**: MAE 14.77 (SBP) / 8.24 (DBP)
* **DeepPhys**: MAE 14.12 (SBP) / 8.04 (DBP)
* **PhysNet**: MAE 13.54 (SBP) / 7.40 (DBP)

## Улучшенная модель (ImprovedRPPGNetV4)
Наша модель основана на глубокой 1D-сверточной сети с рядом оптимизаций для повышения точности:
* **Увеличенное окно**: Анализ 600 кадров (20 секунд сигнала) вместо 300 для захвата большего количества пульсовых циклов.
* **Архитектура**: 3 блока ResNet с каналами до 256 и встроенными SE-блоками (Squeeze-and-Excitation).
* **Регуляризация**: Dropout 0.4 и Weight Decay 1e-4 для предотвращения переобучения.
* **Аугментация**: Добавление гауссовского шума в процессе обучения для устойчивости к условиям освещения.



## Метрики и результаты
Оценка проводилась по средней абсолютной ошибке (MAE) на независимой выборке пациентов (Subject-Independent Split).

| Модель | SBP MAE (верхнее) | DBP MAE (нижнее) |
| :--- | :---: | :---: |
| Baseline (PhysNet) | 13.54 | 7.40 |
| **Наша модель (v5)** | **13.56** | **7.71** |

Нам удалось достичь точности уровня **PhysNet**, значительно превзойдя при этом модели **DeepPhys** и **TS-CAN**.

## Как запустить код

### 1. Подготовка окружения и данных
```bash
sudo apt update && sudo apt install lftp p7zip-full python3-pip libgl1 libglib2.0-0t64 python3.12-venv -y
sudo lftp -u 'логин','пароль' -e "set sftp:auto-confirm yes; pget -n 5 sftp://45.89.225.65:2222/upload/videos.zip; quit"
mkdir -p /home/gorbenkoteh/data
7z x /home/gorbenkoteh/videos.zip -o/home/gorbenkoteh/data
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio opencv-python pandas numpy scipy scikit-learn tqdm mediapipe

### 2. Запуск пейплайн
# Шаг 1: Предобработка видео (извлечение сигналов в .npy)
python3 preprocess_to_npy.py

# Шаг 2: Разделение данных (сопоставление с db.csv и деление по пациентам)
python3 split_data.py

# Шаг 3: Обучение модели на GPU (Tesla V100)
python3 train_full.py
