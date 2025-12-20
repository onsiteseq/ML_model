import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- НАСТРОЙКИ ПУТЕЙ ---
# Укажите путь к папке, где лежат ваши .avi файлы
VIDEO_DIR = './data/mcd_rppg_10sec'           
# Папка, куда будут сохранены извлеченные сигналы
OUTPUT_DIR = './data/signals'  
# Количество потоков (зависит от количества ядер CPU вашей VM)
NUM_WORKERS = 8                

def extract_signal_low_res(video_path):
    """
    Извлекает средний RGB сигнал из низкоразрешенного видео лица.
    Оптимизировано для файлов размером ~79x84.
    """
    cap = cv2.VideoCapture(video_path)
    signals = []
    
    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        
        # Поскольку лицо уже вырезано и отцентрировано,
        # мы берем центральную область (80% кадра).
        # Это исключает артефакты на границах видео.
        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        
        # Обрезаем до центрального ROI
        roi = frame[y1:y2, x1:x2]
        
        # Вычисляем среднее значение каналов. 
        # OpenCV читает в BGR, переводим в RGB для модели.
        mean_bgr = cv2.mean(roi)[:3]
        signals.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]]) 
        
    cap.release()
    return np.array(signals, dtype=np.float32)

def process_file(video_name):
    """Обработка одного файла для ProcessPoolExecutor"""
    video_path = os.path.join(VIDEO_DIR, video_name)
    output_path = os.path.join(OUTPUT_DIR, video_name.replace('.avi', '.npy'))
    
    # Пропускаем, если файл уже обработан
    if os.path.exists(output_path):
        return
        
    signal = extract_signal_low_res(video_path)
    
    if signal is not None and len(signal) > 0:
        np.save(output_path, signal)

def main():
    # Создаем папку для сигналов, если её нет
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Собираем список всех AVI файлов
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    print(f"Найдено видео-файлов для обработки: {len(video_files)}")

    # Запуск многопоточного извлечения
    print(f"Запуск в {NUM_WORKERS} потоках...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_file, video_files), total=len(video_files), desc="Обработка"))

    print(f"\nГотово! Сигналы сохранены в папку: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()