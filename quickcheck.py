import cv2
import os

video_path = 'MCD_front__mcd__1020__front__after__0.avi'
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        print(f"Разрешение видео: {frame.shape[1]}x{frame.shape[0]}")
        print(f"Количество кадров: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    cap.release()
else:
    print("Файл не найден. Проверьте путь.")