import pandas as pd
import os
import glob
import re
from sklearn.model_selection import train_test_split

def create_split():
    # 1. Загружаем базу данных с метками
    db_path = 'db.csv'
    if not os.path.exists(db_path):
        print(f"Ошибка: Файл {db_path} не найден!")
        return
    
    db = pd.read_csv(db_path)
    
    # Оставляем только нужные колонки и убираем дубликаты (т.к. там по 3 камеры на замер)
    db_labels = db[['patient_id', 'step', 'upper_ap', 'lower_ap']].drop_duplicates()
    
    # 2. Получаем список уникальных ID пациентов для честного разделения
    unique_patients = db['patient_id'].unique()
    train_ids, val_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    print(f"Всего пациентов: {len(unique_patients)}")
    print(f"Пациентов в обучении: {len(train_ids)}")
    print(f"Пациентов в валидации: {len(val_ids)}")

    # 3. Собираем информацию о всех доступных .npy файлах
    signal_dir = './data/signals'
    all_files = glob.glob(os.path.join(signal_dir, "*.npy"))
    
    train_data = []
    val_data = []
    
    print("Обработка файлов и сопоставление с метками...")
    for f_path in all_files:
        filename = os.path.basename(f_path)
        
        # Извлекаем ID (например, 1020) и состояние (before/after) из имени файла
        id_match = re.search(r'mcd__(\d+)', filename)
        step_match = re.search(r'(before|after)', filename)
        
        if id_match and step_match:
            p_id = int(id_match.group(1))
            step = step_match.group(1)
            
            # Ищем давление в базе
            label_row = db_labels[(db_labels['patient_id'] == p_id) & (db_labels['step'] == step)]
            
            if not label_row.empty:
                sbp = label_row.iloc[0]['upper_ap']
                dbp = label_row.iloc[0]['lower_ap']
                
                entry = {'filename': filename, 'sbp': sbp, 'dbp': dbp}
                
                # Распределяем по группам в зависимости от ID пациента
                if p_id in train_ids:
                    train_data.append(entry)
                else:
                    val_data.append(entry)

    # 4. Сохраняем новые файлы меток
    pd.DataFrame(train_data).to_csv('train_labels.csv', index=False)
    pd.DataFrame(val_data).to_csv('val_labels.csv', index=False)
    
    print(f"Готово! Создано меток для обучения: {len(train_data)}")
    print(f"Готово! Создано меток для валидации: {len(val_data)}")

if __name__ == "__main__":
    create_split()