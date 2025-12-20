import pandas as pd
import re
import os

def update_csv_with_db(target_csv, db_df):
    if not os.path.exists(target_csv):
        print(f"Файл {target_csv} не найден.")
        return

    df = pd.read_csv(target_csv)
    
    # Создаем словарь для быстрого поиска: {(patient_id, step): (sbp, dbp)}
    # Убираем дубликаты, так как в db.csv по 3 строки на замер (разные камеры)
    db_lookup = db_df.drop_duplicates(['patient_id', 'step']).set_index(['patient_id', 'step'])
    
    new_sbp = []
    new_dbp = []
    
    for filename in df['filename']:
        # Извлекаем ID (например, 1020) и состояние (after/before) из MCD_front__mcd__1020__front__after__6.npy
        p_id_match = re.search(r'mcd__(\d+)', filename)
        step_match = re.search(r'(before|after)', filename)
        
        if p_id_match and step_match:
            p_id = int(p_id_match.group(1))
            step = step_match.group(1)
            
            try:
                # Ищем давление в db.csv
                row = db_lookup.loc[(p_id, step)]
                new_sbp.append(row['upper_ap'])
                new_dbp.append(row['lower_ap'])
            except KeyError:
                # Если данных нет, оставляем заглушку или ставим среднее
                new_sbp.append(120.0)
                new_dbp.append(80.0)
        else:
            new_sbp.append(120.0)
            new_dbp.append(80.0)
            
    df['sbp'] = new_sbp
    df['dbp'] = new_dbp
    
    df.to_csv(target_csv, index=False)
    print(f"Обновлен файл: {target_csv}")

if __name__ == "__main__":
    # Загружаем базу данных с реальными метками
    db = pd.read_csv('db.csv')
    
    # Обновляем оба файла
    update_csv_with_db('train_labels.csv', db)
    update_csv_with_db('val_labels.csv', db)