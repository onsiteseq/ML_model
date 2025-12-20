import pandas as pd
import os
import glob

def split_mcd_chunks(csv_path, signal_dir, val_fold=4):
    # mcd_rppg.csv содержит 1200 записей и колонку fold
    df = pd.read_csv(csv_path)
    df['cond_key'] = df['file'].apply(lambda x: 'after' if 'after' in x else 'before')
    all_chunks = []

    for _, row in df.iterrows():
        p_id, cond, fold = row['patient_id'], row['cond_key'], row['fold']
        # Ищем все чанки этого пациента (MCD_front__mcd__1020...)
        pattern = os.path.join(signal_dir, f"*mcd__{p_id}__*{cond}__*.npy")
        found_files = glob.glob(pattern)
        for f in found_files:
            all_chunks.append({
                'filename': os.path.basename(f),
                'fold': fold,
                'sbp': 120, # PLACEHOLDER: Нужны реальные метки АД
                'dbp': 80
            })

    chunks_df = pd.DataFrame(all_chunks)
    train_df = chunks_df[chunks_df['fold'] != val_fold]
    val_df = chunks_df[chunks_df['fold'] == val_fold]
    train_df.to_csv('train_labels.csv', index=False)
    val_df.to_csv('val_labels.csv', index=False)
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}")

if __name__ == "__main__":
    split_mcd_chunks('mcd_rppg.csv', './data/signals')