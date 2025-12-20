from huggingface_hub import snapshot_download

# Скачиваем весь датасет в указанную папку
snapshot_download(
    repo_id="kyegorov/mcd_rppg", 
    repo_type="dataset", 
    local_dir="./mcd_dataset",
    resume_download=True  # Позволяет продолжить, если интернет прервется
)
print("Скачивание завершено успешно!")
