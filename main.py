import os

project_files = [
    "data/raw_tweets.csv",
    "data/preprocessed.pkl",
    "vocab/vocab.json",
    "models/baseline_model.py",
    "utils/tokenizer.py",
    "utils/dataset.py",
    "utils/vocabulary.py",
    "utils/trainer.py",
    "train.py",
    "evaluate.py",
    "config.py"
]

base_dir = "sentiment_classifier"

for relative_path in project_files:
    file_path = os.path.join(base_dir, relative_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")  # создаём пустой файл

print(f"✅ Проект '{base_dir}' создан со всеми файлами.")


