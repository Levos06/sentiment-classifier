import os
import pandas as pd
from utils.tokenizer import tokenize
from utils.vocabulary import Vocabulary


def main():
    # Путь к исходному csv с твитами
    csv_path = "data/twitter_training.csv"

    # Загружаем данные
    df = pd.read_csv(csv_path, header=None, names=['ID', 'entity', 'sentiment', 'content'])

    # Токенизируем тексты
    tokenized_texts = [tokenize(str(text)) for text in df['content'].fillna('')]

    # Создаем словарь с порогом по частоте, например min_freq=5
    vocab = Vocabulary(min_freq=5)
    vocab.build_vocab(tokenized_texts)

    # Создаем папку vocab, если ее нет
    os.makedirs("vocab", exist_ok=True)

    # Сохраняем словарь в vocab.json
    vocab.save("vocab/vocab.json")

    print(f"Vocabulary built and saved. Size: {len(vocab.idx2token)} tokens")

if __name__ == "__main__":
    main()
