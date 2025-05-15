import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from utils.tokenizer import tokenize

def main():
    # Загружаем датасет
    df = pd.read_csv("../data/twitter_training.csv", header=None,
                     names=["ID", "entity", "sentiment", "content"])

    tweets = df["content"].astype(str).tolist()

    # Токенизируем
    tokenized_tweets = [tokenize(tweet) for tweet in tweets]

    # Собираем частоты токенов
    all_tokens = [token for tokens in tokenized_tweets for token in tokens]
    freqs = Counter(all_tokens)

    # Топ-20 токенов
    top_20 = freqs.most_common(20)
    print("Топ-20 токенов по частоте:")
    for token, freq in top_20:
        print(f"{token}: {freq}")

    # Уникальные токены
    print(f"\nВсего уникальных токенов: {len(freqs)}")

    # Длины твитов в токенах
    lengths = [len(tokens) for tokens in tokenized_tweets]
    print(f"\nСредняя длина твита: {sum(lengths)/len(lengths):.2f}")
    print(f"Максимальная длина твита: {max(lengths)}")
    print(f"Медианная длина твита: {sorted(lengths)[len(lengths)//2]}")

    # Гистограмма распределения частот токенов
    plt.figure(figsize=(10,5))
    plt.hist(list(freqs.values()), bins=50, log=True)
    plt.title("Распределение частот токенов (логарифмический масштаб)")
    plt.xlabel("Частота токена")
    plt.ylabel("Количество токенов")
    plt.show()

if __name__ == "__main__":
    main()
