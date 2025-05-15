import json
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq

        # Спецтокены с фиксированными индексами
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.URL = "<URL>"
        self.MENTION = "<MENTION>"
        self.HASHTAG = "<HASHTAG>"
        self.PIC = "<PIC>"

        self.special_tokens = [self.PAD, self.UNK, self.URL, self.MENTION, self.HASHTAG, self.PIC]

        self.token2idx = {}
        self.idx2token = []
        self.freqs = Counter()

        # Инициализируем словарь спецтокенами
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1

    def build_vocab(self, token_lists):
        """
        token_lists — список списков токенов (например, токенизированные твиты)
        """
        # Собираем частоты по всему корпусу
        for tokens in token_lists:
            self.freqs.update(tokens)

        # Фильтруем по min_freq и пропускаем спецтокены (они уже есть)
        tokens = [token for token, freq in self.freqs.items()
                  if freq >= self.min_freq and token not in self.special_tokens]

        # Сортируем по убыванию частоты
        tokens.sort(key=lambda t: self.freqs[t], reverse=True)

        for token in tokens:
            self._add_token(token)

    def token_to_idx(self, token):
        return self.token2idx.get(token, self.token2idx[self.UNK])

    def idx_to_token(self, idx):
        if 0 <= idx < len(self.idx2token):
            return self.idx2token[idx]
        else:
            return self.UNK

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.token2idx = json.load(f)
        # Восстанавливаем idx2token из token2idx
        self.idx2token = [None] * len(self.token2idx)
        for token, idx in self.token2idx.items():
            self.idx2token[idx] = token

    def __len__(self):
        return len(self.idx2token)



