import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.tokenizer import tokenize


class TweetDataset(Dataset):
    def __init__(self, csv_path, vocab, tokenizer=tokenize):
        df = pd.read_csv(csv_path, header=None, names=['ID', 'entity', 'sentiment', 'content'])
        df['content'] = df['content'].fillna('')
        self.tokenized_texts = [tokenizer(text) for text in df['content']]

        self.texts_indices = [
            [vocab.token_to_idx(token) for token in tokens]
            for tokens in self.tokenized_texts
        ]

        label_map = {'Positive': 1, 'Negative': 0}
        self.labels = [label_map.get(sentiment, 0) for sentiment in df['sentiment']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts_indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    pad_idx = 0

    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    attention_masks = []

    for seq in sequences:
        padding_length = max_len - seq.size(0)
        padded_seq = torch.cat([seq, torch.full((padding_length,), pad_idx, dtype=torch.long)])
        padded_seqs.append(padded_seq)

        mask = torch.cat([
            torch.ones(seq.size(0), dtype=torch.bool),
            torch.zeros(padding_length, dtype=torch.bool)
        ])
        attention_masks.append(mask)

    padded_seqs = torch.stack(padded_seqs)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.long)  # ✅ фикс ошибки

    return padded_seqs, attention_masks, labels

