import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        logits = self.fc(x)  # [batch_size, num_classes]
        return logits





