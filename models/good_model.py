import torch
import torch.nn as nn
import torch.optim as optim


class ImprovedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.output = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # усредняем по длине последовательности
        embedded = embedded.mean(dim=1)  # (batch_size, embedding_dim)

        x = self.fc1(embedded)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        out = self.output(x)
        return out
