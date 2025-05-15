import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.baseline_model import BaselineModel
from utils.vocabulary import Vocabulary
from utils.dataset import TweetDataset
from utils.tokenizer import tokenize
from utils.dataset import collate_fn

from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary()
    vocab.load("vocab/vocab.json")

    full_dataset = TweetDataset("data/twitter_training.csv", vocab, tokenizer=tokenize)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = BaselineModel(vocab_size=len(vocab), embedding_dim=100, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []

    for epoch in range(10):
        # Обучение
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False) as pbar:
            for inputs, lengths, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.view(-1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)

        # Валидация
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1} Val  ", leave=False) as pbar_val:
                for inputs, lengths, labels in pbar_val:
                    inputs = inputs.to(device)
                    labels = labels.view(-1).to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_total_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    pbar_val.set_postfix(loss=loss.item(), acc=val_correct / val_total)

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)

        tqdm.write(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    # plt.show()


if __name__ == "__main__":
    train()
