import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from models.good_model import ImprovedModel
from utils.dataset import TweetDataset, collate_fn
from utils.tokenizer import tokenize
from utils.vocabulary import Vocabulary


def train_with_early_stopping(epochs=30, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary()
    vocab.load("vocab/vocab.json")

    model = ImprovedModel(vocab_size=len(vocab))

    full_dataset = TweetDataset("data/twitter_training.csv", vocab, tokenizer=tokenize)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=True)

    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0

        for inputs, lengths, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_acc = train_correct / total_train
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, lengths, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = val_correct / total_val
        val_losses.append(val_loss)

        tqdm.write(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    train_with_early_stopping()
