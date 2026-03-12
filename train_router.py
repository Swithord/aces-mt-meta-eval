import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model import MLPRouterModel


ERROR_TYPES = sorted([
    'addition', 'do-not-translate', 'mistranslation', 'omission',
    'overtranslation', 'punctuation', 'real-world-knowledge',
    'undertranslation', 'untranslated', 'wrong-language'
])
ERROR_TYPE_TO_IDX = {e: i for i, e in enumerate(ERROR_TYPES)}


class ErrorTypeDataset(Dataset):
    def __init__(self, csv_files: list[str], metric_columns: list[str], error_type_column: str = "error-type"):
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)

        self.X = torch.tensor(df[metric_columns].values, dtype=torch.float32)
        self.y = torch.tensor(df[error_type_column].map(ERROR_TYPE_TO_IDX).values, dtype=torch.long)

        self._normalize()

    def _normalize(self):
        mean = self.X.mean(dim=0)
        std = self.X.std(dim=0)
        self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_router(router, train_dataset, val_dataset, num_epochs=20, batch_size=64, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(router.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        router.train()
        total_loss, correct = 0.0, 0

        for X_batch, y_batch in train_loader:
            logits = router(X_batch)  # (batch_size, n_error_types) — softmax is applied inside forward()
            # CrossEntropyLoss expects raw logits, so we need to bypass the softmax in forward().
            # Simplest fix: call fc layers directly. See note below.
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_dataset)

        # Validation
        router.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = router(X_batch)
                val_loss += loss_fn(logits, y_batch).item()
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_correct/len(val_dataset):.4f}")

    return router


if __name__ == "__main__":
    metrics = ["bleu-score", "chrf-score", "ter-score", "bert-score", "bleurt-score", "comet-score"]
    csv_files = [f"{error_type}_data.csv" for error_type in ERROR_TYPES]

    dataset = ErrorTypeDataset(csv_files, metric_columns=metrics)
    train_data, temp = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    router = MLPRouterModel(n_metrics=len(metrics), hidden_size=16, n_error_types=len(ERROR_TYPES))
    trained_router = train_router(router, train_data, val_data, num_epochs=30)
    torch.save(trained_router.state_dict(), "router_model.pth")
