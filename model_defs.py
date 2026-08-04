"""Общие определения модели/препроцессинга для train_and_save.py и infer.py."""
import numpy as np
import torch
import torch.nn as nn

N_HEADS = 5


def make_input(profile, x_scale):
    """Два канала: линейный профиль и его стандартизованный логарифм."""
    linear = profile / x_scale
    log = np.log10(np.clip(profile, 1e-7, None))
    log = (log - log.mean()) / (log.std() + 1e-9)
    return np.stack([linear, log]).astype(np.float32)


class MultiHeadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten())
        flat = 128 * 12
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(256, 3)) for _ in range(N_HEADS)])
        self.head_cls = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        h = self.encoder(x)
        reg = torch.stack([head(h) for head in self.heads], dim=1)  # (B, N_HEADS, 3)
        return reg, self.head_cls(h).squeeze(-1)


def make_ranker():
    return nn.Sequential(
        nn.Linear(19, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 1))


def select_diverse(cands_norm, score, k=5):
    """Жадный отбор k разнообразных кандидатов (1-й — лучший по score)."""
    order = np.argsort(-score)
    chosen = [order[0]]
    for _ in range(k - 1):
        best_j, best_gain = None, -np.inf
        for j in order:
            if j in chosen:
                continue
            dist = min(np.linalg.norm(cands_norm[j] - cands_norm[i]) for i in chosen)
            gain = dist + 0.02 * score[j]
            if gain > best_gain:
                best_gain, best_j = gain, j
        chosen.append(best_j)
    return chosen
