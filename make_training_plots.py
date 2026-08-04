"""Переобучает 5 WTA-моделей из dl_pro.ipynb с записью истории лоссов
и сохраняет графики обучения в папку training_plots/."""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "training_plots")
os.makedirs(OUT, exist_ok=True)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
GRID = np.arange(-5.0, 5.0 + 1e-9, 0.1)

# ---------- данные (идентично dl_pro.ipynb) ----------
profiles, targets = [], []
for name in sorted(os.listdir(os.path.join(BASE, "..", "new_dataset_V3"))):
    path = os.path.join(BASE, "..", "new_dataset_V3", name)
    if not os.path.isdir(path):
        continue
    params = {}
    with open(os.path.join(path, "parameters.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                params[parts[0]] = parts[1]
    arr = np.loadtxt(os.path.join(path, "Absorption.dat"), skiprows=1)
    if arr[-1, 0] - arr[0, 0] < 5:
        continue
    profiles.append(np.interp(GRID, arr[:, 0], arr[:, 1], left=0.0, right=0.0))
    targets.append([float(params["XUVInt"]), float(params["Helium"]),
                    float(params["Msw"]), int(params["H2a"])])
profiles = np.array(profiles, dtype=np.float32)
targets = np.array(targets)
y_reg = np.log10(targets[:, :3])
y_cls = targets[:, 3].astype(np.float32)
idx_train, idx_val = train_test_split(
    np.arange(len(profiles)), test_size=0.2, random_state=42, stratify=y_cls)
y_mean = y_reg[idx_train].mean(axis=0)
y_std = y_reg[idx_train].std(axis=0)
x_scale = profiles[idx_train].max()


def make_input(profile):
    linear = profile / x_scale
    log = np.log10(np.clip(profile, 1e-7, None))
    log = (log - log.mean()) / (log.std() + 1e-9)
    return np.stack([linear, log]).astype(np.float32)


class ProfileDataset(Dataset):
    def __init__(self, indices, augment=False):
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        profile = profiles[idx]
        if self.augment:
            profile = profile + np.random.normal(0, 0.005 * profile.max(),
                                                 size=profile.shape)
            profile = np.clip(profile, 1e-7, None)
        return (torch.tensor(make_input(profile)),
                torch.tensor((y_reg[idx] - y_mean) / y_std, dtype=torch.float32),
                torch.tensor(y_cls[idx]))


val_loader = DataLoader(ProfileDataset(idx_val), batch_size=256)

# ---------- модель (идентично dl_pro.ipynb) ----------
N_HEADS = 5


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
        reg = torch.stack([head(h) for head in self.heads], dim=1)
        return reg, self.head_cls(h).squeeze(-1)


def wta_parts(pred_heads, y_true):
    per_head = ((pred_heads - y_true.unsqueeze(1)) ** 2).mean(-1)
    return per_head.min(dim=1).values.mean() + 0.1 * per_head.mean()


EPOCHS = 400
bce = nn.BCEWithLogitsLoss()
histories = []
t_start = time.time()

for seed in range(5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(ProfileDataset(idx_train, augment=True),
                              batch_size=64, shuffle=True)
    model = MultiHeadCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = {"train": [], "val": [], "val_reg": [], "val_cls": []}
    best_val, best_ep = float("inf"), -1

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for x, y, c in train_loader:
            x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            reg, cls = model(x)
            loss = wta_parts(reg, y) + bce(cls, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        sched.step()

        model.eval()
        v_reg = v_cls = 0.0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
                reg, cls = model(x)
                v_reg += wta_parts(reg, y).item() * len(x)
                v_cls += bce(cls, c).item() * len(x)
        tr_loss /= len(idx_train)
        v_reg /= len(idx_val)
        v_cls /= len(idx_val)
        v_tot = v_reg + v_cls
        hist["train"].append(tr_loss)
        hist["val"].append(v_tot)
        hist["val_reg"].append(v_reg)
        hist["val_cls"].append(v_cls)
        if v_tot < best_val:
            best_val, best_ep = v_tot, epoch

    histories.append((hist, best_ep))
    print(f"модель {seed}: лучший val loss {best_val:.4f} на эпохе {best_ep + 1}, "
          f"прошло {time.time() - t_start:.0f} с", flush=True)

    # индивидуальный график
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["train"], label="train (WTA + BCE)", lw=1.2)
    ax.plot(hist["val"], label="val (WTA + BCE)", lw=1.2)
    ax.plot(hist["val_reg"], label="val регрессия (WTA)", lw=0.9, alpha=0.7)
    ax.plot(hist["val_cls"], label="val классификация (BCE)", lw=0.9, alpha=0.7)
    ax.axvline(best_ep, color="k", ls="--", lw=0.8,
               label=f"лучший чекпоинт (эп. {best_ep + 1})")
    ax.set_yscale("log")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    ax.set_title(f"Модель {seed} (seed={seed})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"loss_model_{seed}.png"), dpi=150)
    plt.close(fig)

# сводный график
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for seed, (hist, best_ep) in enumerate(histories):
    axes[0].plot(hist["train"], lw=1, label=f"seed {seed}")
    axes[1].plot(hist["val"], lw=1, label=f"seed {seed}")
    axes[1].scatter([best_ep], [hist["val"][best_ep]], s=25, zorder=5)
for ax, title in zip(axes, ["train loss", "val loss (точка — лучший чекпоинт)"]):
    ax.set_yscale("log")
    ax.set_xlabel("эпоха")
    ax.set_title(title)
    ax.legend(fontsize=8)
axes[0].set_ylabel("loss (WTA + BCE)")
fig.suptitle("Обучение ансамбля из 5 WTA-моделей (dl_pro)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_all_models.png"), dpi=150)
plt.close(fig)

np.savez(os.path.join(OUT, "histories.npz"),
         **{f"model{sd}_{k}": np.array(h[k])
            for sd, (h, _) in enumerate(histories) for k in h})
print("готово, графики в", OUT)
