"""Обучает пайплайн dl_pro (5 WTA-моделей + ранкер) и сохраняет всё
необходимое для инференса в checkpoints/:
  model_{0..4}.pt  — веса CNN
  ranker.pt        — веса ранкера
  preproc.npz      — сетка, нормировки, PCA, индексы сплита
После этого метрики и предсказания можно получать без обучения: python3 infer.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from model_defs import MultiHeadCNN, make_ranker, make_input, N_HEADS

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
GRID = np.arange(-5.0, 5.0 + 1e-9, 0.1)
print("device:", DEVICE)

# ---------------- данные ----------------
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

# PCA-16 лог-профилей (по train)
log_prof = np.log10(np.clip(profiles, 1e-7, None))
pca_mu = log_prof[idx_train].mean(axis=0)
_, S, Vt = np.linalg.svd(log_prof[idx_train] - pca_mu, full_matrices=False)
pca_V = Vt[:16]
Z = (log_prof - pca_mu) @ pca_V.T
pca_zm, pca_zs = Z[idx_train].mean(0), Z[idx_train].std(0)
Z = ((Z - pca_zm) / pca_zs).astype(np.float32)


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
        return (torch.tensor(make_input(profile, x_scale)),
                torch.tensor((y_reg[idx] - y_mean) / y_std, dtype=torch.float32),
                torch.tensor(y_cls[idx]))


val_loader = DataLoader(ProfileDataset(idx_val), batch_size=256)
bce = nn.BCEWithLogitsLoss()


def wta_loss(pred_heads, y_true):
    per_head = ((pred_heads - y_true.unsqueeze(1)) ** 2).mean(-1)
    return per_head.min(dim=1).values.mean() + 0.1 * per_head.mean()


# ---------------- обучение 5 моделей ----------------
EPOCHS = 400
for seed in range(5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(ProfileDataset(idx_train, augment=True),
                              batch_size=64, shuffle=True)
    model = MultiHeadCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        for x, y, c in train_loader:
            x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            reg, cls = model(x)
            loss = wta_loss(reg, y) + bce(cls, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, y, c in val_loader:
                x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
                reg, cls = model(x)
                v += (wta_loss(reg, y) + bce(cls, c)).item() * len(x)
        v /= len(idx_val)
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
    torch.save(best_state, os.path.join(CKPT, f"model_{seed}.pt"))
    print(f"модель {seed}: val loss {best_val:.4f} -> checkpoints/model_{seed}.pt",
          flush=True)

# ---------------- обучение ранкера ----------------
torch.manual_seed(0)
np.random.seed(0)
Ztr, yn = Z[idx_train], ((y_reg - y_mean) / y_std)
yn_tr = yn[idx_train]
D_prof = cdist(Ztr, Ztr)
D_par = cdist(yn_tr, yn_tr)
pair_x, pair_y = [], []
for a in range(len(idx_train)):
    pair_x.append(np.concatenate([Ztr[a], yn_tr[a]])); pair_y.append(1.0)
    for _ in range(2):
        pair_x.append(np.concatenate([Ztr[a], yn_tr[a] + np.random.normal(0, 0.05, 3)]))
        pair_y.append(1.0)
    for _ in range(3):
        b = np.random.randint(len(idx_train))
        if D_par[a, b] > 0.5:
            pair_x.append(np.concatenate([Ztr[a], yn_tr[b]])); pair_y.append(0.0)
    close = np.argsort(D_prof[a])[1:15]
    for b in [b for b in close if D_par[a, b] > 0.8][:3]:
        pair_x.append(np.concatenate([Ztr[a], yn_tr[b]])); pair_y.append(0.0)
pair_x = torch.tensor(np.array(pair_x), dtype=torch.float32)
pair_y = torch.tensor(np.array(pair_y), dtype=torch.float32)
ranker = make_ranker()
opt = torch.optim.Adam(ranker.parameters(), lr=1e-3, weight_decay=1e-5)
for step in range(3000):
    idx = torch.randperm(len(pair_x))[:256]
    loss = bce(ranker(pair_x[idx]).squeeze(-1), pair_y[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.save(ranker.state_dict(), os.path.join(CKPT, "ranker.pt"))
print("ранкер -> checkpoints/ranker.pt")

# ---------------- препроцессинг ----------------
np.savez(os.path.join(CKPT, "preproc.npz"),
         grid=GRID, y_mean=y_mean, y_std=y_std, x_scale=x_scale,
         pca_mu=pca_mu, pca_V=pca_V, pca_zm=pca_zm, pca_zs=pca_zs,
         idx_train=idx_train, idx_val=idx_val)
print("препроцессинг -> checkpoints/preproc.npz")
print("\nГотово. Инференс: python3 infer.py [путь к Absorption.dat]")
