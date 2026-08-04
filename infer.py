"""Инференс без обучения — использует веса из checkpoints/.

  python3 infer.py                          # метрики на валидации (воспроизводит таблицу)
  python3 infer.py path/to/Absorption.dat   # топ-5 кандидатов для одного профиля
"""
import os
import sys
import numpy as np
import torch

from model_defs import MultiHeadCNN, make_ranker, make_input, select_diverse

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "checkpoints")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

pp = np.load(os.path.join(CKPT, "preproc.npz"))
GRID = pp["grid"]
y_mean, y_std, x_scale = pp["y_mean"], pp["y_std"], float(pp["x_scale"])

models = []
for seed in range(5):
    m = MultiHeadCNN().to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"model_{seed}.pt"),
                                 map_location=DEVICE))
    m.eval()
    models.append(m)
ranker = make_ranker()
ranker.load_state_dict(torch.load(os.path.join(CKPT, "ranker.pt"),
                                  map_location="cpu"))
ranker.eval()


def pca_features(profile):
    log = np.log10(np.clip(profile, 1e-7, None))
    z = (log - pp["pca_mu"]) @ pp["pca_V"].T
    return ((z - pp["pca_zm"]) / pp["pca_zs"]).astype(np.float32)


def predict_profile(profile):
    """profile: (101,) на сетке GRID -> (top5 кандидатов в log10-шкале, P(H2a))."""
    x = torch.tensor(make_input(profile, x_scale)).unsqueeze(0).to(DEVICE)
    cands, probas = [], []
    with torch.no_grad():
        for m in models:
            reg, cls = m(x)
            cands.append(reg[0].cpu().numpy())
            probas.append(torch.sigmoid(cls)[0].item())
    cands = np.concatenate(cands) * y_std + y_mean          # (25, 3), log10
    p_h2a = float(np.mean(probas))
    cn = (cands - y_mean) / y_std
    feat = np.concatenate([np.repeat(pca_features(profile)[None, :], 25, 0), cn], -1)
    with torch.no_grad():
        score = ranker(torch.tensor(feat, dtype=torch.float32)).squeeze(-1).numpy()
    top5 = cands[select_diverse(cn, score)]
    return top5, p_h2a, cands


def load_dat(path):
    arr = np.loadtxt(path, skiprows=1)
    return np.interp(GRID, arr[:, 0], arr[:, 1], left=0.0, right=0.0)


if len(sys.argv) > 1:
    # ---- один профиль ----
    profile = load_dat(sys.argv[1])
    top5, p_h2a, _ = predict_profile(profile)
    print(f"P(H2a=1) = {p_h2a:.3f}  ->  H2a = {int(p_h2a > 0.5)}")
    print("топ-5 кандидатов (по убыванию ранга):")
    print(f"{'#':>2} {'XUVInt':>8} {'Helium':>8} {'Msw':>10}")
    for i, c in enumerate(top5):
        print(f"{i + 1:>2} {10**c[0]:>8.2f} {10**c[1]:>8.4f} {10**c[2]:>10.3e}")
else:
    # ---- валидация: воспроизводим таблицу ----
    from sklearn.metrics import roc_auc_score, accuracy_score
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
    idx_val = pp["idx_val"]
    true_log = np.log10(targets[idx_val, :3])
    true_lin = 10 ** true_log
    y_cls = targets[idx_val, 3]

    top5_all, proba_all, single_all = [], [], []
    for i in idx_val:
        top5, p_h2a, cands = predict_profile(profiles[i])
        top5_all.append(top5)
        proba_all.append(p_h2a)
        single_all.append(cands.mean(0))
    top5_all = np.array(top5_all)
    proba_all = np.array(proba_all)
    single_all = np.array(single_all)

    def mape(t, p):
        return 100 * np.mean(np.abs((p - t) / t))

    def oracle5(j, log_scale=False):
        if log_scale:
            return 100 * np.mean(np.abs(top5_all[:, :, j] - true_log[:, j:j+1]).min(1)
                                 / np.abs(true_log[:, j]))
        return 100 * np.mean(np.abs(10**top5_all[:, :, j] - true_lin[:, j:j+1]).min(1)
                             / true_lin[:, j])

    auc = roc_auc_score(y_cls, proba_all)
    acc = accuracy_score(y_cls, proba_all > 0.5)
    print("параметр   | best-of-5 | один ответ")
    print("-----------+-----------+-----------")
    print(f"XUVInt     | {oracle5(0):7.1f}%  | {mape(true_lin[:,0], 10**single_all[:,0]):.1f}%")
    print(f"Helium     | {oracle5(1):7.1f}%  | {mape(true_lin[:,1], 10**single_all[:,1]):.1f}%")
    print(f"logMsw     | {oracle5(2, True):7.1f}%  | {mape(true_log[:,2], single_all[:,2]):.1f}%")
    print(f"H2a        | ROC-AUC = {auc:.3f}, accuracy = {acc:.3f}")
