"""Простой бейзлайн: ручные признаки профиля + градиентный бустинг.

Таргеты:
  XUVInt  (int)   -> регрессия, MAPE
  Helium  (fp32)  -> регрессия, MAPE
  Msw     (fp32, ~1e11..1e14) -> регрессия log10(Msw), MAPE по log10
  H2a     (bool)  -> классификация, ROC-AUC
"""
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new_dataset_V3")
GRID = np.arange(-5.0, 5.0 + 1e-9, 0.1)  # общая сетка доплеровских скоростей


def load_run(d):
    kv = {}
    with open(os.path.join(d, "parameters.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                kv[parts[0]] = parts[1]
    arr = np.loadtxt(os.path.join(d, "Absorption.dat"), skiprows=1)
    vv, absorb = arr[:, 0], arr[:, 1]  # VV, FullAbs
    if vv[-1] - vv[0] < 5:  # обрезанный/битый профиль
        return None
    prof = np.interp(GRID, vv, absorb, left=0.0, right=0.0)
    return kv, prof


def features(vv, a):
    """Скалярные морфологические признаки профиля."""
    peak = a.max()
    v_peak = vv[a.argmax()]
    ew = np.trapz(a, vv)                       # эквивалентная ширина
    above = a >= peak / 2
    fwhm = vv[above][-1] - vv[above][0] if above.any() else 0.0
    blue = np.trapz(a[vv < 0], vv[vv < 0])
    red = np.trapz(a[vv > 0], vv[vv > 0])
    asym = (blue - red) / (blue + red + 1e-12)  # сине-красная асимметрия
    mean_v = np.trapz(vv * a, vv) / (ew + 1e-12)
    var_v = np.trapz((vv - mean_v) ** 2 * a, vv) / (ew + 1e-12)
    w10 = a >= 0.1 * peak
    width10 = vv[w10][-1] - vv[w10][0] if w10.any() else 0.0
    core_to_wing = a[np.abs(vv) <= 1].sum() / (a[np.abs(vv) > 2].sum() + 1e-12)
    return [peak, v_peak, ew, fwhm, asym, mean_v, np.sqrt(var_v), width10, core_to_wing]


X, y_xuv, y_he, y_logmsw, y_h2a = [], [], [], [], []
skipped = 0
for name in sorted(os.listdir(DATA)):
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        continue
    out = load_run(d)
    if out is None:
        skipped += 1
        continue
    kv, prof = out
    X.append(features(GRID, prof))
    y_xuv.append(float(kv["XUVInt"]))
    y_he.append(float(kv["Helium"]))
    y_logmsw.append(np.log10(float(kv["Msw"])))
    y_h2a.append(int(kv["H2a"]))

X = np.array(X)
print(f"loaded {len(X)} runs, skipped {skipped}")

idx = np.arange(len(X))
tr, va = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_h2a)


def mape(t, p):
    return 100 * np.mean(np.abs((p - t) / t))


for label, y in [("XUVInt", y_xuv), ("Helium", y_he), ("logMsw", y_logmsw)]:
    y = np.array(y)
    m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
    print(f"{label:8s} MAPE = {mape(y[va], m.predict(X[va])):6.1f}%")

y = np.array(y_h2a)
clf = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
proba = clf.predict_proba(X[va])[:, 1]
print(f"H2a      ROC-AUC = {roc_auc_score(y[va], proba):.3f}, "
      f"acc = {accuracy_score(y[va], proba > 0.5):.3f}")
