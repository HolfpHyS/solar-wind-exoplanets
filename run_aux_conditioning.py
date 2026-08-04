"""Reproducible, train-only runner for the physical-conditioning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from planet_pseudolabels import PLANET_NAMES, make_planet_labels


GRID = np.arange(-5.0, 5.0 + 1e-9, 0.1)
PHYS_COLUMNS = [
    "pl_bmassj",
    "pl_radj",
    "pl_orbsmax",
    "pl_orbper",
    "pl_eqt",
    "pl_dens",
    "st_teff",
    "st_rad",
    "st_mass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "participant" / "new_dataset_V3",
    )
    parser.add_argument(
        "--planet-params",
        type=Path,
        default=Path(__file__).resolve().parent / "aux_data" / "planet_params.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "aux_conditioning_train_only",
    )
    parser.add_argument(
        "--planet-label-mode",
        choices=("train-only",),
        default="train-only",
        help="Kept explicit in run manifests; leaky compatibility mode is not supported.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--model-seeds", default="0,1,2,3,4")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ranker-steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")

    profiles: list[np.ndarray] = []
    targets: list[list[float]] = []
    planet_names: list[str] = []
    object_ids: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    for run_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        params_path = run_dir / "parameters.txt"
        absorption_path = run_dir / "Absorption.dat"
        params: dict[str, str] = {}
        with params_path.open() as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    params[parts[0]] = parts[1]
        arr = np.loadtxt(absorption_path, skiprows=1)
        if arr[-1, 0] - arr[0, 0] < 5:
            continue

        profiles.append(np.interp(GRID, arr[:, 0], arr[:, 1], left=0.0, right=0.0))
        targets.append([
            float(params["XUVInt"]),
            float(params["Helium"]),
            float(params["Msw"]),
            int(params["H2a"]),
        ])
        planet_names.append(params.get("PName", "<нет>"))
        object_ids.append(run_dir.name)
        manifest_rows.append({
            "object_id": run_dir.name,
            "parameters_sha256": sha256_file(params_path),
            "absorption_sha256": sha256_file(absorption_path),
            "planet_name_raw": params.get("PName", "<нет>"),
        })

    return (
        np.asarray(profiles, dtype=np.float32),
        np.asarray(targets),
        np.asarray(planet_names),
        np.asarray(object_ids),
        pd.DataFrame(manifest_rows),
    )


def make_input(profile: np.ndarray, x_scale: float) -> np.ndarray:
    linear = profile / x_scale
    log_profile = np.log10(np.clip(profile, 1e-7, None))
    log_profile = (log_profile - log_profile.mean()) / (log_profile.std() + 1e-9)
    return np.stack([linear, log_profile]).astype(np.float32)


class ProfileDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        profiles: np.ndarray,
        y_normalized: np.ndarray,
        y_cls: np.ndarray,
        planet_indices: np.ndarray,
        x_scale: float,
        *,
        augment: bool,
    ) -> None:
        self.indices = np.asarray(indices, dtype=int)
        self.profiles = profiles
        self.y_normalized = y_normalized
        self.y_cls = y_cls
        self.planet_indices = planet_indices
        self.x_scale = x_scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        profile = self.profiles[index]
        if self.augment:
            profile = np.clip(
                profile + np.random.normal(0, 0.005 * profile.max(), profile.shape),
                1e-7,
                None,
            )
        return (
            torch.tensor(make_input(profile, self.x_scale)),
            torch.tensor(self.y_normalized[index], dtype=torch.float32),
            torch.tensor(self.y_cls[index], dtype=torch.float32),
            torch.tensor(self.planet_indices[index], dtype=torch.long),
        )


class ConditionedCNN(nn.Module):
    def __init__(self, physical_table: torch.Tensor, n_heads: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        flat = 128 * 12
        self.head_planet = nn.Sequential(
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat + 9, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 3),
            )
            for _ in range(n_heads)
        ])
        self.head_cls = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.register_buffer("physical_table", physical_table.detach().clone())

    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)
        planet_logits = self.head_planet(embedding)
        physical = torch.softmax(planet_logits, dim=-1) @ self.physical_table
        conditioned = torch.cat([embedding, physical], dim=-1)
        regression = torch.stack([head(conditioned) for head in self.heads], dim=1)
        return regression, self.head_cls(embedding).squeeze(-1), planet_logits


def build_physical_table(csv_path: Path) -> np.ndarray:
    catalog = pd.read_csv(csv_path)
    catalog["pl_name"] = (
        catalog["pl_name"].astype(str).str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
    )
    rows = []
    for name in PLANET_NAMES:
        match = catalog.loc[catalog["pl_name"] == name, PHYS_COLUMNS]
        if len(match) != 1:
            raise ValueError(f"expected exactly one catalog row for {name}, found {len(match)}")
        values = match.iloc[0].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"invalid physical parameters for {name}: {values}")
        rows.append(np.log10(values))
    physical = np.stack(rows)
    return ((physical - physical.mean(axis=0)) / (physical.std(axis=0) + 1e-9)).astype(np.float32)


def wta_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per_head = ((prediction - target.unsqueeze(1)) ** 2).mean(dim=-1)
    return per_head.min(dim=1).values.mean() + 0.1 * per_head.mean()


def evaluate_regression(candidates_log: np.ndarray, truth_log: np.ndarray) -> dict[str, float]:
    prediction_linear = 10 ** candidates_log
    truth_linear = 10 ** truth_log
    result = {
        "xuv_mape_pct": float(
            100
            * np.mean(
                np.abs(prediction_linear[:, :, 0] - truth_linear[:, None, 0]).min(axis=1)
                / truth_linear[:, 0]
            )
        ),
        "helium_mape_pct": float(
            100
            * np.mean(
                np.abs(prediction_linear[:, :, 1] - truth_linear[:, None, 1]).min(axis=1)
                / truth_linear[:, 1]
            )
        ),
        "log_msw_mape_pct": float(
            100
            * np.mean(
                np.abs(candidates_log[:, :, 2] - truth_log[:, None, 2]).min(axis=1)
                / np.abs(truth_log[:, 2])
            )
        ),
        "log_msw_mae_dex": float(
            np.mean(np.abs(candidates_log[:, :, 2] - truth_log[:, None, 2]).min(axis=1))
        ),
        "linear_msw_mape_pct": float(
            100
            * np.mean(
                np.abs(prediction_linear[:, :, 2] - truth_linear[:, None, 2]).min(axis=1)
                / truth_linear[:, 2]
            )
        ),
    }
    return result


def make_ranker() -> nn.Module:
    return nn.Sequential(
        nn.Linear(19, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )


def select_diverse(candidates_normalized: np.ndarray, scores: np.ndarray, k: int = 5) -> list[int]:
    order = np.argsort(-scores)
    chosen = [int(order[0])]
    for _ in range(k - 1):
        best_index: int | None = None
        best_gain = -np.inf
        for candidate_index in order:
            candidate_index = int(candidate_index)
            if candidate_index in chosen:
                continue
            distance = min(
                np.linalg.norm(candidates_normalized[candidate_index] - candidates_normalized[index])
                for index in chosen
            )
            gain = distance + 0.02 * scores[candidate_index]
            if gain > best_gain:
                best_gain = gain
                best_index = candidate_index
        if best_index is None:
            raise RuntimeError("unable to select diverse candidate")
        chosen.append(best_index)
    return chosen


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_artifact_manifest(
    run_dir: Path,
    planet_params: Path,
    *,
    source_status: str,
) -> None:
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("planet_pseudolabels.py"),
        Path(__file__).resolve().with_name("verify_aux_conditioning_run.py"),
        planet_params.resolve(),
    ]
    artifacts = []
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        run_dir / "artifact_manifest.json",
        {
            "source_status": source_status,
            "source_files": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in source_paths
                if path.exists()
            ],
            "artifacts": artifacts,
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.model_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one model seed is required")
    if args.epochs <= 0 or args.ranker_steps <= 0:
        raise ValueError("epochs and ranker-steps must be positive")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass
    device = resolve_device(args.device)

    print(f"device={device} threads={torch.get_num_threads()}", flush=True)
    profiles, targets, raw_planets, object_ids, manifest = load_dataset(args.data_dir.resolve())
    y_reg = np.log10(targets[:, :3])
    y_cls = targets[:, 3].astype(np.float32)
    idx_train, idx_val = train_test_split(
        np.arange(len(profiles)),
        test_size=0.2,
        random_state=args.split_seed,
        stratify=y_cls,
    )
    train_mask = np.zeros(len(profiles), dtype=bool)
    train_mask[idx_train] = True
    manifest["split"] = np.where(train_mask, "train", "validation")

    y_mean = y_reg[idx_train].mean(axis=0)
    y_std = y_reg[idx_train].std(axis=0)
    y_normalized = (y_reg - y_mean) / y_std
    x_scale = float(profiles[idx_train].max())

    log_profiles = np.log10(np.clip(profiles, 1e-7, None))
    pca_mean = log_profiles[idx_train].mean(axis=0)
    _, _, pca_vectors_all = np.linalg.svd(
        log_profiles[idx_train] - pca_mean,
        full_matrices=False,
    )
    pca_vectors = pca_vectors_all[:16]
    z_raw = (log_profiles - pca_mean) @ pca_vectors.T
    pca_z_mean = z_raw[idx_train].mean(axis=0)
    pca_z_std = z_raw[idx_train].std(axis=0)
    z = ((z_raw - pca_z_mean) / pca_z_std).astype(np.float32)

    labels = make_planet_labels(
        z,
        raw_planets,
        idx_train,
    )

    manifest["planet_label_used"] = labels.labels
    manifest["planet_is_known"] = labels.known_mask
    manifest["planet_classifier_fit"] = labels.fit_mask
    manifest.to_csv(args.output_dir / "dataset_manifest.csv", index=False)
    (args.output_dir / "split_train_ids.txt").write_text("\n".join(object_ids[idx_train]) + "\n")
    (args.output_dir / "split_val_ids.txt").write_text("\n".join(object_ids[idx_val]) + "\n")

    unknown_train = (~labels.known_mask) & train_mask
    unknown_val = (~labels.known_mask) & (~train_mask)
    pseudo_summary = {
        "mode": args.planet_label_mode,
        "known_total": int(labels.known_mask.sum()),
        "known_fit": int(labels.fit_mask.sum()),
        "known_fit_train": int((labels.fit_mask & train_mask).sum()),
        "known_fit_validation": int((labels.fit_mask & ~train_mask).sum()),
        "unknown_total": int((~labels.known_mask).sum()),
        "unknown_train": int(unknown_train.sum()),
        "unknown_validation": int(unknown_val.sum()),
        "train_pseudolabel_counts": dict(Counter(labels.labels[unknown_train])),
        "validation_pseudolabel_counts": dict(Counter(labels.labels[unknown_val])),
    }
    write_json(args.output_dir / "planet_pseudolabel_summary.json", pseudo_summary)
    print(json.dumps(pseudo_summary, ensure_ascii=False), flush=True)

    physical_table_np = build_physical_table(args.planet_params.resolve())
    physical_table = torch.tensor(physical_table_np, dtype=torch.float32, device=device)

    val_dataset = ProfileDataset(
        idx_val,
        profiles,
        y_normalized,
        y_cls,
        labels.indices,
        x_scale,
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    per_seed_metrics: list[dict[str, object]] = []
    validation_candidates: list[np.ndarray] = []
    validation_probabilities: list[np.ndarray] = []
    validation_planet_predictions: list[np.ndarray] = []
    best_epochs: list[int] = []
    best_losses: list[float] = []

    run_start = time.perf_counter()
    for seed in seeds:
        model_start = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_dataset = ProfileDataset(
            idx_train,
            profiles,
            y_normalized,
            y_cls,
            labels.indices,
            x_scale,
            augment=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        model = ConditionedCNN(physical_table).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(args.epochs):
            model.train()
            for batch_x, batch_y, batch_class, batch_planet in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_class = batch_class.to(device)
                batch_planet = batch_planet.to(device)
                pred_reg, pred_class, pred_planet = model(batch_x)
                loss = (
                    wta_loss(pred_reg, batch_y)
                    + bce(pred_class, batch_class)
                    + 0.3 * ce(pred_planet, batch_planet)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, batch_class, batch_planet in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_class = batch_class.to(device)
                    batch_planet = batch_planet.to(device)
                    pred_reg, pred_class, pred_planet = model(batch_x)
                    batch_loss = (
                        wta_loss(pred_reg, batch_y)
                        + bce(pred_class, batch_class)
                        + 0.3 * ce(pred_planet, batch_planet)
                    )
                    validation_loss += batch_loss.item() * len(batch_x)
            validation_loss /= len(idx_val)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == args.epochs:
                print(
                    f"seed={seed} epoch={epoch + 1}/{args.epochs} "
                    f"val_loss={validation_loss:.6f} best={best_loss:.6f}@{best_epoch + 1}",
                    flush=True,
                )

        if best_state is None:
            raise RuntimeError(f"no checkpoint captured for seed {seed}")
        model.load_state_dict(best_state)
        model.eval()
        pred_reg_parts: list[np.ndarray] = []
        pred_class_parts: list[np.ndarray] = []
        pred_planet_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _, _, _ in val_loader:
                pred_reg, pred_class, pred_planet = model(batch_x.to(device))
                pred_reg_parts.append(pred_reg.cpu().numpy())
                pred_class_parts.append(torch.sigmoid(pred_class).cpu().numpy())
                pred_planet_parts.append(pred_planet.argmax(dim=-1).cpu().numpy())

        pred_reg_log = np.concatenate(pred_reg_parts) * y_std + y_mean
        pred_class_probability = np.concatenate(pred_class_parts)
        pred_planet_index = np.concatenate(pred_planet_parts)
        validation_candidates.append(pred_reg_log)
        validation_probabilities.append(pred_class_probability)
        validation_planet_predictions.append(pred_planet_index)
        best_epochs.append(best_epoch + 1)
        best_losses.append(float(best_loss))

        known_val_mask = labels.known_mask[idx_val]
        seed_result = {
            "seed": seed,
            "best_epoch": best_epoch + 1,
            "best_validation_loss": float(best_loss),
            "oracle_of_5": evaluate_regression(pred_reg_log, y_reg[idx_val]),
            "h2a_roc_auc": float(roc_auc_score(y_cls[idx_val], pred_class_probability)),
            "h2a_accuracy": float(
                accuracy_score(y_cls[idx_val], pred_class_probability >= 0.5)
            ),
            "planet_accuracy_known_validation": float(
                accuracy_score(
                    labels.indices[idx_val][known_val_mask],
                    pred_planet_index[known_val_mask],
                )
            ),
            "elapsed_seconds": time.perf_counter() - model_start,
        }
        per_seed_metrics.append(seed_result)
        print(json.dumps(seed_result, ensure_ascii=False), flush=True)

        if not args.no_save_models:
            torch.save(
                {
                    "state_dict": best_state,
                    "seed": seed,
                    "best_epoch": best_epoch + 1,
                    "best_validation_loss": float(best_loss),
                    "planet_label_mode": args.planet_label_mode,
                },
                args.output_dir / f"model_{seed}.pt",
            )

    all_candidates = np.concatenate(
        [candidates[:, None, :, :] for candidates in validation_candidates],
        axis=1,
    ).reshape(len(idx_val), -1, 3)
    ensemble_probability = np.mean(validation_probabilities, axis=0)
    candidates_normalized = (all_candidates - y_mean) / y_std

    torch.manual_seed(0)
    np.random.seed(0)
    z_train = z[idx_train]
    y_train_normalized = y_normalized[idx_train]
    profile_distance = cdist(z_train, z_train)
    parameter_distance = cdist(y_train_normalized, y_train_normalized)
    pair_x: list[np.ndarray] = []
    pair_y: list[float] = []
    for anchor in range(len(idx_train)):
        pair_x.append(np.concatenate([z_train[anchor], y_train_normalized[anchor]]))
        pair_y.append(1.0)
        for _ in range(2):
            pair_x.append(
                np.concatenate([
                    z_train[anchor],
                    y_train_normalized[anchor] + np.random.normal(0, 0.05, 3),
                ])
            )
            pair_y.append(1.0)
        for _ in range(3):
            other = np.random.randint(len(idx_train))
            if parameter_distance[anchor, other] > 0.5:
                pair_x.append(np.concatenate([z_train[anchor], y_train_normalized[other]]))
                pair_y.append(0.0)
        close = np.argsort(profile_distance[anchor])[1:15]
        hard = [other for other in close if parameter_distance[anchor, other] > 0.8][:3]
        for other in hard:
            pair_x.append(np.concatenate([z_train[anchor], y_train_normalized[other]]))
            pair_y.append(0.0)

    pair_x_tensor = torch.tensor(np.asarray(pair_x), dtype=torch.float32)
    pair_y_tensor = torch.tensor(np.asarray(pair_y), dtype=torch.float32)
    ranker = make_ranker()
    ranker_optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-3, weight_decay=1e-5)
    for step in range(args.ranker_steps):
        batch_indices = torch.randperm(len(pair_x_tensor))[:256]
        ranker_loss = bce(
            ranker(pair_x_tensor[batch_indices]).squeeze(dim=-1),
            pair_y_tensor[batch_indices],
        )
        ranker_optimizer.zero_grad()
        ranker_loss.backward()
        ranker_optimizer.step()
        if step == 0 or (step + 1) % 500 == 0 or step + 1 == args.ranker_steps:
            print(
                f"ranker step={step + 1}/{args.ranker_steps} loss={ranker_loss.item():.6f}",
                flush=True,
            )
    ranker.eval()
    ranker_features = np.concatenate(
        [
            np.repeat(z[idx_val][:, None, :], all_candidates.shape[1], axis=1),
            candidates_normalized,
        ],
        axis=-1,
    )
    with torch.no_grad():
        scores = (
            ranker(torch.tensor(ranker_features, dtype=torch.float32).reshape(-1, 19))
            .reshape(len(idx_val), all_candidates.shape[1])
            .numpy()
        )

    selected_indices = np.stack([
        select_diverse(candidates_normalized[row], scores[row])
        for row in range(len(idx_val))
    ])
    selected_candidates = np.take_along_axis(
        all_candidates,
        selected_indices[:, :, None],
        axis=1,
    )
    selected_candidates_normalized = (
        selected_candidates - y_mean
    ) / y_std
    top1_indices = scores.argmax(axis=1)
    top1_candidates = all_candidates[np.arange(len(idx_val)), top1_indices][:, None, :]
    mean_candidates = all_candidates.mean(axis=1, keepdims=True)
    truth_normalized = y_normalized[idx_val]
    joint_distances = np.linalg.norm(
        candidates_normalized - truth_normalized[:, None, :],
        axis=-1,
    )
    joint_oracle_indices = joint_distances.argmin(axis=1)
    joint_oracle_candidates = all_candidates[
        np.arange(len(idx_val)),
        joint_oracle_indices,
    ][:, None, :]
    selected_joint_distances = np.linalg.norm(
        selected_candidates_normalized - truth_normalized[:, None, :],
        axis=-1,
    )
    selected_joint_indices = selected_joint_distances.argmin(axis=1)
    selected_joint_oracle_candidates = selected_candidates[
        np.arange(len(idx_val)),
        selected_joint_indices,
    ][:, None, :]

    known_val = labels.known_mask[idx_val]
    classifier_planet_prediction = labels.classifier.predict(z[idx_val][known_val])
    final_metrics = {
        "protocol": {
            "planet_label_mode": args.planet_label_mode,
            "split_seed": args.split_seed,
            "model_seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "ranker_steps": args.ranker_steps,
            "validation_role": "development holdout; not an independent test",
        },
        "dataset": {
            "rows": len(profiles),
            "train_rows": len(idx_train),
            "validation_rows": len(idx_val),
        },
        "pseudolabels": pseudo_summary,
        "per_seed": per_seed_metrics,
        "ensemble": {
            "candidate_count": int(all_candidates.shape[1]),
            "selected_diverse_best_of_5": evaluate_regression(
                selected_candidates,
                y_reg[idx_val],
            ),
            "oracle_of_all_candidates": evaluate_regression(
                all_candidates,
                y_reg[idx_val],
            ),
            "joint_candidate_oracle_all_25": evaluate_regression(
                joint_oracle_candidates,
                y_reg[idx_val],
            ),
            "joint_candidate_oracle_selected_5": evaluate_regression(
                selected_joint_oracle_candidates,
                y_reg[idx_val],
            ),
            "ranker_top_1": evaluate_regression(top1_candidates, y_reg[idx_val]),
            "mean_of_all_candidates": evaluate_regression(
                mean_candidates,
                y_reg[idx_val],
            ),
            "h2a_roc_auc": float(roc_auc_score(y_cls[idx_val], ensemble_probability)),
            "h2a_accuracy": float(
                accuracy_score(y_cls[idx_val], ensemble_probability >= 0.5)
            ),
            "planet_classifier_accuracy_known_validation": float(
                accuracy_score(
                    raw_planets[idx_val][known_val],
                    classifier_planet_prediction,
                )
            ),
            "best_epochs": best_epochs,
            "best_validation_losses": best_losses,
        },
        "elapsed_seconds": time.perf_counter() - run_start,
    }

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "planet_params": str(args.planet_params.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "planet_label_mode": args.planet_label_mode,
        "epochs": args.epochs,
        "model_seeds": seeds,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "ranker_steps": args.ranker_steps,
        "threads": args.threads,
        "device": str(device),
        "grid": {"start": -5.0, "stop": 5.0, "step": 0.1, "points": len(GRID)},
        "split": {
            "test_size": 0.2,
            "stratify": "H2a",
            "random_state": args.split_seed,
        },
        "preprocessing": {
            "pca_components": 16,
            "profile_clip_min": 1e-7,
            "fit_population": "train-only",
        },
        "model": {
            "input_channels": 2,
            "regression_heads": 5,
            "physical_features": len(PHYS_COLUMNS),
            "dropout": 0.2,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": args.epochs,
            "augmentation_noise_sigma_times_profile_max": 0.005,
            "wta_mean_head_weight": 0.1,
            "planet_ce_weight": 0.3,
            "h2a_loss": "BCEWithLogitsLoss",
        },
        "ranker": {
            "architecture": [19, 256, 256, 1],
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "dropout": 0.2,
            "seed": 0,
        },
        "planet_classifier": {
            "type": "GradientBoostingClassifier",
            "random_state": 0,
            "fit_population": "known training rows only",
        },
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
    }

    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "metrics.json", final_metrics)
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        object_ids=object_ids[idx_val],
        true_log=y_reg[idx_val],
        true_h2a=y_cls[idx_val],
        all_candidates_log=all_candidates,
        ensemble_h2a_probability=ensemble_probability,
        ranker_scores=scores,
        selected_indices=selected_indices,
        top1_indices=top1_indices,
        joint_oracle_indices=joint_oracle_indices,
        selected_joint_oracle_indices=selected_joint_indices,
    )
    np.savez_compressed(
        args.output_dir / "preproc.npz",
        grid=GRID,
        y_mean=y_mean,
        y_std=y_std,
        x_scale=x_scale,
        pca_mean=pca_mean,
        pca_vectors=pca_vectors,
        pca_z_mean=pca_z_mean,
        pca_z_std=pca_z_std,
        physical_table=physical_table_np,
        planet_names=np.asarray(PLANET_NAMES),
        idx_train=idx_train,
        idx_val=idx_val,
        train_ids=object_ids[idx_train],
        validation_ids=object_ids[idx_val],
    )
    if not args.no_save_models:
        torch.save(ranker.state_dict(), args.output_dir / "ranker.pt")
    write_artifact_manifest(
        args.output_dir,
        args.planet_params,
        source_status="generated by this runner at run completion",
    )

    print("FINAL_METRICS", json.dumps(final_metrics["ensemble"], ensure_ascii=False), flush=True)
    print(f"saved={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
