"""Leakage-aware OOF ranker pilot for the conditioned candidate generator.

The experiment has two independent stages:

1. Generate out-of-fold candidates for the 490-row outer training split.
   Every candidate for an OOF row is produced by a ConditionedCNN that did
   not use that row for gradient updates, preprocessing, planet pseudo-label
   fitting, or checkpoint selection.
2. Train small listwise rankers on the actual OOF candidate distribution.
   Ranker epochs are selected with a whole producer fold held out, after which
   every ranker is refit on all OOF rows.
3. Refit the candidate generator on all 490 outer-train rows for a fixed
   number of epochs and evaluate frozen rankers on fresh predictions.  The
   primary score excludes historical-development rows whose exact
   target/profile duplicate group intersects outer train; the complete
   historical 123-row score is retained only as a secondary comparison.

The 123 rows are still a development set that has been inspected in prior
experiments. Results from this script are therefore a development pilot, not
an untouched final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)
from torch.utils.data import DataLoader

from planet_pseudolabels import make_planet_labels
from run_aux_conditioning import (
    ConditionedCNN,
    ProfileDataset,
    build_physical_table,
    load_dataset,
    make_ranker,
    resolve_device,
    sha256_file,
    wta_loss,
    write_json,
)


@dataclass
class Preprocessing:
    y_mean: np.ndarray
    y_std: np.ndarray
    x_scale: float
    pca_mean: np.ndarray
    pca_vectors: np.ndarray
    pca_z_mean: np.ndarray
    pca_z_std: np.ndarray
    z: np.ndarray


class RankerMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "participant"
        / "new_dataset_V3",
    )
    parser.add_argument(
        "--planet-params",
        type=Path,
        default=Path(__file__).resolve().parent / "aux_data" / "planet_params.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "oof_ranker_pilot",
    )
    parser.add_argument("--outer-split-seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--generator-seeds", default="0,1")
    parser.add_argument("--generator-epochs", type=int, default=250)
    parser.add_argument("--ranker-seeds", default="0,1,2")
    parser.add_argument("--ranker-epochs", type=int, default=250)
    parser.add_argument("--ranker-split-seed", type=int, default=2718)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ranker-batch-objects", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--objectives",
        default="xuv,joint,combined",
        help="Comma-separated subset of xuv,joint,combined.",
    )
    parser.add_argument(
        "--reuse-oof",
        action="store_true",
        help="Reuse output-dir/oof_candidates.npz and skip fold generators.",
    )
    parser.add_argument(
        "--reuse-final",
        action="store_true",
        help="Reuse output-dir/dev_candidates.npz and skip final refit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast integrity run: 2 folds, one seed, two generator epochs.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def fit_preprocessing(
    profiles: np.ndarray,
    y_log: np.ndarray,
    fit_indices: np.ndarray,
) -> Preprocessing:
    fit_indices = np.asarray(fit_indices, dtype=int)
    y_mean = y_log[fit_indices].mean(axis=0)
    y_std = y_log[fit_indices].std(axis=0)
    x_scale = float(profiles[fit_indices].max())
    log_profiles = np.log10(np.clip(profiles, 1e-7, None))
    pca_mean = log_profiles[fit_indices].mean(axis=0)
    _, _, vectors = np.linalg.svd(
        log_profiles[fit_indices] - pca_mean,
        full_matrices=False,
    )
    pca_vectors = vectors[:16]
    z_raw = (log_profiles - pca_mean) @ pca_vectors.T
    pca_z_mean = z_raw[fit_indices].mean(axis=0)
    pca_z_std = z_raw[fit_indices].std(axis=0)
    if np.any(pca_z_std == 0):
        raise ValueError("zero PCA feature standard deviation")
    z = ((z_raw - pca_z_mean) / pca_z_std).astype(np.float32)
    return Preprocessing(
        y_mean=y_mean,
        y_std=y_std,
        x_scale=x_scale,
        pca_mean=pca_mean,
        pca_vectors=pca_vectors,
        pca_z_mean=pca_z_mean,
        pca_z_std=pca_z_std,
        z=z,
    )


def save_preprocessing(path: Path, preproc: Preprocessing) -> None:
    np.savez(
        path,
        y_mean=preproc.y_mean,
        y_std=preproc.y_std,
        x_scale=preproc.x_scale,
        pca_mean=preproc.pca_mean,
        pca_vectors=preproc.pca_vectors,
        pca_z_mean=preproc.pca_z_mean,
        pca_z_std=preproc.pca_z_std,
    )


def load_preprocessing(path: Path, profiles: np.ndarray) -> Preprocessing:
    saved = np.load(path)
    log_profiles = np.log10(np.clip(profiles, 1e-7, None))
    z_raw = (
        log_profiles - saved["pca_mean"]
    ) @ saved["pca_vectors"].T
    z = (
        (z_raw - saved["pca_z_mean"]) / saved["pca_z_std"]
    ).astype(np.float32)
    return Preprocessing(
        y_mean=saved["y_mean"],
        y_std=saved["y_std"],
        x_scale=float(saved["x_scale"]),
        pca_mean=saved["pca_mean"],
        pca_vectors=saved["pca_vectors"],
        pca_z_mean=saved["pca_z_mean"],
        pca_z_std=saved["pca_z_std"],
        z=z,
    )


def target_and_profile_groups(
    profiles: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Union exact target duplicates and exact interpolated-profile duplicates."""

    indices = np.asarray(indices, dtype=int)
    parent = np.arange(len(indices))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    target_owner: dict[tuple[float, ...], int] = {}
    profile_owner: dict[str, int] = {}
    for local, global_index in enumerate(indices):
        target_key = tuple(float(value) for value in targets[global_index])
        if target_key in target_owner:
            union(local, target_owner[target_key])
        else:
            target_owner[target_key] = local
        profile_key = hashlib.sha256(
            np.ascontiguousarray(profiles[global_index]).tobytes()
        ).hexdigest()
        if profile_key in profile_owner:
            union(local, profile_owner[profile_key])
        else:
            profile_owner[profile_key] = local

    roots = [find(item) for item in range(len(indices))]
    root_to_group: dict[int, int] = {}
    groups = np.empty(len(indices), dtype=int)
    for item, root in enumerate(roots):
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        groups[item] = root_to_group[root]
    return groups


def train_fixed_generator(
    *,
    profiles: np.ndarray,
    y_log: np.ndarray,
    y_cls: np.ndarray,
    raw_planets: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    preproc: Preprocessing,
    physical_table: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    save_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Train for fixed epochs and predict rows never used for checkpoint choice."""

    train_indices = np.asarray(train_indices, dtype=int)
    predict_indices = np.asarray(predict_indices, dtype=int)
    labels = make_planet_labels(
        preproc.z,
        raw_planets,
        train_indices,
    )
    predict_mask = np.zeros(len(profiles), dtype=bool)
    predict_mask[predict_indices] = True
    fit_predict_overlap = int((labels.fit_mask & predict_mask).sum())
    if fit_predict_overlap:
        raise AssertionError(
            f"planet classifier fit contains {fit_predict_overlap} prediction rows"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    normalized_targets = (y_log - preproc.y_mean) / preproc.y_std
    train_loader = DataLoader(
        ProfileDataset(
            train_indices,
            profiles,
            normalized_targets,
            y_cls,
            labels.indices,
            preproc.x_scale,
            augment=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    predict_loader = DataLoader(
        ProfileDataset(
            predict_indices,
            profiles,
            normalized_targets,
            y_cls,
            labels.indices,
            preproc.x_scale,
            augment=False,
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = ConditionedCNN(physical_table).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    last_loss = float("nan")
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
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
            epoch_loss += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        scheduler.step()
        last_loss = epoch_loss / seen
        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
            or epoch + 1 == epochs
        ):
            print(
                f"generator seed={seed} epoch={epoch + 1}/{epochs} "
                f"train_loss={last_loss:.6f}",
                flush=True,
            )

    model.eval()
    candidate_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _, _, _ in predict_loader:
            pred_reg, pred_class, _ = model(batch_x.to(device))
            candidate_parts.append(pred_reg.cpu().numpy())
            probability_parts.append(torch.sigmoid(pred_class).cpu().numpy())
    candidates = np.concatenate(candidate_parts) * preproc.y_std + preproc.y_mean
    probabilities = np.concatenate(probability_parts)
    elapsed = time.perf_counter() - started
    state = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "seed": seed,
        "epochs": epochs,
        "fixed_epoch_protocol": True,
    }
    if save_path is not None:
        torch.save(state, save_path)
    return (
        candidates,
        probabilities,
        {
            "seed": seed,
            "epochs": epochs,
            "last_train_loss": last_loss,
            "elapsed_seconds": elapsed,
            "planet_classifier_fit_rows": int(labels.fit_mask.sum()),
            "planet_classifier_fit_prediction_rows": fit_predict_overlap,
        },
    )


def make_ranker_features(
    profile_z: np.ndarray,
    candidates_log: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    candidate_norm = (candidates_log - y_mean) / y_std
    median = np.median(candidate_norm, axis=1, keepdims=True)
    delta = candidate_norm - median
    distance_to_median = np.linalg.norm(delta, axis=-1, keepdims=True)
    pairwise = np.linalg.norm(
        candidate_norm[:, :, None, :] - candidate_norm[:, None, :, :],
        axis=-1,
    )
    diagonal = np.arange(candidate_norm.shape[1])
    pairwise[:, diagonal, diagonal] = np.inf
    neighbour_count = min(3, candidate_norm.shape[1] - 1)
    nearest_mean = np.sort(pairwise, axis=-1)[:, :, :neighbour_count].mean(
        axis=-1,
        keepdims=True,
    )
    repeated_profile = np.repeat(
        profile_z[:, None, :],
        candidate_norm.shape[1],
        axis=1,
    )
    features = np.concatenate(
        [
            repeated_profile,
            candidate_norm,
            delta,
            np.abs(delta),
            distance_to_median,
            nearest_mean,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def candidate_cost_components(
    candidates_log: np.ndarray,
    truth_log: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, np.ndarray]:
    if not np.all(np.isfinite(candidates_log)):
        raise ValueError("candidate pool contains non-finite predictions")
    if not np.all(np.isfinite(truth_log)):
        raise ValueError("truth contains non-finite values")
    truth_linear_xuv = 10.0 ** truth_log[:, 0]
    candidate_linear_xuv = 10.0 ** candidates_log[:, :, 0]
    xuv_ape = (
        np.abs(candidate_linear_xuv - truth_linear_xuv[:, None])
        / truth_linear_xuv[:, None]
    )
    candidate_norm = (candidates_log - y_mean) / y_std
    truth_norm = (truth_log - y_mean) / y_std
    joint = np.linalg.norm(
        candidate_norm - truth_norm[:, None, :],
        axis=-1,
    )
    if not np.all(np.isfinite(xuv_ape)) or not np.all(np.isfinite(joint)):
        raise ValueError("candidate costs contain non-finite values")
    return {
        "xuv": xuv_ape.astype(np.float32),
        "joint": joint.astype(np.float32),
    }


def objective_costs(
    components: dict[str, np.ndarray],
    objective: str,
    scale_fit_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build costs without consulting the held-out ranker fold."""

    scale_fit_indices = np.asarray(scale_fit_indices, dtype=int)
    if objective in ("xuv", "joint"):
        return components[objective], {}
    if objective != "combined":
        raise ValueError(f"unknown objective: {objective}")
    xuv_scale = max(
        float(np.median(components["xuv"][scale_fit_indices])),
        1e-6,
    )
    joint_scale = max(
        float(np.median(components["joint"][scale_fit_indices])),
        1e-6,
    )
    combined = (
        0.5 * components["xuv"] / xuv_scale
        + 0.5 * components["joint"] / joint_scale
    )
    return (
        combined.astype(np.float32),
        {
            "xuv_scale_fit_on_ranker_train": xuv_scale,
            "joint_scale_fit_on_ranker_train": joint_scale,
        },
    )


def listwise_loss(
    scores: torch.Tensor,
    costs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    target = torch.softmax(-costs / temperature, dim=-1)
    return -(target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


def train_ranker_ensemble(
    *,
    features: np.ndarray,
    cost_components: dict[str, np.ndarray],
    producer_fold_ids: np.ndarray,
    objective: str,
    seeds: list[int],
    epochs: int,
    batch_objects: int,
    split_seed: int,
    output_dir: Path,
) -> tuple[list[RankerMLP], dict[str, object], np.ndarray]:
    indices = np.arange(len(features))
    producer_fold_ids = np.asarray(producer_fold_ids, dtype=int)
    unique_folds = np.unique(producer_fold_ids)
    if len(unique_folds) < 2:
        raise ValueError("ranker epoch selection needs at least two OOF folds")
    input_dim = features.shape[-1]
    models: list[RankerMLP] = []
    model_summaries: list[dict[str, object]] = []
    validation_score_sum = np.zeros(features.shape[:2], dtype=np.float64)
    validation_score_count = np.zeros(len(features), dtype=int)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    for seed_position, seed in enumerate(seeds):
        validation_fold = int(
            unique_folds[(split_seed + seed_position) % len(unique_folds)]
        )
        validation_indices = indices[producer_fold_ids == validation_fold]
        train_indices = indices[producer_fold_ids != validation_fold]
        costs, cost_scales = objective_costs(
            cost_components,
            objective,
            train_indices,
        )
        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        ranges = (
            costs[train_indices].max(axis=1)
            - costs[train_indices].min(axis=1)
        )
        temperature = max(float(np.median(ranges)), 1e-3)

        torch.manual_seed(seed)
        selection_rng = np.random.RandomState(seed)
        model = RankerMLP(input_dim)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        best_validation = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        patience = max(20, epochs // 8)
        epochs_without_improvement = 0
        started = time.perf_counter()
        selection_epochs_run = 0
        for epoch in range(epochs):
            model.train()
            permutation = selection_rng.permutation(train_indices)
            for start in range(0, len(permutation), batch_objects):
                batch = permutation[start : start + batch_objects]
                scores = model(features_tensor[batch])
                loss = listwise_loss(
                    scores,
                    costs_tensor[batch],
                    temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    listwise_loss(
                        model(features_tensor[validation_indices]),
                        costs_tensor[validation_indices],
                        temperature,
                    ).item()
                )
            if validation_loss < best_validation - 1e-7:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            selection_epochs_run = epoch + 1
            if epochs_without_improvement >= patience:
                break
        if best_epoch < 1 or best_state is None:
            raise RuntimeError("ranker training did not select an epoch")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            validation_scores = (
                model(features_tensor[validation_indices]).numpy()
            )
        validation_score_sum[validation_indices] += validation_scores
        validation_score_count[validation_indices] += 1

        # The held-out producer fold was used only to select the epoch count.
        # Reset the model and refit on every OOF object for that fixed count.
        torch.manual_seed(seed)
        refit_rng = np.random.RandomState(seed)
        model = RankerMLP(input_dim)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        for _ in range(best_epoch):
            model.train()
            permutation = refit_rng.permutation(indices)
            for start in range(0, len(permutation), batch_objects):
                batch = permutation[start : start + batch_objects]
                scores = model(features_tensor[batch])
                loss = listwise_loss(
                    scores,
                    costs_tensor[batch],
                    temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            final_oof_loss = float(
                listwise_loss(
                    model(features_tensor),
                    costs_tensor,
                    temperature,
                ).item()
            )
        refit_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": refit_state,
                "objective": objective,
                "seed": seed,
                "input_dim": input_dim,
                "temperature": temperature,
                "best_epoch": best_epoch,
                "best_internal_validation_loss": best_validation,
                "validation_producer_fold": validation_fold,
                "cost_scales": cost_scales,
                "refit_on_all_oof_objects": True,
            },
            output_dir / f"ranker_{objective}_{seed}.pt",
        )
        models.append(model)
        model_summaries.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_internal_validation_loss": best_validation,
                "validation_producer_fold": validation_fold,
                "selection_train_objects": len(train_indices),
                "selection_validation_objects": len(validation_indices),
                "selection_epochs_run": selection_epochs_run,
                "cost_scales": cost_scales,
                "refit_objects": len(indices),
                "final_all_oof_loss": final_oof_loss,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    validation_scores = np.full(features.shape[:2], np.nan, dtype=np.float64)
    covered = validation_score_count > 0
    validation_scores[covered] = (
        validation_score_sum[covered]
        / validation_score_count[covered, None]
    )
    return (
        models,
        {
            "objective": objective,
            "epoch_selection": (
                "whole producer fold held out; each final model refit on all "
                "OOF objects for its selected fixed epoch count"
            ),
            "final_refit_objects": len(indices),
            "cross_validated_score_objects": int(covered.sum()),
            "cross_validated_score_fraction": float(covered.mean()),
            "models": model_summaries,
        },
        validation_scores,
    )


def score_ranker_ensemble(
    models: list[RankerMLP],
    features: np.ndarray,
) -> np.ndarray:
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        scores = np.stack([model(tensor).numpy() for model in models])
    return scores.mean(axis=0)


def score_legacy_ranker(
    *,
    profile_z: np.ndarray,
    candidates_log: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    checkpoint: Path,
) -> np.ndarray:
    candidate_norm = (candidates_log - y_mean) / y_std
    features = np.concatenate(
        [
            np.repeat(profile_z[:, None, :], candidates_log.shape[1], axis=1),
            candidate_norm,
        ],
        axis=-1,
    )
    ranker = make_ranker()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    ranker.load_state_dict(state)
    ranker.eval()
    with torch.no_grad():
        return (
            ranker(torch.tensor(features, dtype=torch.float32).reshape(-1, 19))
            .reshape(candidates_log.shape[:2])
            .numpy()
        )


def mmr_indices(
    candidates_normalized: np.ndarray,
    scores: np.ndarray,
    *,
    k: int = 5,
    score_weight: float = 0.75,
) -> np.ndarray:
    result = np.empty((len(scores), k), dtype=int)
    for row in range(len(scores)):
        score_row = scores[row]
        scaled_score = (score_row - score_row.min()) / (
            np.ptp(score_row) + 1e-9
        )
        distances = np.linalg.norm(
            candidates_normalized[row, :, None, :]
            - candidates_normalized[row, None, :, :],
            axis=-1,
        )
        distance_scale = max(float(distances.max()), 1e-9)
        chosen = [int(np.argmax(score_row))]
        for _ in range(k - 1):
            best_index: int | None = None
            best_gain = -np.inf
            for candidate_index in range(len(score_row)):
                if candidate_index in chosen:
                    continue
                diversity = min(
                    distances[candidate_index, previous] for previous in chosen
                ) / distance_scale
                gain = (
                    score_weight * scaled_score[candidate_index]
                    + (1.0 - score_weight) * diversity
                )
                if gain > best_gain:
                    best_gain = gain
                    best_index = candidate_index
            if best_index is None:
                raise RuntimeError("MMR selection failed")
            chosen.append(best_index)
        result[row] = chosen
    return result


def regression_metrics(
    candidates_log: np.ndarray,
    truth_log: np.ndarray,
) -> dict[str, float]:
    if not np.all(np.isfinite(candidates_log)):
        raise ValueError("metric input contains non-finite candidates")
    prediction_linear = 10.0 ** candidates_log
    truth_linear = 10.0 ** truth_log
    absolute_log_error = np.abs(candidates_log - truth_log[:, None, :])
    absolute_linear_error = np.abs(
        prediction_linear - truth_linear[:, None, :]
    )
    return {
        "xuv_mape_pct": float(
            100.0
            * np.mean(
                absolute_linear_error[:, :, 0].min(axis=1)
                / truth_linear[:, 0]
            )
        ),
        "helium_mape_pct": float(
            100.0
            * np.mean(
                absolute_linear_error[:, :, 1].min(axis=1)
                / truth_linear[:, 1]
            )
        ),
        "log_msw_mape_pct": float(
            100.0
            * np.mean(
                absolute_log_error[:, :, 2].min(axis=1)
                / np.abs(truth_log[:, 2])
            )
        ),
        "log_msw_mae_dex": float(
            absolute_log_error[:, :, 2].min(axis=1).mean()
        ),
        "linear_msw_mape_pct": float(
            100.0
            * np.mean(
                absolute_linear_error[:, :, 2].min(axis=1)
                / truth_linear[:, 2]
            )
        ),
    }


def evaluate_scores(
    *,
    name: str,
    scores: np.ndarray,
    candidates_log: np.ndarray,
    truth_log: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, object]:
    rows = np.arange(len(truth_log))
    order = np.argsort(-scores, axis=1)
    candidate_norm = (candidates_log - y_mean) / y_std
    truth_norm = (truth_log - y_mean) / y_std
    joint_distance = np.linalg.norm(
        candidate_norm - truth_norm[:, None, :],
        axis=-1,
    )
    best_joint = joint_distance.argmin(axis=1)
    inverse_rank = np.empty_like(order)
    inverse_rank[rows[:, None], order] = np.arange(order.shape[1])[None, :]
    best_joint_rank = inverse_rank[rows, best_joint] + 1

    truth_linear_xuv = 10.0 ** truth_log[:, 0]
    xuv_error = (
        np.abs(10.0 ** candidates_log[:, :, 0] - truth_linear_xuv[:, None])
        / truth_linear_xuv[:, None]
    )
    best_xuv = xuv_error.argmin(axis=1)
    best_xuv_rank = inverse_rank[rows, best_xuv] + 1
    joint_correlations = []
    xuv_correlations = []
    for row in rows:
        joint_correlation = spearmanr(
            scores[row],
            -joint_distance[row],
        ).statistic
        xuv_correlation = spearmanr(
            scores[row],
            -xuv_error[row],
        ).statistic
        joint_correlations.append(
            float(joint_correlation)
            if np.isfinite(joint_correlation)
            else 0.0
        )
        xuv_correlations.append(
            float(xuv_correlation)
            if np.isfinite(xuv_correlation)
            else 0.0
        )

    threshold_ranking: dict[str, dict[str, float]] = {}
    for threshold in (0.10, 0.20):
        ordered_relevant = xuv_error[rows[:, None], order] <= threshold
        has_relevant = ordered_relevant.any(axis=1)
        first_rank = ordered_relevant.argmax(axis=1) + 1
        reciprocal_rank = np.where(has_relevant, 1.0 / first_rank, 0.0)
        threshold_ranking[f"within_{int(threshold * 100)}pct"] = {
            "objects_with_relevant_candidate": float(has_relevant.mean()),
            "mean_reciprocal_rank_first_relevant": float(
                reciprocal_rank.mean()
            ),
        }

    output: dict[str, object] = {
        "name": name,
        "mean_spearman_score_vs_negative_joint_distance": float(
            np.mean(joint_correlations)
        ),
        "mean_spearman_score_vs_negative_xuv_ape": float(
            np.mean(xuv_correlations)
        ),
        "joint_argmin_candidate_rank": {
            "mean": float(best_joint_rank.mean()),
            "median": float(np.median(best_joint_rank)),
            "mean_reciprocal_rank": float(
                np.mean(1.0 / best_joint_rank)
            ),
            "recall_at_1": float(np.mean(best_joint_rank <= 1)),
            "recall_at_3": float(np.mean(best_joint_rank <= 3)),
            "recall_at_5": float(np.mean(best_joint_rank <= 5)),
        },
        "xuv_argmin_candidate_rank": {
            "mean": float(best_xuv_rank.mean()),
            "median": float(np.median(best_xuv_rank)),
            "mean_reciprocal_rank": float(np.mean(1.0 / best_xuv_rank)),
            "recall_at_1": float(np.mean(best_xuv_rank <= 1)),
            "recall_at_3": float(np.mean(best_xuv_rank <= 3)),
            "recall_at_5": float(np.mean(best_xuv_rank <= 5)),
        },
        "xuv_relevant_candidate_ranking": threshold_ranking,
        "ranked_coverage": {},
    }
    for k in (1, 3, 5):
        selected_indices = order[:, :k]
        selected = candidates_log[rows[:, None], selected_indices]
        selected_joint = joint_distance[rows[:, None], selected_indices]
        selected_xuv = xuv_error[rows[:, None], selected_indices]
        selected_error_linear = np.abs(
            10.0 ** selected - (10.0 ** truth_log)[:, None, :]
        )
        simultaneous = np.any(
            (
                selected_error_linear[:, :, 0]
                / (10.0 ** truth_log[:, 0])[:, None]
                <= 0.20
            )
            & (
                selected_error_linear[:, :, 1]
                / (10.0 ** truth_log[:, 1])[:, None]
                <= 0.20
            )
            & (
                np.abs(selected[:, :, 2] - truth_log[:, None, 2])
                <= 0.10
            ),
            axis=1,
        )
        output["ranked_coverage"][str(k)] = {
            "coordinate_oracle_metrics_at_k": regression_metrics(
                selected,
                truth_log,
            ),
            "mean_minimum_joint_distance": float(
                selected_joint.min(axis=1).mean()
            ),
            "joint_coverage_distance_le_0_5": float(
                np.mean(selected_joint.min(axis=1) <= 0.5)
            ),
            "xuv_hit_within_10pct": float(
                np.mean(selected_xuv.min(axis=1) <= 0.10)
            ),
            "xuv_hit_within_20pct": float(
                np.mean(selected_xuv.min(axis=1) <= 0.20)
            ),
            "simultaneous_hit_20pct_20pct_0_1dex": float(
                simultaneous.mean()
            ),
        }

    top1 = candidates_log[rows, order[:, 0]][:, None, :]
    output["actual_top1_metrics"] = regression_metrics(top1, truth_log)

    mmr = mmr_indices(candidate_norm, scores, k=5, score_weight=0.75)
    mmr_candidates = candidates_log[rows[:, None], mmr]
    mmr_joint = joint_distance[rows[:, None], mmr]
    mmr_xuv = xuv_error[rows[:, None], mmr]
    mmr_error_linear = np.abs(
        10.0 ** mmr_candidates - (10.0 ** truth_log)[:, None, :]
    )
    simultaneous_mmr = np.any(
        (
            mmr_error_linear[:, :, 0] / (10.0 ** truth_log[:, 0])[:, None]
            <= 0.20
        )
        & (
            mmr_error_linear[:, :, 1] / (10.0 ** truth_log[:, 1])[:, None]
            <= 0.20
        )
        & (np.abs(mmr_candidates[:, :, 2] - truth_log[:, None, 2]) <= 0.10),
        axis=1,
    )
    output["mmr_selected_5"] = {
        "coordinate_oracle_metrics_at_5": regression_metrics(
            mmr_candidates,
            truth_log,
        ),
        "mean_minimum_joint_distance": float(mmr_joint.min(axis=1).mean()),
        "joint_coverage_distance_le_0_5": float(
            np.mean(mmr_joint.min(axis=1) <= 0.5)
        ),
        "xuv_hit_within_10pct": float(
            np.mean(mmr_xuv.min(axis=1) <= 0.10)
        ),
        "xuv_hit_within_20pct": float(
            np.mean(mmr_xuv.min(axis=1) <= 0.20)
        ),
        "simultaneous_20pct_20pct_0_1dex": float(
            simultaneous_mmr.mean()
        ),
    }
    return output


def write_artifact_manifest(run_dir: Path, source_paths: list[Path]) -> None:
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
            "source_status": (
                "OOF ranker development pilot; primary metrics remove every "
                "historical-dev row in a duplicate group crossing train/dev; "
                "this is still not an untouched final test"
            ),
            "source_files": [
                {
                    "path": str(path.resolve()),
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
    if args.smoke:
        args.folds = 2
        args.generator_seeds = "0"
        args.generator_epochs = 2
        args.ranker_seeds = "0"
        args.ranker_epochs = 8
    generator_seeds = parse_int_list(args.generator_seeds)
    ranker_seeds = parse_int_list(args.ranker_seeds)
    objectives = [
        item.strip() for item in args.objectives.split(",") if item.strip()
    ]
    allowed_objectives = {"xuv", "joint", "combined"}
    if not objectives or not set(objectives).issubset(allowed_objectives):
        raise ValueError(
            f"objectives must be a non-empty subset of {sorted(allowed_objectives)}"
        )
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if args.generator_epochs <= 0 or args.ranker_epochs <= 0:
        raise ValueError("epoch counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass
    device = resolve_device(args.device)
    print(
        f"device={device} threads={torch.get_num_threads()} "
        f"folds={args.folds} generator_seeds={generator_seeds}",
        flush=True,
    )

    profiles, targets, raw_planets, object_ids, dataset_manifest = load_dataset(
        args.data_dir.resolve()
    )
    y_log = np.log10(targets[:, :3])
    y_cls = targets[:, 3].astype(np.float32)
    all_indices = np.arange(len(profiles))
    all_groups = target_and_profile_groups(
        profiles,
        targets,
        all_indices,
    )
    outer_train, outer_dev = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=args.outer_split_seed,
        stratify=y_cls,
    )
    train_group_set = set(all_groups[outer_train].tolist())
    dev_group_set = set(all_groups[outer_dev].tolist())
    crossing_groups = train_group_set & dev_group_set
    strict_dev_mask = ~np.isin(
        all_groups[outer_dev],
        np.asarray(sorted(crossing_groups), dtype=int),
    )
    strict_dev = outer_dev[strict_dev_mask]
    if len(strict_dev) == 0:
        raise ValueError("duplicate filtering removed the entire outer dev set")
    outer_preproc = fit_preprocessing(profiles, y_log, outer_train)
    save_preprocessing(args.output_dir / "preproc.npz", outer_preproc)
    physical_table_np = build_physical_table(args.planet_params.resolve())
    physical_table = torch.tensor(
        physical_table_np,
        dtype=torch.float32,
        device=device,
    )
    groups = all_groups[outer_train]
    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.fold_seed,
    )
    fold_splits = list(
        splitter.split(
            outer_train,
            y_cls[outer_train].astype(int),
            groups=groups,
        )
    )
    candidate_count = len(generator_seeds) * 5
    oof_path = args.output_dir / "oof_candidates.npz"
    fold_records: list[dict[str, object]] = []
    generator_summaries: list[dict[str, object]] = []

    if args.reuse_oof:
        saved = np.load(oof_path)
        oof_candidates = saved["candidates_log"]
        oof_probabilities = saved["h2a_probability"]
        fold_ids = saved["fold_ids"]
        if saved["object_ids"].tolist() != object_ids[outer_train].tolist():
            raise ValueError("reused OOF object IDs do not match current split")
        if oof_candidates.shape[1] != candidate_count:
            raise ValueError("reused OOF candidate count does not match seeds")
        required_metadata = {
            "generator_seeds": np.asarray(generator_seeds, dtype=int),
            "generator_epochs": np.asarray(args.generator_epochs),
            "folds": np.asarray(args.folds),
            "fold_seed": np.asarray(args.fold_seed),
            "outer_split_seed": np.asarray(args.outer_split_seed),
            "planet_params_sha256": np.asarray(
                sha256_file(args.planet_params.resolve())
            ),
        }
        for key, expected in required_metadata.items():
            if key not in saved or not np.array_equal(saved[key], expected):
                raise ValueError(
                    f"reused OOF artifact metadata mismatch: {key}"
                )
        if not np.array_equal(saved["groups"], groups):
            raise ValueError("reused OOF duplicate groups do not match")
        if not np.allclose(
            saved["truth_log"],
            y_log[outer_train],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("reused OOF truth fingerprint does not match")
    else:
        oof_candidates = np.full(
            (len(outer_train), candidate_count, 3),
            np.nan,
            dtype=np.float64,
        )
        oof_probabilities = np.full(
            (len(outer_train), len(generator_seeds)),
            np.nan,
            dtype=np.float32,
        )
        fold_ids = np.full(len(outer_train), -1, dtype=int)
        for fold, (fit_local, heldout_local) in enumerate(fold_splits):
            fit_global = outer_train[fit_local]
            heldout_global = outer_train[heldout_local]
            fold_ids[heldout_local] = fold
            fold_preproc = fit_preprocessing(profiles, y_log, fit_global)
            print(
                f"fold={fold} train={len(fit_global)} "
                f"heldout={len(heldout_global)}",
                flush=True,
            )
            per_seed_candidates = []
            for seed_position, seed in enumerate(generator_seeds):
                candidates, probability, summary = train_fixed_generator(
                    profiles=profiles,
                    y_log=y_log,
                    y_cls=y_cls,
                    raw_planets=raw_planets,
                    train_indices=fit_global,
                    predict_indices=heldout_global,
                    preproc=fold_preproc,
                    physical_table=physical_table,
                    seed=seed,
                    epochs=args.generator_epochs,
                    batch_size=args.batch_size,
                    device=device,
                    save_path=None,
                )
                per_seed_candidates.append(candidates)
                oof_probabilities[heldout_local, seed_position] = probability
                summary.update(
                    {
                        "stage": "oof",
                        "fold": fold,
                        "train_rows": len(fit_global),
                        "prediction_rows": len(heldout_global),
                    }
                )
                generator_summaries.append(summary)
            oof_candidates[heldout_local] = np.concatenate(
                per_seed_candidates,
                axis=1,
            )
            train_id_set = set(object_ids[fit_global].tolist())
            for local_index in heldout_local:
                row_id = object_ids[outer_train[local_index]]
                if row_id in train_id_set:
                    raise AssertionError("OOF row appears in producer train IDs")
                fold_records.append(
                    {
                        "object_id": row_id,
                        "outer_train_local_index": int(local_index),
                        "fold": fold,
                        "group": int(groups[local_index]),
                        "producer_train_rows": len(fit_global),
                        "row_in_producer_train": False,
                    }
                )
        if np.any(fold_ids < 0) or not np.all(np.isfinite(oof_candidates)):
            raise AssertionError("OOF candidates are incomplete")
        np.savez(
            oof_path,
            object_ids=object_ids[outer_train],
            candidates_log=oof_candidates,
            truth_log=y_log[outer_train],
            true_h2a=y_cls[outer_train],
            h2a_probability=oof_probabilities.mean(axis=1),
            fold_ids=fold_ids,
            groups=groups,
            generator_seeds=np.asarray(generator_seeds, dtype=int),
            generator_epochs=np.asarray(args.generator_epochs),
            folds=np.asarray(args.folds),
            fold_seed=np.asarray(args.fold_seed),
            outer_split_seed=np.asarray(args.outer_split_seed),
            planet_params_sha256=np.asarray(
                sha256_file(args.planet_params.resolve())
            ),
        )
        pd.DataFrame(fold_records).sort_values(
            "outer_train_local_index"
        ).to_csv(args.output_dir / "oof_manifest.csv", index=False)

    oof_features = make_ranker_features(
        outer_preproc.z[outer_train],
        oof_candidates,
        outer_preproc.y_mean,
        outer_preproc.y_std,
    )
    cost_components = candidate_cost_components(
        oof_candidates,
        y_log[outer_train],
        outer_preproc.y_mean,
        outer_preproc.y_std,
    )
    trained_rankers: dict[str, list[RankerMLP]] = {}
    ranker_summaries: dict[str, object] = {}
    ranker_cv_scores: dict[str, np.ndarray] = {}
    ranker_cv_metrics: dict[str, object] = {}
    for objective in objectives:
        models, summary, validation_scores = train_ranker_ensemble(
            features=oof_features,
            cost_components=cost_components,
            producer_fold_ids=fold_ids,
            objective=objective,
            seeds=ranker_seeds,
            epochs=args.ranker_epochs,
            batch_objects=args.ranker_batch_objects,
            split_seed=args.ranker_split_seed,
            output_dir=args.output_dir,
        )
        trained_rankers[objective] = models
        ranker_summaries[objective] = summary
        ranker_cv_scores[objective] = validation_scores
        covered = np.all(np.isfinite(validation_scores), axis=1)
        if np.any(covered):
            ranker_cv_metrics[objective] = evaluate_scores(
                name=f"producer_fold_cv_{objective}",
                scores=validation_scores[covered],
                candidates_log=oof_candidates[covered],
                truth_log=y_log[outer_train][covered],
                y_mean=outer_preproc.y_mean,
                y_std=outer_preproc.y_std,
            )
        print(
            f"ranker objective={objective} "
            f"internal_val={summary['models'][0]['best_internal_validation_loss']:.6f}",
            flush=True,
        )
    np.savez(
        args.output_dir / "oof_ranker_cross_validation_scores.npz",
        object_ids=object_ids[outer_train],
        fold_ids=fold_ids,
        **ranker_cv_scores,
    )

    dev_path = args.output_dir / "dev_candidates.npz"
    final_generator_summaries: list[dict[str, object]] = []
    if args.reuse_final:
        saved = np.load(dev_path)
        dev_candidates = saved["candidates_log"]
        dev_h2a_probability = saved["h2a_probability"]
        if saved["object_ids"].tolist() != object_ids[outer_dev].tolist():
            raise ValueError("reused dev object IDs do not match current split")
        if dev_candidates.shape[1] != candidate_count:
            raise ValueError("reused dev candidate count does not match seeds")
        required_metadata = {
            "generator_seeds": np.asarray(generator_seeds, dtype=int),
            "generator_epochs": np.asarray(args.generator_epochs),
            "outer_split_seed": np.asarray(args.outer_split_seed),
            "planet_params_sha256": np.asarray(
                sha256_file(args.planet_params.resolve())
            ),
        }
        for key, expected in required_metadata.items():
            if key not in saved or not np.array_equal(saved[key], expected):
                raise ValueError(
                    f"reused final artifact metadata mismatch: {key}"
                )
    else:
        per_seed_dev_candidates = []
        per_seed_dev_probability = []
        for seed in generator_seeds:
            candidates, probability, summary = train_fixed_generator(
                profiles=profiles,
                y_log=y_log,
                y_cls=y_cls,
                raw_planets=raw_planets,
                train_indices=outer_train,
                predict_indices=outer_dev,
                preproc=outer_preproc,
                physical_table=physical_table,
                seed=seed,
                epochs=args.generator_epochs,
                batch_size=args.batch_size,
                device=device,
                save_path=args.output_dir / f"final_model_{seed}.pt",
            )
            per_seed_dev_candidates.append(candidates)
            per_seed_dev_probability.append(probability)
            summary.update(
                {
                    "stage": "final_refit",
                    "train_rows": len(outer_train),
                    "prediction_rows": len(outer_dev),
                }
            )
            final_generator_summaries.append(summary)
        dev_candidates = np.concatenate(per_seed_dev_candidates, axis=1)
        dev_h2a_probability = np.mean(per_seed_dev_probability, axis=0)
        np.savez(
            dev_path,
            object_ids=object_ids[outer_dev],
            candidates_log=dev_candidates,
            h2a_probability=dev_h2a_probability,
            generator_seeds=np.asarray(generator_seeds, dtype=int),
            generator_epochs=np.asarray(args.generator_epochs),
            outer_split_seed=np.asarray(args.outer_split_seed),
            planet_params_sha256=np.asarray(
                sha256_file(args.planet_params.resolve())
            ),
        )
        np.savez(
            args.output_dir / "dev_truth_for_evaluation.npz",
            object_ids=object_ids[outer_dev],
            true_log=y_log[outer_dev],
            true_h2a=y_cls[outer_dev],
        )

    dev_features = make_ranker_features(
        outer_preproc.z[outer_dev],
        dev_candidates,
        outer_preproc.y_mean,
        outer_preproc.y_std,
    )
    strict_positions = np.flatnonzero(strict_dev_mask)
    score_blocks: dict[str, np.ndarray] = {}
    strict_metrics: dict[str, object] = {}
    historical_metrics: dict[str, object] = {}
    for objective, models in trained_rankers.items():
        scores = score_ranker_ensemble(models, dev_features)
        score_blocks[objective] = scores
        strict_metrics[objective] = evaluate_scores(
            name=f"oof_{objective}",
            scores=scores[strict_positions],
            candidates_log=dev_candidates[strict_positions],
            truth_log=y_log[strict_dev],
            y_mean=outer_preproc.y_mean,
            y_std=outer_preproc.y_std,
        )
        historical_metrics[objective] = evaluate_scores(
            name=f"oof_{objective}",
            scores=scores,
            candidates_log=dev_candidates,
            truth_log=y_log[outer_dev],
            y_mean=outer_preproc.y_mean,
            y_std=outer_preproc.y_std,
        )

    legacy_dir = (
        Path(__file__).resolve().parent
        / "runs"
        / "aux_conditioning_train_only"
    )
    legacy_checkpoint = legacy_dir / "ranker.pt"
    legacy_preproc_path = legacy_dir / "preproc.npz"
    legacy_train_ids_path = legacy_dir / "split_train_ids.txt"
    legacy_baseline_status = "not_available"
    legacy_split_matches = False
    if legacy_train_ids_path.exists():
        legacy_train_ids = set(
            legacy_train_ids_path.read_text(encoding="utf-8").splitlines()
        )
        current_train_ids = set(
            str(item) for item in object_ids[outer_train].tolist()
        )
        legacy_split_matches = legacy_train_ids == current_train_ids
    if (
        legacy_checkpoint.exists()
        and legacy_preproc_path.exists()
        and legacy_split_matches
    ):
        legacy_preproc = load_preprocessing(
            legacy_preproc_path,
            profiles,
        )
        legacy_scores = score_legacy_ranker(
            profile_z=legacy_preproc.z[outer_dev],
            candidates_log=dev_candidates,
            y_mean=legacy_preproc.y_mean,
            y_std=legacy_preproc.y_std,
            checkpoint=legacy_checkpoint,
        )
        legacy_baseline_status = (
            "included; exact outer-train ID set and saved preprocessing match"
        )
        score_blocks["legacy"] = legacy_scores
        strict_metrics["legacy"] = evaluate_scores(
            name="legacy_synthetic_pair_ranker",
            scores=legacy_scores[strict_positions],
            candidates_log=dev_candidates[strict_positions],
            truth_log=y_log[strict_dev],
            y_mean=outer_preproc.y_mean,
            y_std=outer_preproc.y_std,
        )
        historical_metrics["legacy"] = evaluate_scores(
            name="legacy_synthetic_pair_ranker",
            scores=legacy_scores,
            candidates_log=dev_candidates,
            truth_log=y_log[outer_dev],
            y_mean=outer_preproc.y_mean,
            y_std=outer_preproc.y_std,
        )
    elif legacy_checkpoint.exists():
        legacy_baseline_status = (
            "excluded; saved legacy outer-train IDs or preprocessing do not "
            "match the current run"
        )

    np.savez(
        args.output_dir / "dev_ranker_scores.npz",
        object_ids=object_ids[outer_dev],
        **score_blocks,
    )
    strict_oracle_metrics = regression_metrics(
        dev_candidates[strict_positions],
        y_log[strict_dev],
    )
    historical_oracle_metrics = regression_metrics(
        dev_candidates,
        y_log[outer_dev],
    )
    final_metrics = {
        "status": "development_pilot",
        "warning": (
            "The outer development split has been inspected in prior "
            "experiments and is not an untouched final test. Primary metrics "
            "exclude every dev row whose exact target/profile duplicate group "
            "intersects outer train."
        ),
        "dataset": {
            "rows": len(profiles),
            "outer_train_rows": len(outer_train),
            "outer_development_rows": len(outer_dev),
            "primary_nonoverlap_development_rows": len(strict_dev),
            "train_dev_crossing_duplicate_groups": len(crossing_groups),
            "excluded_overlapping_development_rows": int(
                (~strict_dev_mask).sum()
            ),
            "oof_folds": args.folds,
            "duplicate_row_excess_outer_train": int(
                len(groups) - len(np.unique(groups))
            ),
            "nontrivial_duplicate_groups_outer_train": int(
                np.sum(
                    np.unique(groups, return_counts=True)[1] > 1
                )
            ),
        },
        "candidate_generator": {
            "seeds": generator_seeds,
            "heads_per_model": 5,
            "candidate_count": candidate_count,
            "fixed_epochs": args.generator_epochs,
            "checkpoint_selection": "none; fixed epoch",
            "oof_models": generator_summaries,
            "final_models": final_generator_summaries,
        },
        "rankers": ranker_summaries,
        "ranker_producer_fold_cross_validation": {
            "status": (
                "objective-selection diagnostic; candidates and ranker scores "
                "are both out of fold for every covered object"
            ),
            "evaluation": ranker_cv_metrics,
        },
        "legacy_baseline_status": legacy_baseline_status,
        "primary_nonoverlap_evaluation": {
            "oracle_all_candidates_coordinate_metrics": strict_oracle_metrics,
            "rankers": strict_metrics,
        },
        "secondary_historical_123_evaluation": {
            "status": (
                "comparability only; includes duplicate-overlap rows and is "
                "not the primary leakage-free score"
            ),
            "oracle_all_candidates_coordinate_metrics": (
                historical_oracle_metrics
            ),
            "rankers": historical_metrics,
        },
    }
    write_json(args.output_dir / "metrics.json", final_metrics)

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "planet_params": str(args.planet_params.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "outer_split_seed": args.outer_split_seed,
        "fold_seed": args.fold_seed,
        "folds": args.folds,
        "grouping": "union of exact target tuples and exact interpolated profiles",
        "primary_evaluation": (
            "historical outer dev after excluding any group that intersects "
            "outer train"
        ),
        "generator_seeds": generator_seeds,
        "generator_epochs": args.generator_epochs,
        "batch_size": args.batch_size,
        "generator_checkpoint_selection": "fixed epochs; no OOF/dev early stopping",
        "ranker_seeds": ranker_seeds,
        "ranker_epochs": args.ranker_epochs,
        "ranker_batch_objects": args.ranker_batch_objects,
        "ranker_split_seed": args.ranker_split_seed,
        "ranker_epoch_selection": (
            "whole OOF producer fold held out, then refit on all OOF rows"
        ),
        "ranker_features": (
            "profile_pca16 + candidate3 + delta_to_pool_median3 + "
            "abs_delta3 + distance_to_median1 + mean_3nn_distance1"
        ),
        "ranker_model": "MLP 27->64->32->1, dropout=0.1",
        "objectives": objectives,
        "combined_objective_definition": (
            "0.5 * scaled XUV APE + 0.5 * scaled three-coordinate joint "
            "distance; therefore XUV is intentionally weighted twice"
        ),
        "device": str(device),
        "threads": args.threads,
        "reuse_oof": args.reuse_oof,
        "reuse_final": args.reuse_final,
        "argv": sys.argv,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(args.output_dir / "config.json", config)
    dataset_manifest["split"] = "unused"
    dataset_manifest["evaluation_status"] = "not_evaluated"
    dataset_manifest.loc[outer_train, "split"] = "outer_train"
    dataset_manifest.loc[outer_dev, "split"] = "outer_development"
    dataset_manifest.loc[outer_dev, "evaluation_status"] = (
        "secondary_historical"
    )
    dataset_manifest.loc[strict_dev, "evaluation_status"] = (
        "primary_nonoverlap"
    )
    dataset_manifest.to_csv(
        args.output_dir / "dataset_manifest.csv",
        index=False,
    )
    write_artifact_manifest(
        args.output_dir,
        [
            Path(__file__),
            Path(__file__).resolve().with_name("run_aux_conditioning.py"),
            Path(__file__).resolve().with_name("planet_pseudolabels.py"),
            args.planet_params,
        ],
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
