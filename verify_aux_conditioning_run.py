"""Independently verify saved metrics for an aux-conditioning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def regression_metrics(candidates_log: np.ndarray, truth_log: np.ndarray) -> dict[str, float]:
    prediction_linear = 10.0**candidates_log
    truth_linear = 10.0**truth_log
    absolute_log_error = np.abs(candidates_log - truth_log[:, None, :])
    absolute_linear_error = np.abs(prediction_linear - truth_linear[:, None, :])
    return {
        "xuv_mape_pct": float(
            100.0
            * np.mean(absolute_linear_error[:, :, 0].min(axis=1) / truth_linear[:, 0])
        ),
        "helium_mape_pct": float(
            100.0
            * np.mean(absolute_linear_error[:, :, 1].min(axis=1) / truth_linear[:, 1])
        ),
        "log_msw_mape_pct": float(
            100.0
            * np.mean(absolute_log_error[:, :, 2].min(axis=1) / np.abs(truth_log[:, 2]))
        ),
        "log_msw_mae_dex": float(np.mean(absolute_log_error[:, :, 2].min(axis=1))),
        "linear_msw_mape_pct": float(
            100.0
            * np.mean(absolute_linear_error[:, :, 2].min(axis=1) / truth_linear[:, 2])
        ),
    }


def assert_metric_block(
    name: str,
    calculated: dict[str, float],
    recorded: dict[str, float],
) -> None:
    for metric, value in calculated.items():
        if not np.isclose(value, recorded[metric], rtol=0.0, atol=1e-10):
            raise AssertionError(
                f"{name}.{metric}: calculated={value}, recorded={recorded[metric]}"
            )


def main() -> None:
    run_dir = parse_args().run_dir.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    predictions = np.load(run_dir / "validation_predictions.npz")
    preprocessing = np.load(run_dir / "preproc.npz")
    manifest = pd.read_csv(run_dir / "dataset_manifest.csv")

    truth = predictions["true_log"]
    candidates = predictions["all_candidates_log"]
    rows = np.arange(len(truth))
    selected = candidates[rows[:, None], predictions["selected_indices"]]
    truth_normalized = (
        truth - preprocessing["y_mean"]
    ) / preprocessing["y_std"]
    selected_normalized = (
        selected - preprocessing["y_mean"]
    ) / preprocessing["y_std"]
    selected_joint_indices = np.linalg.norm(
        selected_normalized - truth_normalized[:, None, :],
        axis=-1,
    ).argmin(axis=1)
    selected_joint = selected[rows, selected_joint_indices][:, None, :]
    top1_indices = predictions["top1_indices"]
    top1 = candidates[rows, top1_indices][:, None, :]
    joint = candidates[rows, predictions["joint_oracle_indices"]][:, None, :]
    mean = candidates.mean(axis=1, keepdims=True)

    if not np.array_equal(top1_indices, predictions["ranker_scores"].argmax(axis=1)):
        raise AssertionError("saved top1 indices do not match ranker-score argmax")

    calculated_blocks = {
        "selected_diverse_best_of_5": regression_metrics(selected, truth),
        "oracle_of_all_candidates": regression_metrics(candidates, truth),
        "ranker_top_1": regression_metrics(top1, truth),
        "mean_of_all_candidates": regression_metrics(mean, truth),
    }
    if "joint_candidate_oracle_all_25" in metrics["ensemble"]:
        calculated_blocks["joint_candidate_oracle_all_25"] = regression_metrics(
            joint, truth
        )
    else:
        calculated_blocks["joint_candidate_oracle"] = regression_metrics(joint, truth)
    if "joint_candidate_oracle_selected_5" in metrics["ensemble"]:
        calculated_blocks["joint_candidate_oracle_selected_5"] = regression_metrics(
            selected_joint, truth
        )
    if "selected_joint_oracle_indices" in predictions.files and not np.array_equal(
        predictions["selected_joint_oracle_indices"],
        selected_joint_indices,
    ):
        raise AssertionError("saved selected-5 joint-oracle indices do not match")
    for name, calculated in calculated_blocks.items():
        assert_metric_block(name, calculated, metrics["ensemble"][name])

    probability = predictions["ensemble_h2a_probability"]
    truth_h2a = predictions["true_h2a"]
    auc = float(roc_auc_score(truth_h2a, probability))
    accuracy = float(accuracy_score(truth_h2a, probability >= 0.5))
    if not np.isclose(auc, metrics["ensemble"]["h2a_roc_auc"], atol=1e-12):
        raise AssertionError("recalculated H2a AUC does not match metrics.json")
    if not np.isclose(accuracy, metrics["ensemble"]["h2a_accuracy"], atol=1e-12):
        raise AssertionError("recalculated H2a accuracy does not match metrics.json")

    validation_ids = (run_dir / "split_val_ids.txt").read_text().splitlines()
    if validation_ids != predictions["object_ids"].tolist():
        raise AssertionError("validation IDs differ between split file and predictions")
    if len(manifest) != metrics["dataset"]["rows"]:
        raise AssertionError("manifest row count does not match metrics.json")
    fit_validation = manifest.loc[
        (manifest["split"] == "validation") & manifest["planet_classifier_fit"]
    ]
    if len(fit_validation):
        raise AssertionError("planet classifier fit contains validation rows")

    print(
        json.dumps(
            {
                "status": "ok",
                "validation_rows": len(truth),
                "candidates_per_row": candidates.shape[1],
                "planet_classifier_fit_validation_rows": 0,
                "selected_diverse_best_of_5": calculated_blocks[
                    "selected_diverse_best_of_5"
                ],
                "h2a_roc_auc": auc,
                "h2a_accuracy": accuracy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
