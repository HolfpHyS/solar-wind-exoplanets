"""Train-only planet pseudolabels for conditioned experiments.

The original notebooks fitted the auxiliary planet classifier on every row
with a known planet name, including validation rows.  Predictions from that
classifier were then used as supervision for unknown training rows.  This
module makes the fit population explicit and enforces the leakage-free
train-only protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


PLANET_NAMES = ("WASP107b", "WASP69b", "WASP52b")
PLANET_TO_INDEX = {name: index for index, name in enumerate(PLANET_NAMES)}


@dataclass(frozen=True)
class PlanetLabelResult:
    labels: np.ndarray
    indices: np.ndarray
    known_mask: np.ndarray
    fit_mask: np.ndarray
    classifier: GradientBoostingClassifier


def make_planet_labels(
    features: np.ndarray,
    raw_names: np.ndarray,
    train_indices: np.ndarray,
    *,
    random_state: int = 0,
) -> PlanetLabelResult:
    """Fit the planet classifier and assign labels to unknown rows.

    Parameters
    ----------
    features:
        Profile features whose preprocessing was fitted on train only.
    raw_names:
        Original planet labels, including placeholders for unknown rows.
    train_indices:
        Indices of the training split.
    """

    features = np.asarray(features)
    raw_names = np.asarray(raw_names)
    train_indices = np.asarray(train_indices, dtype=int)

    if len(features) != len(raw_names):
        raise ValueError("features and raw_names must contain the same rows")

    known_mask = np.isin(raw_names, PLANET_NAMES)
    train_mask = np.zeros(len(raw_names), dtype=bool)
    train_mask[train_indices] = True
    fit_mask = known_mask & train_mask

    fit_classes = set(np.unique(raw_names[fit_mask]))
    missing = set(PLANET_NAMES) - fit_classes
    if missing:
        raise ValueError(f"planet classifier fit is missing classes: {sorted(missing)}")

    classifier = GradientBoostingClassifier(random_state=random_state)
    classifier.fit(features[fit_mask], raw_names[fit_mask])

    labels = raw_names.astype(object).copy()
    labels[~known_mask] = classifier.predict(features[~known_mask])
    unknown_labels = set(np.unique(labels[~known_mask]))
    if not unknown_labels.issubset(PLANET_TO_INDEX):
        raise ValueError(f"unexpected predicted planet labels: {sorted(unknown_labels)}")

    indices = np.array([PLANET_TO_INDEX[label] for label in labels], dtype=np.int64)
    return PlanetLabelResult(
        labels=np.asarray(labels, dtype=str),
        indices=indices,
        known_mask=known_mask,
        fit_mask=fit_mask,
        classifier=classifier,
    )
