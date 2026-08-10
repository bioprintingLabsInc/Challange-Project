#!/usr/bin/env python3
"""Evaluate nested non-RNA shortcut baselines for the DNT feature bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_v4_seven_models import (
    RANDOM_STATE,
    bootstrap_intervals,
    fit_branch,
    metrics,
    validation_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nested_probability(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> np.ndarray:
    probability = np.full(len(y), np.nan)
    for fold, (training, testing) in enumerate(
        validation_splits(X, y, 5, RANDOM_STATE, groups), start=1
    ):
        fitted = fit_branch(
            X.iloc[training],
            y[training],
            RANDOM_STATE + 50000 + fold,
            groups[training] if groups is not None else None,
        )
        probability[testing] = fitted.predict_proba(X.iloc[testing])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Shortcut baseline produced incomplete predictions")
    return probability


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.feature_bundle)
    X_all: pd.DataFrame = bundle["X"]
    metadata = bundle["metadata"].set_index("chemical_code", drop=False)
    codes = metadata.index[metadata["split"].eq("training")]
    y = metadata.loc[codes, "label"].astype(int).to_numpy()
    group_column = next(
        (name for name in ("validation_group", "structure_group") if name in metadata),
        None,
    )
    groups = (
        metadata.loc[codes, group_column].astype(str).to_numpy()
        if group_column
        else None
    )
    chemical_columns = [column for column in X_all if column.startswith("chemical__")]
    families = {
        "chemical_context_only": chemical_columns,
        "pubchem_descriptors_only": [
            column for column in chemical_columns if column.startswith("chemical__pubchem__")
        ],
        "dose_only": [
            column for column in chemical_columns if column.startswith("chemical__dose__")
        ],
    }
    metric_rows = []
    predictions = metadata.loc[codes, ["chemical_code", "chemical", "label"]].copy()
    for name, columns in families.items():
        probability = nested_probability(X_all.loc[codes, columns], y, groups)
        result = metrics(y, probability)
        result.update(bootstrap_intervals(y, probability))
        metric_rows.append({"baseline": name, "feature_count": len(columns), **result})
        predictions[f"{name}__probability"] = probability
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "v4_shortcut_baseline_metrics.csv", index=False
    )
    predictions.to_csv(args.output_dir / "v4_shortcut_baseline_predictions.csv", index=False)
    print(pd.DataFrame(metric_rows)[["baseline", "roc_auc", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
