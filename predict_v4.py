#!/usr/bin/env python3
"""Apply the fitted six-domain branches and final DNT combiner to a feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    features = pd.read_csv(args.features)
    if not features.columns.is_unique:
        duplicates = features.columns[features.columns.duplicated()].unique().tolist()
        raise ValueError(f"Feature table has duplicate columns: {duplicates[:10]}")
    if "chemical_code" in features:
        features = features.set_index("chemical_code", drop=False)
    missing = sorted(
        {
            column
            for branch in model["branch_order"]
            for column in model["branch_columns"][branch]
            if column not in features
        }
    )
    missing += sorted(
        column for column in model["public_score_columns"].values() if column not in features
    )
    if missing:
        raise ValueError(
            f"Feature table is missing {len(set(missing))} required columns; "
            f"first missing columns: {sorted(set(missing))[:10]}"
        )

    required_numeric = sorted(
        {
            column
            for branch in model["branch_order"]
            for column in model["branch_columns"][branch]
        }
        | set(model["public_score_columns"].values())
    )
    non_numeric = [
        column
        for column in required_numeric
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise ValueError(
            "Required model features must be numeric; non-numeric columns: "
            f"{non_numeric[:10]}"
        )

    public_columns = list(model["public_score_columns"].values())
    invalid_public = {
        column: int((~features[column].between(0.0, 1.0) | features[column].isna()).sum())
        for column in public_columns
        if (~features[column].between(0.0, 1.0) | features[column].isna()).any()
    }
    if invalid_public:
        raise ValueError(
            "Public integrated scores must be finite values between 0 and 1; "
            f"invalid counts: {invalid_public}"
        )

    branch_scores = pd.DataFrame(index=features.index)
    for branch in model["branch_order"]:
        columns = model["branch_columns"][branch]
        probability = model["branch_models"][branch].predict_proba(features[columns])[:, 1]
        if branch in model["public_score_columns"]:
            alpha = model["public_fusion_alphas"][branch]
            public = features[model["public_score_columns"][branch]].to_numpy()
            probability = (1 - alpha) * probability + alpha * public
        branch_scores[branch] = probability

    if not np.isfinite(branch_scores.to_numpy()).all():
        raise ValueError(
            "Branch prediction produced non-finite values. Check required numeric features "
            "and regenerate the table with the matching feature-preparation pipeline."
        )

    output = features[[column for column in ("chemical_code", "chemical") if column in features]].copy()
    for branch in model["branch_order"]:
        output[f"{branch}__probability"] = branch_scores[branch]
    output["positive_branch_count"] = branch_scores.ge(0.5).sum(axis=1)
    output["proposal_4_of_6_prediction"] = (output["positive_branch_count"] >= 4).astype(int)
    output["seven_model_probability"] = model["combiner"].predict_proba(
        branch_scores[model["branch_order"]]
    )[:, 1]
    output["seven_model_prediction"] = (
        output["seven_model_probability"] >= model["decision_threshold"]
    ).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
