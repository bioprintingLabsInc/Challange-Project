#!/usr/bin/env python3
"""Train six domain classifiers and a leakage-controlled stacked DNT combiner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


RANDOM_STATE = 42
BRANCH_ORDER = [
    "growth_proliferation",
    "progenitor_organization",
    "excitatory_inhibitory_balance",
    "synaptic_integrity",
    "electrophysiology",
    "transcriptomic_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v3-metrics", type=Path)
    parser.add_argument("--model-version", default="4.1")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation (useful on restricted or headless systems)",
    )
    return parser.parse_args()


def branch_pipeline(n_features: int) -> tuple[Pipeline, dict[str, list[object]]]:
    percentiles = [1, 2, 5] if n_features > 1000 else [25, 50, 100]
    sklearn_major_minor = tuple(int(part) for part in SKLEARN_VERSION.split(".")[:2])
    classifier_options = {
        "solver": "saga",
        "l1_ratio": 0.0,
        "class_weight": "balanced",
        "max_iter": 10000,
        "random_state": RANDOM_STATE,
    }
    # Before 1.8, scikit-learn requires the explicit elastic-net penalty for l1_ratio.
    # In 1.8+, penalty is deprecated and l1_ratio directly selects the L1/L2 continuum.
    if sklearn_major_minor < (1, 8):
        classifier_options["penalty"] = "elasticnet"
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(0.0)),
            ("select", SelectPercentile(f_classif, percentile=50)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(**classifier_options),
            ),
        ]
    )
    grid = {
        "select__percentile": percentiles,
        "classifier__C": [0.01, 0.1, 1.0],
        # The elastic-net endpoints are true L2 (0.0) and true L1 (1.0) fits.
        "classifier__l1_ratio": [0.0, 1.0],
    }
    return pipeline, grid


def validation_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
        validate_splits(splits, y, groups=None)
        return splits
    if pd.isna(groups).any():
        raise ValueError("Validation groups contain missing values")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Group-aware validation needs at least {n_splits} distinct groups; "
            f"found {len(np.unique(groups))}"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    validate_splits(splits, y, groups=groups)
    return splits


def validate_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray | None,
) -> None:
    for fold, (train, test) in enumerate(splits, start=1):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise ValueError(f"Validation fold {fold} does not contain both classes")
        if groups is not None and not set(groups[train]).isdisjoint(groups[test]):
            raise RuntimeError(f"Validation fold {fold} leaks groups across train and test")


def fit_branch(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> GridSearchCV:
    pipeline, grid = branch_pipeline(X.shape[1])
    cv = validation_splits(X, y, n_splits=3, seed=seed, groups=groups)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def meta_model() -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def fit_meta(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> LogisticRegression:
    # With only 48 chemicals in each outer-training fold, tuning a six-input combiner
    # produced highly unstable C values and coefficients. Keep this regularization fixed
    # a priori; the branch models still perform fully nested hyperparameter selection.
    del seed, groups
    model = meta_model()
    model.fit(X, y)
    return model


def nested_stack(
    X: pd.DataFrame,
    y: np.ndarray,
    branch_columns: dict[str, list[str]],
    label: str,
    public_score_columns: dict[str, str] | None = None,
    fusion_alphas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, object]]]:
    outer_splits = validation_splits(
        X, y, n_splits=5, seed=RANDOM_STATE, groups=groups
    )
    stack_oof = np.full(len(y), np.nan)
    branch_oof = pd.DataFrame(index=np.arange(len(y)), columns=BRANCH_ORDER, dtype=float)
    fold_rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, start=1):
        outer_train_groups = groups[outer_train] if groups is not None else None
        inner_splits = validation_splits(
            X.iloc[outer_train],
            y[outer_train],
            n_splits=4,
            seed=RANDOM_STATE + outer_fold,
            groups=outer_train_groups,
        )
        meta_training = pd.DataFrame(
            index=np.arange(len(outer_train)), columns=BRANCH_ORDER, dtype=float
        )
        test_scores = pd.DataFrame(index=np.arange(len(outer_test)), columns=BRANCH_ORDER)
        for branch_number, branch in enumerate(BRANCH_ORDER):
            columns = branch_columns[branch]
            inner_probability = np.full(len(outer_train), np.nan)
            for inner_fold, (inner_train_local, inner_valid_local) in enumerate(
                inner_splits, start=1
            ):
                train_index = outer_train[inner_train_local]
                valid_index = outer_train[inner_valid_local]
                fitted = fit_branch(
                    X.iloc[train_index][columns],
                    y[train_index],
                    RANDOM_STATE + outer_fold * 100 + branch_number * 10 + inner_fold,
                    groups[train_index] if groups is not None else None,
                )
                inner_probability[inner_valid_local] = fitted.predict_proba(
                    X.iloc[valid_index][columns]
                )[:, 1]
            if np.isnan(inner_probability).any():
                raise RuntimeError(f"Incomplete inner OOF predictions for {branch}")
            fitted_outer = fit_branch(
                X.iloc[outer_train][columns],
                y[outer_train],
                RANDOM_STATE + outer_fold * 1000 + branch_number,
                outer_train_groups,
            )
            probability = fitted_outer.predict_proba(X.iloc[outer_test][columns])[:, 1]
            selected_alpha = 0.0
            if public_score_columns and branch in public_score_columns:
                public_column = public_score_columns[branch]
                public_train = X.iloc[outer_train][public_column].to_numpy()
                public_test = X.iloc[outer_test][public_column].to_numpy()
                candidates = []
                for alpha in fusion_alphas:
                    fused = (1 - alpha) * inner_probability + alpha * public_train
                    candidates.append((roc_auc_score(y[outer_train], fused), -alpha, alpha))
                selected_alpha = max(candidates)[2]
                inner_probability = (
                    (1 - selected_alpha) * inner_probability + selected_alpha * public_train
                )
                probability = (1 - selected_alpha) * probability + selected_alpha * public_test
            meta_training[branch] = inner_probability
            test_scores[branch] = probability
            branch_oof.loc[outer_test, branch] = probability
            fold_rows.append(
                {
                    "evaluation": label,
                    "outer_fold": outer_fold,
                    "branch": branch,
                    "train_n": len(outer_train),
                    "test_n": len(outer_test),
                    "feature_count": len(columns),
                    "public_fusion_alpha": selected_alpha,
                    "best_parameters": json.dumps(fitted_outer.best_params_, sort_keys=True),
                }
            )
        meta = fit_meta(
            meta_training[BRANCH_ORDER],
            y[outer_train],
            RANDOM_STATE + outer_fold * 10000,
            outer_train_groups,
        )
        stack_oof[outer_test] = meta.predict_proba(test_scores[BRANCH_ORDER])[:, 1]
        fold_rows.append(
            {
                "evaluation": label,
                "outer_fold": outer_fold,
                "branch": "__combiner__",
                "train_n": len(outer_train),
                "test_n": len(outer_test),
                "feature_count": len(BRANCH_ORDER),
                "public_fusion_alpha": np.nan,
                "best_parameters": json.dumps(
                    {"C": meta.C, "selection": "pre-specified"}, sort_keys=True
                ),
            }
        )
    if np.isnan(stack_oof).any() or branch_oof.isna().any().any():
        raise RuntimeError("Nested stacking produced missing out-of-fold predictions")
    return stack_oof, branch_oof, fold_rows


def metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray | None = None,
) -> dict[str, float | int]:
    if prediction is None:
        prediction = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def bootstrap_intervals(
    y: np.ndarray,
    probability: np.ndarray,
    iterations: int = 4000,
    prediction: np.ndarray | None = None,
) -> dict[str, float]:
    """Conditional intervals for the fixed set of out-of-fold predictions.

    These intervals quantify chemical-sampling uncertainty. They intentionally do not claim to
    include the additional variation caused by refitting and selecting the entire model.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    auc_values, accuracy_values = [], []
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if prediction is None:
        prediction = probability >= 0.5
    for _ in range(iterations):
        index = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        auc_values.append(roc_auc_score(y[index], probability[index]))
        accuracy_values.append(accuracy_score(y[index], prediction[index]))
    return {
        "conditional_oof_roc_auc_ci_low": float(np.quantile(auc_values, 0.025)),
        "conditional_oof_roc_auc_ci_high": float(np.quantile(auc_values, 0.975)),
        "conditional_oof_accuracy_ci_low": float(np.quantile(accuracy_values, 0.025)),
        "conditional_oof_accuracy_ci_high": float(np.quantile(accuracy_values, 0.975)),
    }


def save_plots(output: Path, y: np.ndarray, probability: np.ndarray, branch: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6.3, 5.2))
    fpr, tpr, _ = roc_curve(y, probability)
    axis.plot(fpr, tpr, linewidth=2.5, label=f"7-model stack (AUC={roc_auc_score(y, probability):.3f})")
    for name in BRANCH_ORDER:
        branch_fpr, branch_tpr, _ = roc_curve(y, branch[name])
        axis.plot(
            branch_fpr,
            branch_tpr,
            alpha=0.55,
            linewidth=1.2,
            label=f"{name.replace('_', ' ')} ({roc_auc_score(y, branch[name]):.2f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Nested chemical-level ROC curves")
    axis.legend(fontsize=7.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "v4_nested_roc_curves.png", dpi=190)
    plt.close(figure)

    values = pd.Series({name: roc_auc_score(y, branch[name]) for name in BRANCH_ORDER}).sort_values()
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.barh([name.replace("_", " ") for name in values.index], values.values, color="#2563EB")
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(xlim=(0.35, 1.0), xlabel="Nested out-of-fold ROC AUC", title="Six domain models")
    figure.tight_layout()
    figure.savefig(output / "v4_branch_auc.png", dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.feature_bundle)
    X_all: pd.DataFrame = bundle["X"]
    metadata: pd.DataFrame = bundle["metadata"].set_index("chemical_code", drop=False)
    branches: dict[str, list[str]] = bundle["branch_columns"]
    training_codes = metadata.index[metadata["split"].eq("training")]
    benchmark_codes = metadata.index[metadata["split"].eq("legacy_benchmark")]
    X = X_all.loc[training_codes]
    y = metadata.loc[training_codes, "label"].astype(int).to_numpy()
    if not X_all.index.is_unique or not metadata.index.is_unique:
        raise ValueError("Feature and metadata chemical_code values must be unique")
    if not X_all.index.equals(metadata.index):
        missing_metadata = X_all.index.difference(metadata.index).tolist()
        missing_features = metadata.index.difference(X_all.index).tolist()
        raise ValueError(
            "Feature/metadata chemical codes are misaligned; "
            f"missing metadata={missing_metadata[:5]}, missing features={missing_features[:5]}"
        )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Training labels must be binary 0/1; found {sorted(np.unique(y))}")
    group_column = next(
        (column for column in ("validation_group", "structure_group") if column in metadata),
        None,
    )
    groups = (
        metadata.loc[training_codes, group_column].astype(str).to_numpy()
        if group_column
        else None
    )
    if groups is None:
        warnings.warn(
            "No validation_group or structure_group column was found. Validation will be "
            "chemical-level stratified CV, which does not prevent close structural analogues "
            "from appearing in different folds.",
            stacklevel=2,
        )

    no_public_branches = {
        branch: [column for column in columns if not column.startswith("public__")]
        for branch, columns in branches.items()
    }
    public_score_columns = {
        branch: f"public__{branch}__integrated_score"
        for branch in (
            "growth_proliferation",
            "progenitor_organization",
            "synaptic_integrity",
            "electrophysiology",
        )
    }
    integrated_probability, integrated_branch, integrated_folds = nested_stack(
        X,
        y,
        no_public_branches,
        "integrated_public_assays_late_fusion",
        public_score_columns=public_score_columns,
        groups=groups,
    )
    no_public_probability, no_public_branch, no_public_folds = nested_stack(
        X, y, no_public_branches, "no_public_assay_transfer", groups=groups
    )

    rows = []
    primary_metrics = metrics(y, integrated_probability)
    primary_metrics.update(bootstrap_intervals(y, integrated_probability))
    rows.append({"model": "seven_model_integrated", **primary_metrics})
    no_public_metrics = metrics(y, no_public_probability)
    no_public_metrics.update(bootstrap_intervals(y, no_public_probability))
    rows.append({"model": "seven_model_no_public_assays", **no_public_metrics})
    for branch in BRANCH_ORDER:
        branch_metrics = metrics(y, integrated_branch[branch].to_numpy())
        branch_metrics.update(bootstrap_intervals(y, integrated_branch[branch].to_numpy()))
        rows.append({"model": f"branch__{branch}", **branch_metrics})

    equal_weight_probability = integrated_branch[BRANCH_ORDER].mean(axis=1).to_numpy()
    equal_weight_metrics = metrics(y, equal_weight_probability)
    equal_weight_metrics.update(bootstrap_intervals(y, equal_weight_probability))
    rows.append({"model": "equal_weight_mean_of_six", **equal_weight_metrics})

    assay_ablation_rows = []
    for branch in BRANCH_ORDER:
        with_public = metrics(y, integrated_branch[branch].to_numpy())
        without_public = metrics(y, no_public_branch[branch].to_numpy())
        assay_ablation_rows.append(
            {
                "branch": branch,
                "with_public_assay_roc_auc": with_public["roc_auc"],
                "without_public_assay_roc_auc": without_public["roc_auc"],
                "roc_auc_difference": with_public["roc_auc"] - without_public["roc_auc"],
                "with_public_assay_accuracy": with_public["accuracy"],
                "without_public_assay_accuracy": without_public["accuracy"],
                "accuracy_difference": with_public["accuracy"] - without_public["accuracy"],
            }
        )
    pd.DataFrame(assay_ablation_rows).to_csv(
        args.output_dir / "v4_public_assay_ablation_by_branch.csv", index=False
    )

    four_of_six_prediction = (integrated_branch.ge(0.5).sum(axis=1) >= 4).astype(int).to_numpy()
    four_of_six_score = integrated_branch.mean(axis=1).to_numpy()
    rule_metrics = metrics(
        y,
        four_of_six_score,
        prediction=four_of_six_prediction,
    )
    rule_metrics.update(
        bootstrap_intervals(
            y,
            four_of_six_score,
            prediction=four_of_six_prediction,
        )
    )
    rows.append(
        {
            "model": "proposal_rule_at_least_4_of_6",
            "ranking_score": "mean_of_six_branch_probabilities",
            "decision_rule": "at_least_4_branch_probabilities_ge_0.5",
            **rule_metrics,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "v4_model_metrics.csv", index=False)

    prediction_table = metadata.loc[training_codes, ["chemical_code", "chemical", "label"]].copy()
    prediction_table["seven_model_probability"] = integrated_probability
    prediction_table["seven_model_prediction"] = (integrated_probability >= 0.5).astype(int)
    prediction_table["no_public_assay_probability"] = no_public_probability
    for branch in BRANCH_ORDER:
        prediction_table[f"{branch}__probability"] = integrated_branch[branch].to_numpy()
        prediction_table[f"no_public__{branch}__probability"] = no_public_branch[
            branch
        ].to_numpy()
    prediction_table["positive_branch_count"] = integrated_branch.ge(0.5).sum(axis=1).to_numpy()
    prediction_table["proposal_4_of_6_prediction"] = four_of_six_prediction
    prediction_table.to_csv(args.output_dir / "v4_nested_cv_predictions.csv", index=False)
    pd.DataFrame(integrated_folds + no_public_folds).to_csv(
        args.output_dir / "v4_nested_fold_parameters.csv", index=False
    )

    final_branches = {}
    for number, branch in enumerate(BRANCH_ORDER):
        final_branches[branch] = fit_branch(
            X[no_public_branches[branch]],
            y,
            RANDOM_STATE + 9000 + number,
            groups,
        )
    final_fusion_alphas = {}
    final_training_branch_scores = no_public_branch.copy()
    for branch, public_column in public_score_columns.items():
        candidates = []
        public_values = X[public_column].to_numpy()
        for alpha in (0.0, 0.05, 0.1, 0.2):
            fused = (1 - alpha) * no_public_branch[branch].to_numpy() + alpha * public_values
            candidates.append((roc_auc_score(y, fused), -alpha, alpha))
        alpha = max(candidates)[2]
        final_fusion_alphas[branch] = alpha
        final_training_branch_scores[branch] = (
            (1 - alpha) * no_public_branch[branch].to_numpy() + alpha * public_values
        )
    pd.DataFrame(
        [
            {
                "branch": branch,
                "public_score_column": public_score_columns.get(branch, ""),
                "final_public_fusion_alpha": final_fusion_alphas.get(branch, 0.0),
            }
            for branch in BRANCH_ORDER
        ]
    ).to_csv(args.output_dir / "final_public_fusion_weights.csv", index=False)
    final_meta = fit_meta(
        final_training_branch_scores[BRANCH_ORDER],
        y,
        RANDOM_STATE + 19000,
        groups,
    )
    final_meta_no_public = fit_meta(
        no_public_branch[BRANCH_ORDER],
        y,
        RANDOM_STATE + 29000,
        groups,
    )
    meta_weights = pd.DataFrame(
        {
            "domain": BRANCH_ORDER,
            "coefficient": final_meta.coef_[0],
            "odds_ratio_for_probability_increase_0_to_1": np.exp(final_meta.coef_[0]),
        }
    )
    meta_weights.to_csv(args.output_dir / "final_combiner_weights.csv", index=False)
    pd.DataFrame(
        {
            "domain": BRANCH_ORDER,
            "coefficient": final_meta_no_public.coef_[0],
            "odds_ratio_for_probability_increase_0_to_1": np.exp(
                final_meta_no_public.coef_[0]
            ),
        }
    ).to_csv(args.output_dir / "final_no_public_combiner_weights.csv", index=False)

    benchmark_scores = pd.DataFrame(index=benchmark_codes)
    benchmark_no_public_scores = pd.DataFrame(index=benchmark_codes)
    for branch in BRANCH_ORDER:
        raw_probability = final_branches[branch].predict_proba(
            X_all.loc[benchmark_codes, no_public_branches[branch]]
        )[:, 1]
        benchmark_no_public_scores[branch] = raw_probability
        benchmark_scores[branch] = raw_probability
        if branch in public_score_columns:
            alpha = final_fusion_alphas[branch]
            benchmark_scores[branch] = (
                (1 - alpha) * benchmark_scores[branch]
                + alpha * X_all.loc[benchmark_codes, public_score_columns[branch]]
            )
    benchmark_probability = final_meta.predict_proba(benchmark_scores[BRANCH_ORDER])[:, 1]
    benchmark_no_public_probability = final_meta_no_public.predict_proba(
        benchmark_no_public_scores[BRANCH_ORDER]
    )[:, 1]
    benchmark = metadata.loc[benchmark_codes, ["chemical_code", "chemical", "label"]].copy()
    for branch in BRANCH_ORDER:
        benchmark[f"{branch}__probability"] = benchmark_scores[branch]
    benchmark["seven_model_probability"] = benchmark_probability
    benchmark["seven_model_prediction"] = (benchmark_probability >= 0.5).astype(int)
    benchmark["positive_branch_count"] = benchmark_scores.ge(0.5).sum(axis=1)
    benchmark["no_public_assay_probability"] = benchmark_no_public_probability
    benchmark["no_public_assay_prediction"] = (
        benchmark_no_public_probability >= 0.5
    ).astype(int)
    benchmark.to_csv(args.output_dir / "v4_legacy_benchmark_predictions.csv", index=False)

    model = {
        "model_variant": "integrated_public_assays",
        "branch_order": BRANCH_ORDER,
        "branch_columns": no_public_branches,
        "public_score_columns": public_score_columns,
        "public_fusion_alphas": final_fusion_alphas,
        "branch_models": final_branches,
        "combiner": final_meta,
        "training_oof_branch_scores": integrated_branch,
        "decision_threshold": 0.5,
        "model_version": args.model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "combiner_parameters": {"C": final_meta.C, "selection": "pre-specified"},
        "required_input_kind": "pre-engineered chemical-level feature table",
        "branch_score_semantics": (
            "Biologically constrained scores trained against the shared overall toxic/control "
            "label; they are not calibrated probabilities of domain-specific injury."
        ),
        "validation_group_column": group_column,
        "uncertainty_method": (
            "Class-stratified bootstrap of fixed nested out-of-fold predictions; does not "
            "include full model-refitting or model-selection uncertainty."
        ),
        "proposal_rule": "positive if at least four of six branch probabilities are >= 0.5",
        "scope_warning": (
            "Research prioritization model. Performance is internal chemical-level nested CV on "
            "60 labeled GSE63935 chemicals and is not prospective regulatory validation."
        ),
    }
    integrated_model_path = args.output_dir / "dnt_v4_seven_model_stack.joblib"
    joblib.dump(model, integrated_model_path)
    no_public_model = {
        **model,
        "model_variant": "no_public_assay_transfer",
        "public_score_columns": {},
        "public_fusion_alphas": {},
        "combiner": final_meta_no_public,
        "combiner_parameters": {
            "C": final_meta_no_public.C,
            "selection": "pre-specified",
        },
        "training_oof_branch_scores": no_public_branch,
    }
    no_public_model_path = args.output_dir / "dnt_v4_no_public_stack.joblib"
    joblib.dump(no_public_model, no_public_model_path)

    comparison = []
    if args.v3_metrics and args.v3_metrics.exists():
        v3 = json.loads(args.v3_metrics.read_text())
        source = v3.get("primary_nested_cv", v3)
        comparison.append(
            {
                "version": "V3 full multimodal",
                "roc_auc": source.get("roc_auc"),
                "accuracy": source.get("accuracy"),
                "evaluation": "nested chemical-level CV",
            }
        )
    comparison.append(
        {
            "version": "V4 seven-model integrated",
            "roc_auc": primary_metrics["roc_auc"],
            "accuracy": primary_metrics["accuracy"],
            "evaluation": "nested chemical-level stack CV",
        }
    )
    comparison.append(
        {
            "version": "V4 no-public-assay stack",
            "roc_auc": no_public_metrics["roc_auc"],
            "accuracy": no_public_metrics["accuracy"],
            "evaluation": "nested chemical-level stack CV",
        }
    )
    comparison.append(
        {
            "version": "V4 equal-weight branch mean",
            "roc_auc": equal_weight_metrics["roc_auc"],
            "accuracy": equal_weight_metrics["accuracy"],
            "evaluation": "nested chemical-level branch CV",
        }
    )
    pd.DataFrame(comparison).to_csv(args.output_dir / "v3_v4_comparison.csv", index=False)
    manifest = {
        "model_version": args.model_version,
        "created_at_utc": model["trained_at_utc"],
        "feature_bundle": {
            "file": str(args.feature_bundle),
            "sha256": hashlib.sha256(args.feature_bundle.read_bytes()).hexdigest(),
        },
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": {
            "integrated": {
                "file": integrated_model_path.name,
                "sha256": hashlib.sha256(integrated_model_path.read_bytes()).hexdigest(),
                "nested_cv_roc_auc": primary_metrics["roc_auc"],
                "nested_cv_accuracy": primary_metrics["accuracy"],
            },
            "no_public_assay_transfer": {
                "file": no_public_model_path.name,
                "sha256": hashlib.sha256(no_public_model_path.read_bytes()).hexdigest(),
                "nested_cv_roc_auc": no_public_metrics["roc_auc"],
                "nested_cv_accuracy": no_public_metrics["accuracy"],
            },
        },
        "preferred_internal_candidate": "no_public_assay_transfer",
        "candidate_selection_note": (
            "Marginally higher nested ROC AUC and accuracy with lower complexity; difference "
            "is not an independent external comparison."
        ),
        "python_version": platform.python_version(),
        "scikit_learn_version": SKLEARN_VERSION,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "training_chemicals": int(len(training_codes)),
        "legacy_benchmark_chemicals": int(len(benchmark_codes)),
        "validation_group_column": group_column,
        "branch_order": BRANCH_ORDER,
        "decision_threshold": model["decision_threshold"],
        "scope_warning": model["scope_warning"],
    }
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if not args.skip_plots:
        save_plots(args.output_dir, y, integrated_probability, integrated_branch)
    print(summary[["model", "roc_auc", "accuracy", "balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
