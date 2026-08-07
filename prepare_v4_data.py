#!/usr/bin/env python3
"""Build six-domain features from V3 RNA features and independent public assays."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
DESCRIPTORS = [
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "Complexity",
    "Charge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-bundle", type=Path, required=True)
    parser.add_argument("--v3-descriptors", type=Path, required=True)
    parser.add_argument("--carstens", type=Path, required=True)
    parser.add_argument("--carstens-descriptors", type=Path, required=True)
    parser.add_argument("--cohn", type=Path, required=True)
    parser.add_argument("--expression-2d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structure-groups",
        type=Path,
        help="Optional CSV with chemical_code and structure_group columns",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def domain_endpoints(columns: list[str]) -> dict[str, list[str]]:
    return {
        "growth_proliferation": [column for column in columns if "hNP1_Pro_" in column],
        "progenitor_organization": [column for column in columns if "_NOG_" in column],
        "synaptic_integrity": [
            column for column in columns if "Synap&Neur_Matur" in column
        ],
        "electrophysiology": [
            column
            for column in columns
            if "MEA_dev_" in column and "_AB_" not in column and "_LDH_" not in column
        ],
    }


def public_surrogate() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=700,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def public_scores(
    carstens: pd.DataFrame,
    descriptors: pd.DataFrame,
    gse_descriptors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    if descriptors["casn"].duplicated().any():
        raise ValueError("Carstens descriptor table contains duplicate CAS numbers")
    if gse_descriptors["chemical_code"].duplicated().any():
        raise ValueError("GSE descriptor table contains duplicate chemical_code values")
    missing_descriptor_columns = [
        column
        for column in ["chemical_code", *DESCRIPTORS]
        if column not in gse_descriptors
    ]
    if missing_descriptor_columns:
        raise ValueError(
            f"GSE descriptor table is missing columns: {missing_descriptor_columns}"
        )
    panel = carstens.merge(
        descriptors, on="casn", how="left", suffixes=("", "_descriptor"), validate="many_to_one"
    )
    endpoint_map = domain_endpoints(list(carstens.columns))
    empty_domains = [domain for domain, endpoints in endpoint_map.items() if not endpoints]
    if empty_domains:
        raise ValueError(f"No Carstens assay endpoints found for domains: {empty_domains}")
    gse_input = gse_descriptors.set_index("chemical_code")[DESCRIPTORS]
    gse_has_descriptor = gse_input.notna().any(axis=1)
    result = pd.DataFrame(index=gse_input.index)
    metric_rows = []
    target_rows = panel[["chnm", "casn", "dsstox_substance_id"]].copy()
    fitted = {}
    for domain, endpoints in endpoint_map.items():
        y = panel[endpoints].notna().any(axis=1).astype(int).to_numpy()
        target_rows[f"{domain}__active"] = y
        target_rows[f"{domain}__endpoint_fraction_active"] = (
            panel[endpoints].notna().mean(axis=1).to_numpy()
        )
        usable = panel[DESCRIPTORS].notna().any(axis=1).to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        estimator = public_surrogate()
        probability = cross_val_predict(
            estimator,
            panel.loc[usable, DESCRIPTORS],
            y[usable],
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metric_rows.append(
            {
                "domain": domain,
                "panel_n": int(usable.sum()),
                "active_n": int(y[usable].sum()),
                "inactive_n": int((1 - y[usable]).sum()),
                "descriptor_surrogate_cv_roc_auc": roc_auc_score(y[usable], probability),
                "descriptor_surrogate_cv_balanced_accuracy": balanced_accuracy_score(
                    y[usable], prediction
                ),
                "endpoint_count": len(endpoints),
            }
        )
        estimator.fit(panel.loc[usable, DESCRIPTORS], y[usable])
        fitted[domain] = estimator
        transferred = estimator.predict_proba(gse_input)[:, 1]
        # An all-missing descriptor row should contribute neutral evidence instead of the
        # imputer-induced probability of an imaginary median chemical.
        transferred[~gse_has_descriptor.to_numpy()] = 0.5
        result[f"public__{domain}__surrogate_probability"] = transferred
        result[f"public__{domain}__descriptor_available"] = gse_has_descriptor.astype(int)
    return result, pd.DataFrame(metric_rows), target_rows, fitted


def two_d_prior_features(expression_path: Path, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = pd.read_csv(expression_path, index_col=0)
    numeric = expression.select_dtypes(include=[np.number])
    variance = numeric.var(axis=0).sort_values(ascending=False)
    available_symbols = {column.rsplit("__", 1)[-1] for column in X if column.startswith("gene__")}
    ranked = [gene for gene in variance.index if gene in available_symbols]
    selected = ranked[:250]
    rows = pd.DataFrame(
        {
            "gene": variance.index,
            "variance_2d_tpm": variance.values,
            "present_in_v3_candidate_features": [gene in available_symbols for gene in variance.index],
            "selected_top_250": [gene in selected for gene in variance.index],
        }
    )
    features = pd.DataFrame(index=X.index)
    for timepoint in ("d16", "d21", "delta"):
        columns = [f"gene__{timepoint}__{gene}" for gene in selected]
        columns = [column for column in columns if column in X]
        values = X[columns]
        features[f"external2d__{timepoint}__signed_mean"] = values.mean(axis=1)
        features[f"external2d__{timepoint}__absolute_mean"] = values.abs().mean(axis=1)
        features[f"external2d__{timepoint}__fraction_abs_gt_0p5"] = (
            values.abs().gt(0.5).mean(axis=1)
        )
    return features, rows


def marker_columns(X: pd.DataFrame, genes: list[str]) -> list[str]:
    wanted = set(genes)
    return [
        column
        for column in X
        if column.startswith("gene__") and column.rsplit("__", 1)[-1] in wanted
    ]


def path_columns(X: pd.DataFrame, pathways: list[str]) -> list[str]:
    prefixes = tuple(f"path__{name}__" for name in pathways)
    return [column for column in X if column.startswith(prefixes)]


def branch_definitions(X: pd.DataFrame) -> dict[str, list[str]]:
    development = [column for column in X if column.startswith("development__")]
    chemical = [column for column in X if column.startswith("chemical__")]
    external2d = [column for column in X if column.startswith("external2d__")]
    public = lambda domain: [f"public__{domain}__integrated_score"]
    return {
        "growth_proliferation": path_columns(
            X, ["cell_cycle", "apoptotic_process", "dna_damage_response", "oxidative_stress"]
        )
        + marker_columns(
            X,
            ["MKI67", "PCNA", "TOP2A", "CCNB1", "CDK1", "SOX2", "NES", "BAX", "BCL2"],
        )
        + chemical
        + public("growth_proliferation"),
        "progenitor_organization": path_columns(
            X,
            [
                "nervous_system_development",
                "neurogenesis",
                "neuron_differentiation",
                "neuron_migration",
                "gliogenesis",
                "myelination",
            ],
        )
        + marker_columns(
            X,
            ["SOX2", "PAX6", "EOMES", "BCL11B", "SATB2", "FOXG1", "TBR1", "NES"],
        )
        + development
        + chemical
        + public("progenitor_organization"),
        "excitatory_inhibitory_balance": marker_columns(
            X,
            [
                "SLC17A7",
                "SLC17A6",
                "CAMK2A",
                "SATB2",
                "TBR1",
                "GAD1",
                "GAD2",
                "SLC32A1",
                "DLX1",
                "DLX2",
            ],
        )
        + path_columns(X, ["neuron_differentiation", "neuron_migration"]),
        "synaptic_integrity": path_columns(
            X, ["synapse_organization", "axonogenesis", "dendrite_development"]
        )
        + marker_columns(X, ["SYN1", "DLG4", "SYP", "SNAP25", "GRIN1", "PSD95"])
        + development
        + chemical
        + public("synaptic_integrity"),
        "electrophysiology": path_columns(
            X, ["synapse_organization", "nervous_system_development", "neuron_differentiation"]
        )
        + marker_columns(
            X,
            [
                "SCN1A",
                "SCN2A",
                "CACNA1C",
                "KCNQ2",
                "GRIN1",
                "GRIA1",
                "GABRA1",
                "SNAP25",
            ],
        )
        + chemical
        + public("electrophysiology"),
        "transcriptomic_risk": [column for column in X if column.startswith("gene__")]
        + [column for column in X if column.startswith("path__")]
        + development
        + external2d,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.v3_bundle)
    X = bundle["X"].copy()
    metadata = bundle["metadata"].copy()
    if not X.index.is_unique or metadata["chemical_code"].duplicated().any():
        raise ValueError("Chemical codes must be unique in features and metadata")
    metadata_codes = pd.Index(metadata["chemical_code"])
    if set(X.index) != set(metadata_codes):
        raise ValueError("V3 feature and metadata chemical codes do not match")
    metadata = metadata.set_index("chemical_code", drop=False).loc[X.index].reset_index(drop=True)
    if args.structure_groups:
        groups = pd.read_csv(args.structure_groups)
        required_group_columns = {"chemical_code", "structure_group"}
        if not required_group_columns.issubset(groups.columns):
            raise ValueError(
                "Structure-group CSV must contain chemical_code and structure_group columns"
            )
        if groups["chemical_code"].duplicated().any():
            raise ValueError("Structure-group CSV contains duplicate chemical_code values")
        metadata = metadata.merge(
            groups[["chemical_code", "structure_group"]],
            on="chemical_code",
            how="left",
            validate="one_to_one",
        )
        training_missing = metadata.loc[
            metadata["split"].eq("training"), "structure_group"
        ].isna()
        if training_missing.any():
            raise ValueError(
                f"Structure groups are missing for {int(training_missing.sum())} training chemicals"
            )
    carstens = pd.read_excel(args.carstens)
    carstens_descriptors = pd.read_csv(args.carstens_descriptors)
    gse_descriptors = pd.read_csv(args.v3_descriptors)

    surrogate, surrogate_metrics, panel_targets, fitted = public_scores(
        carstens, carstens_descriptors, gse_descriptors
    )
    carstens_lookup = {
        normalize_name(row["chnm"]): row for _, row in panel_targets.iterrows()
    }
    match_rows = []
    for domain in (
        "growth_proliferation",
        "progenitor_organization",
        "synaptic_integrity",
        "electrophysiology",
    ):
        integrated = surrogate[f"public__{domain}__surrogate_probability"].copy()
        measured = pd.Series(np.nan, index=integrated.index)
        for row in metadata.itertuples(index=False):
            match = carstens_lookup.get(normalize_name(row.chemical))
            if match is not None:
                measured.loc[row.chemical_code] = match[f"{domain}__active"]
                match_rows.append(
                    {
                        "chemical_code": row.chemical_code,
                        "chemical": row.chemical,
                        "source": "Carstens integrated DNT-NAM panel",
                        "domain": domain,
                        "measured_active": match[f"{domain}__active"],
                    }
                )
        has_measurement = measured.notna()
        integrated.loc[has_measurement] = (
            integrated.loc[has_measurement] + measured.loc[has_measurement]
        ) / 2
        X[f"public__{domain}__integrated_score"] = integrated.reindex(X.index)
        surrogate[f"public__{domain}__measured_available"] = has_measurement.astype(int)

    cohn = pd.read_excel(args.cohn, sheet_name="Primary Screening Data", header=4)
    cohn_lookup = {
        normalize_name(row["Chemical Name"]): row for _, row in cohn.dropna(subset=["Chemical Name"]).iterrows()
    }
    cohn_rows = []
    for row in metadata.itertuples(index=False):
        match = cohn_lookup.get(normalize_name(row.chemical))
        if match is None:
            continue
        o1 = float(match["O1 Normalized to DMSO"])
        viability = float(match["Viability"])
        usable = viability >= 0.75 and o1 > 0
        disruption = float(min(abs(np.log2(max(o1, 0.05))) / 2.0, 1.0)) if usable else np.nan
        cohn_rows.append(
            {
                "chemical_code": row.chemical_code,
                "chemical": row.chemical,
                "casn": match["CASN"],
                "oligodendrocyte_o1_normalized": o1,
                "viability": viability,
                "usable_noncytotoxic": usable,
                "progenitor_disruption_score": disruption,
            }
        )
        if usable:
            column = "public__progenitor_organization__integrated_score"
            X.loc[row.chemical_code, column] = (X.loc[row.chemical_code, column] + disruption) / 2
            match_rows.append(
                {
                    "chemical_code": row.chemical_code,
                    "chemical": row.chemical,
                    "source": "Cohn oligodendrocyte primary screen",
                    "domain": "progenitor_organization",
                    "measured_active": disruption,
                }
            )

    two_d_features, two_d_gene_prior = two_d_prior_features(args.expression_2d, X)
    X = X.join(two_d_features)
    branches = branch_definitions(X)
    for domain, columns in branches.items():
        if not columns:
            raise RuntimeError(f"No features for {domain}")
        missing = [column for column in columns if column not in X]
        if missing:
            raise RuntimeError(f"Missing features for {domain}: {missing[:5]}")

    output_bundle = {
        "X": X,
        "metadata": metadata,
        "branch_columns": branches,
        "target_semantics": (
            "Every branch is trained against the same chemical-level overall toxic/control "
            "label. Branch outputs are not domain-specific ground-truth probabilities."
        ),
        "source_note": (
            "Six domain feature sets combining GSE63935 RNA effects, pathway and developmental "
            "features, GSE126786 2D expression variability, EPA Carstens DNT-NAM assay transfer "
            "scores, and non-cytotoxic Cohn oligodendrocyte measurements."
        ),
    }
    joblib.dump(output_bundle, args.output_dir / "v4_feature_bundle.joblib")
    portable = X.copy()
    portable.insert(0, "chemical_code", portable.index)
    name_lookup = metadata.set_index("chemical_code")["chemical"]
    portable.insert(1, "chemical", portable["chemical_code"].map(name_lookup))
    portable.to_csv(
        args.output_dir / "v4_features_for_prediction.csv.gz",
        index=False,
        compression="gzip",
    )
    joblib.dump(fitted, args.output_dir / "public_assay_surrogate_models.joblib")
    surrogate.to_csv(args.output_dir / "public_domain_scores_by_gse_chemical.csv")
    surrogate_metrics.to_csv(args.output_dir / "public_assay_surrogate_metrics.csv", index=False)
    panel_targets.to_csv(args.output_dir / "carstens_domain_targets.csv", index=False)
    pd.DataFrame(match_rows).to_csv(args.output_dir / "public_assay_matches.csv", index=False)
    pd.DataFrame(cohn_rows).to_csv(args.output_dir / "cohn_matches.csv", index=False)
    two_d_gene_prior.to_csv(args.output_dir / "gse126786_2d_gene_variability_prior.csv", index=False)
    pd.DataFrame(
        [
            {"domain": domain, "feature_count": len(columns), "features": json.dumps(columns)}
            for domain, columns in branches.items()
        ]
    ).to_csv(args.output_dir / "branch_feature_manifest.csv", index=False)
    print("Prepared", X.shape, "with branch sizes", {k: len(v) for k, v in branches.items()})


if __name__ == "__main__":
    main()
