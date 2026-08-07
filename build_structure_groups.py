#!/usr/bin/env python3
"""Create analogue-safe structure groups from SMILES using Morgan fingerprints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise SystemExit(
        "RDKit is required. Install the project requirements before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.similarity_threshold < 1.0:
        raise ValueError("--similarity-threshold must be between 0 and 1")
    table = pd.read_csv(args.descriptors)
    smiles_column = next(
        (column for column in ("ConnectivitySMILES", "CanonicalSMILES", "SMILES") if column in table),
        None,
    )
    if "chemical_code" not in table or smiles_column is None:
        raise ValueError(
            "Descriptor CSV must contain chemical_code and a SMILES, CanonicalSMILES, "
            "or ConnectivitySMILES column"
        )
    if table["chemical_code"].duplicated().any():
        raise ValueError("Descriptor CSV contains duplicate chemical_code values")

    fingerprints = []
    valid_rows = []
    invalid_rows = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row in table.itertuples(index=False):
        code = str(getattr(row, "chemical_code"))
        smiles = getattr(row, smiles_column)
        molecule = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        if molecule is None:
            invalid_rows.append((code, smiles))
            continue
        fingerprints.append(generator.GetFingerprint(molecule))
        valid_rows.append((code, smiles))

    distances = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        1.0 - args.similarity_threshold,
        isDistData=True,
    )
    assignment = {}
    for cluster_number, members in enumerate(clusters, start=1):
        for member in members:
            assignment[valid_rows[member][0]] = f"structure_{cluster_number:03d}"
    # Unresolved structures receive unique groups: they cannot accidentally bridge folds.
    for number, (code, _) in enumerate(invalid_rows, start=1):
        assignment[code] = f"unresolved_{number:03d}"

    result = table[["chemical_code"]].copy()
    result["structure_group"] = result["chemical_code"].astype(str).map(assignment)
    result["structure_similarity_threshold"] = args.similarity_threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Saved {len(result)} chemicals in {result.structure_group.nunique()} groups "
        f"to {args.output}; unresolved structures: {len(invalid_rows)}"
    )


if __name__ == "__main__":
    main()
