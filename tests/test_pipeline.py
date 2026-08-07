from __future__ import annotations

import importlib.util
import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train = load_module("train_v4", "train_v4_seven_models.py")
prepare = load_module("prepare_v4", "prepare_v4_data.py")


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.X = pd.DataFrame(rng.normal(size=(40, 8)))
        self.y = np.tile([0, 1, 0, 1], 10)
        self.groups = np.repeat(np.arange(10), 4)

    def test_group_splits_do_not_leak(self) -> None:
        splits = train.validation_splits(self.X, self.y, 5, 42, self.groups)
        for training, testing in splits:
            self.assertTrue(
                set(self.groups[training]).isdisjoint(self.groups[testing])
            )

    def test_vote_metrics_use_explicit_decisions(self) -> None:
        probability = np.full(len(self.y), 0.5)
        result = train.metrics(self.y, probability, prediction=self.y.copy())
        self.assertEqual(result["accuracy"], 1.0)

    def test_bootstrap_names_are_explicit(self) -> None:
        result = train.bootstrap_intervals(
            self.y, np.linspace(0.0, 1.0, len(self.y)), iterations=50
        )
        self.assertTrue(
            all(name.startswith("conditional_oof_") for name in result)
        )


class DataDefinitionTests(unittest.TestCase):
    def test_expected_domains_exist(self) -> None:
        columns = [
            "x_hNP1_Pro_a",
            "x_NOG_a",
            "x_Synap&Neur_Matur_a",
            "x_MEA_dev_a",
            "x_MEA_dev_AB_a",
            "x_MEA_dev_LDH_a",
        ]
        endpoints = prepare.domain_endpoints(columns)
        self.assertEqual(set(endpoints), set(train.BRANCH_ORDER) - {
            "excitatory_inhibitory_balance",
            "transcriptomic_risk",
        })
        self.assertEqual(endpoints["electrophysiology"], ["x_MEA_dev_a"])


class ArtifactTests(unittest.TestCase):
    def test_processed_bundle_integrity(self) -> None:
        bundle = joblib.load(ROOT / "data/processed/v4_feature_bundle.joblib")
        X = bundle["X"]
        metadata = bundle["metadata"].set_index("chemical_code", drop=False)
        self.assertTrue(X.index.is_unique)
        self.assertTrue(X.columns.is_unique)
        self.assertTrue(X.index.equals(metadata.index))
        self.assertEqual(set(bundle["branch_columns"]), set(train.BRANCH_ORDER))

    def test_model_artifact_hashes_and_metadata(self) -> None:
        manifest = json.loads((ROOT / "output/model_manifest.json").read_text())
        bundle_path = ROOT / manifest["feature_bundle"]["file"]
        self.assertEqual(
            hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            manifest["feature_bundle"]["sha256"],
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "train_v4_seven_models.py").read_bytes()).hexdigest(),
            manifest["training_script_sha256"],
        )
        for variant, information in manifest["artifacts"].items():
            path = ROOT / "output" / information["file"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), information["sha256"]
            )
            model = joblib.load(path)
            self.assertEqual(model["model_version"], manifest["model_version"])
            self.assertTrue(model["model_variant"])


if __name__ == "__main__":
    unittest.main()
