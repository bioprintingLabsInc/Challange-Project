# DNT seven-model research system, Version 4.1

This repository contains a chemical-level developmental-neurotoxicity (DNT) research model.
Six biologically constrained classifiers produce branch scores, and a seventh regularized
logistic-regression model combines them.

## Status and scope

Version 4.1 is an internal research-prioritization model. It is not a regulatory, clinical, or
diagnostic test. All branches are trained against the same overall toxic/control label; branch
scores are not calibrated probabilities of domain-specific injury.

The preferred internal candidate is currently the **no-public-assay stack** because it is simpler
and was marginally better in the same nested evaluation. The integrated candidate is retained
because the difference is small and public functional evidence may prove useful externally.

| Candidate | Nested ROC AUC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| No-public-assay stack | 0.876 | 75.0% | 85.3% | 61.5% |
| Integrated public-assay stack | 0.870 | 73.3% | 82.4% | 61.5% |
| Equal-weight six-branch mean | 0.856 | 75.0% | 73.5% | 76.9% |

These estimates use nested chemical-level cross-validation on 60 chemicals. The current feature
bundle does not contain structure groups, so close analogues may cross fold boundaries. See
[`MODEL_STATUS.md`](MODEL_STATUS.md) for the validation boundary.

## Architecture

1. Growth and proliferation
2. Progenitor development and cortical organization
3. Excitatory/inhibitory neuronal balance
4. Synaptic integrity
5. Electrophysiology
6. Transcriptomic risk
7. Pre-specified regularized logistic combiner over the six branch scores

See [`ALGORITHM.md`](ALGORITHM.md) for the full data flow and leakage controls.

## Main artifacts

- `output/dnt_v4_no_public_stack.joblib` — preferred internal candidate.
- `output/dnt_v4_seven_model_stack.joblib` — integrated public-assay candidate.
- `output/model_manifest.json` — hashes, runtime versions, and validation metadata.
- `data/processed/v4_feature_bundle.joblib` — exact feature bundle used for training.
- `data/processed/v4_features_for_prediction.csv.gz` — portable compatible prediction input.

Only load trusted Joblib files. Joblib deserialization can execute arbitrary code.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Predict

```bash
python predict_v4.py \
  --model output/dnt_v4_no_public_stack.joblib \
  --features data/processed/v4_features_for_prediction.csv.gz \
  --output predictions.csv
```

The prediction command requires a pre-engineered chemical-level feature table. It does not create
RNA, pathway, developmental, or public-assay features from a chemical name or SMILES alone.

## Retrain

```bash
./run_training.sh
```

For analogue-separated evaluation, first create a `chemical_code,structure_group` CSV with
`build_structure_groups.py`, then rebuild the feature bundle with `prepare_v4_data.py
--structure-groups ...` before training.

## Documentation

- [`ALGORITHM.md`](ALGORITHM.md) — architecture, equations, validation, and prediction contract.
- [`DATASETS.md`](DATASETS.md) — dataset roles and provenance.
- [`MODEL_REPORT.md`](MODEL_REPORT.md) — current results and limitations.
- [`FILE_GUIDE.md`](FILE_GUIDE.md) — file-by-file reference.
- [`MODEL_STATUS.md`](MODEL_STATUS.md) — release status and unresolved scientific work.
