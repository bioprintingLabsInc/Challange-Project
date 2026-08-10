# File guide

## Active Version 4.1 artifacts

- `output/dnt_v4_no_public_stack.joblib` — preferred simpler internal candidate.
- `output/dnt_v4_seven_model_stack.joblib` — integrated public-assay candidate.
- `output/model_manifest.json` — artifact hashes, runtime versions, sample counts, and scope.
- `output/v4_model_metrics.csv` — nested metrics and conditional bootstrap intervals.
- `output/v4_nested_cv_predictions.csv` — integrated and no-public out-of-fold predictions.
- `output/v4_nested_fold_parameters.csv` — fold-specific branch parameters and fusion weights.
- `output/final_combiner_weights.csv` — integrated combiner coefficients.
- `output/final_no_public_combiner_weights.csv` — no-public combiner coefficients.
- `output/final_public_fusion_weights.csv` — final fitted public fusion weights.
- `output/v4_public_assay_ablation_by_branch.csv` — branch-level public-data comparison.
- `output/v4_shortcut_baseline_metrics.csv` — nested dose/descriptor shortcut audit.
- `output/v4_legacy_benchmark_predictions.csv` — continuity-check predictions.

`output_v4_0_historical/` preserves the original Version 4.0 outputs. Do not mix its metrics or
plots with Version 4.1 results.

## Processed data

- `data/processed/v4_feature_bundle.joblib` — exact 70-row feature matrix, metadata, and branch
  definitions used by training.
- `data/processed/v4_features_for_prediction.csv.gz` — portable compatible input table.
- `data/processed/public_assay_surrogate_models.joblib` — four fitted descriptor-transfer models.
- `data/processed/public_assay_surrogate_metrics.csv` — independent transfer-model validation.
- `data/processed/branch_feature_manifest.csv` — branch-to-feature mapping.
- `data/processed/carstens_domain_targets.csv` — derived EPA domain targets.
- `data/processed/public_domain_scores_by_gse_chemical.csv` — transferred public scores.
- `data/processed/public_assay_matches.csv` and `cohn_matches.csv` — direct-match audits.
- `data/processed/gse126786_2d_gene_variability_prior.csv` — label-free gene ranking.

## Code

- `prepare_v4_data.py` — builds the feature bundle and portable prediction table.
- `train_v4_seven_models.py` — nested validation, ablations, final fitting, and provenance.
- `predict_v4.py` — validates a compatible feature table and applies either model artifact.
- `fetch_public_descriptors.py` — retrieves and caches auditable PubChem descriptors.
- `build_structure_groups.py` — clusters Morgan fingerprints for analogue-safe validation.
- `audit_shortcuts.py` — tests how much label signal exists in dose/descriptors without RNA.
- `run_training.sh` — reproducible thread-limited Version 4.1 training command.
- `tests/test_pipeline.py` — leakage, metric, definition, and artifact-integrity tests.

## Documentation

- `README.md` — project entry point.
- `ALGORITHM.md` — technical model and validation description.
- `DATASETS.md` and `DATASET_GUIDE.md` — dataset overview and detailed provenance.
- `MODEL_REPORT.md` — current results.
- `MODEL_STATUS.md` — interpretation boundary and remaining work.
