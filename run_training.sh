#!/usr/bin/env bash
set -euo pipefail

# Limit native numerical libraries for reproducible, portable nested fitting.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dnt_v4_matplotlib"

python3 train_v4_seven_models.py \
  --feature-bundle data/processed/v4_feature_bundle.joblib \
  --output-dir output \
  --model-version 4.1 \
  --skip-plots

python3 audit_shortcuts.py \
  --feature-bundle data/processed/v4_feature_bundle.joblib \
  --output-dir output
