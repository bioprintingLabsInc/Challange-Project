# Model status

## Active development version

The corrected training pipeline targets model version 4.1. It adds effective L1/L2 tuning,
a pre-specified regularized combiner, optional structure-group-aware nested validation, explicit
conditional-bootstrap reporting, input validation, and machine-readable artifact provenance.

Combiner regularization is intentionally fixed at `C=0.5`. Tuning a six-input combiner inside
48-chemical outer-training folds was empirically unstable and produced extreme coefficients.
The equal-weight mean is reported as a transparent ensemble baseline.

## Interpretation boundary

All six biological branches are trained against the same overall chemical toxic/control label.
They are biologically constrained views of that target, not calibrated probabilities of six
independently measured injuries.

## Validation levels

- **Chemical-level nested CV:** available when no structure groups are supplied. It prevents the
  same chemical from crossing folds but does not guarantee separation of close analogues.
- **Structure-group nested CV:** used automatically when `validation_group` or `structure_group`
  is present in feature-bundle metadata. This is the preferred evaluation.
- **Prospective external validation:** not yet available and required before regulatory use.

Every generated model includes `validation_group_column`, `uncertainty_method`, and
`scope_warning` metadata. `output/model_manifest.json` records the artifact checksum and runtime
versions after training.
