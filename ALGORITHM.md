# Algorithm

## Prediction target

The system predicts the chemical-level overall toxic/control label from GSE63935-derived features.
One chemical is one observation; biological replicates and time points are summarized before
validation.

## Branch models

Each biological branch receives a predeclared subset of RNA marker, pathway, developmental,
chemical-context, dose, or external-prior features. Within every training fold it performs:

1. Median imputation fitted on the training fold.
2. Removal of zero-variance features fitted on the training fold.
3. ANOVA F-test feature selection fitted on the training fold.
4. Standardization fitted on the training fold.
5. Class-balanced elastic-net logistic regression.

The search evaluates `C ∈ {0.01, 0.1, 1.0}`, L2 and L1 endpoints, and branch-dependent feature
percentiles. Hyperparameters are selected only from the relevant training partition.

## Public transfer and fusion

Four Extra Trees models learn EPA domain activity from eight PubChem descriptors. Their
cross-validated performance is reported independently. For a GSE chemical, a direct assay match
is blended with the transferred score when available. An all-missing descriptor row receives
neutral transferred evidence (`0.5`).

For eligible branch `j`:

```text
fused_j = (1 - alpha_j) * RNA_branch_j + alpha_j * public_score_j
alpha_j ∈ {0, 0.05, 0.10, 0.20}
```

Alpha selection occurs inside each outer-training fold. The held-out fold is never used to choose
its fusion weight.

## Stacking

Inside every outer-training fold, four-fold out-of-fold branch predictions train the combiner.
The held-out outer fold receives predictions from branch models fitted only on outer-training
chemicals. The combiner is class-balanced logistic regression with pre-specified `C=0.5`.

```text
final_probability = sigmoid(intercept + sum(beta_j * fused_j))
final_prediction = final_probability >= 0.5
```

Combiner regularization is not tuned because doing so with roughly 48 outer-training chemicals
caused unstable coefficients. An equal-weight mean is reported as a transparent baseline.

## Validation

The outer evaluation uses five folds. Hyperparameter search, stacking inputs, and public-fusion
selection are nested within the outer-training data. If metadata contains `validation_group` or
`structure_group`, every validation level uses `StratifiedGroupKFold` and verifies that groups do
not cross train/test boundaries. Without groups, the code warns and uses chemical-level
`StratifiedKFold`.

Confidence intervals use class-stratified bootstrap resampling of the fixed nested out-of-fold
predictions. They capture chemical-sampling uncertainty conditional on those predictions, not all
uncertainty from refitting the entire system.

## Prediction contract

`predict_v4.py` requires every fitted branch feature plus any public score required by the selected
artifact. Required columns must be numeric; public scores must be finite and within `[0, 1]`.
Median imputation handles ordinary missing branch features. The no-public artifact does not require
public-score columns.
