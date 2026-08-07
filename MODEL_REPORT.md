# Version 4.1 model report

## Bottom line

The corrected Version 4.1 pipeline produced two complete candidates. On the same 60-chemical
nested evaluation, the no-public-assay model reached ROC AUC 0.876 and 75.0% accuracy; the
integrated public-assay model reached ROC AUC 0.870 and 73.3% accuracy. The differences are too
small and uncertain to establish superiority, but the no-public model is the simpler internal
default.

| Model | ROC AUC | PR AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| No-public-assay stack | 0.876 | 0.909 | 75.0% | 73.4% | 85.3% | 61.5% |
| Integrated stack | 0.870 | 0.906 | 73.3% | 71.9% | 82.4% | 61.5% |
| Equal-weight branch mean | 0.856 | 0.905 | 75.0% | 75.2% | 73.5% | 76.9% |
| Four-of-six decision rule | 0.856* | 0.905* | 71.7% | 72.7% | 64.7% | 80.8% |

`*` Ranking metrics use the mean branch score; classification metrics use the four-of-six vote.

The no-public stack classified 45 of 60 out-of-fold chemicals correctly: 29 true positives, 16
true negatives, 10 false positives, and 5 false negatives. Its conditional out-of-fold bootstrap
intervals were 0.779–0.948 for ROC AUC and 63.3%–85.0% for accuracy.

## What changed from Version 4.0

- L1/L2 branch regularization is now genuinely effective and version-compatible.
- Public fusion may select 0%; external evidence is no longer forced into a branch.
- Combiner regularization is fixed at `C=0.5` because fold-level tuning was unstable at this
  sample size.
- Structure-group-aware splitting is supported and checked for leakage.
- Conditional bootstrap intervals preserve class counts and are labeled honestly.
- The equal-weight ensemble and vote rule are reported separately.
- Integrated and no-public fitted models are both saved and hashed.
- Prediction inputs, bundle alignment, public-score ranges, and outputs are validated.

## Branch results

| Branch | ROC AUC | Accuracy |
|---|---:|---:|
| Growth/proliferation | 0.775 | 73.3% |
| Progenitor/organization | 0.698 | 66.7% |
| E/I balance | 0.708 | 61.7% |
| Synaptic integrity | 0.871 | 78.3% |
| Electrophysiology | 0.840 | 80.0% |
| Transcriptomic risk | 0.762 | 71.7% |

These are six views of the same overall labels, not six independent domain validation studies.

## Public assay evidence

Descriptor-transfer models supply candidate evidence for proliferation, progenitor/neurite,
synaptic, and electrophysiology branches. Within each outer-training fold, late fusion selects
0%, 5%, 10%, or 20% using only outer-training predictions. The public features did not improve
the overall nested result in Version 4.1. Both model variants are therefore retained.

## Shortcut audit

Dose and physicochemical features were evaluated without RNA, pathways, or developmental
features using the same nested chemical-level design.

| Baseline | ROC AUC | Accuracy |
|---|---:|---:|
| Chemical context (dose + descriptors) | 0.683 | 55.0% |
| PubChem descriptors only | 0.578 | 56.7% |
| Dose only | 0.697 | 66.7% |

The complete model materially exceeds these baselines, but dose alone contains nontrivial label
signal. This may reflect genuine concentration-dependent risk, study-design differences, or both.
Future external validation should match or stratify dose distributions and report an RNA-only
ablation.

## Legacy benchmark

Both candidates classified 9 of 10 legacy benchmark chemicals correctly, with oleic acid as the
false positive. This is a continuity check, not external validation, because these chemicals were
reviewed during earlier development.

## Remaining scientific limitations

- Only 60 chemicals contribute training labels.
- The current evaluation is chemical-separated but not structure-group-separated.
- Every branch uses the same overall label rather than matched domain-specific outcomes.
- The 10 legacy chemicals are not an untouched test set.
- Public transfer uses a small physicochemical descriptor set.
- Dose contains appreciable target signal and may partly encode study design.
- Probabilities are not externally calibrated.
- Conditional bootstrap intervals do not include complete model-selection uncertainty.

A prospective, scaffold-separated external set with functional domain outcomes is required before
interpreting these scores as absolute risk or using them for regulatory decisions.
