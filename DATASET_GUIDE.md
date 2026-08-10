# Dataset guide

This guide explains every public data source considered for Version 4, what it contains, what entered the model, and what was not usable. Overlapping releases are listed separately for traceability but are not double counted.

## GSE63935 / Schwartz neural-tissue RNA data

- **Contains:** Human neural-tissue gene-expression measurements after chemical exposure, with toxic/reference-control chemical assignments and multiple developmental time points. The recovered Version 3 bundle contains 70 chemical-level rows: 60 used for training and 10 retained as a legacy benchmark.
- **Used:** Gene-level expression effects, pathway scores, developmental-trajectory features, chemical dose, and PubChem descriptors. These are the main features from which the six domain models are built.
- **Not used:** Individual RNA samples are not treated as independent examples. Replicates and time points are summarized at the chemical level so the same chemical cannot be split between training and testing.
- **Limitation:** The class labels are chemical-level study labels, not direct measurements of each of the six proposal domains.
- **Design caution:** Dose alone carries measurable label signal in the current panel. This may
  represent both concentration-dependent biology and differences in how toxic and control
  chemicals were selected or tested; dose-matched external validation is needed.
- **Source:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63935

## GSE126786 / DevTox2D

- **Contains:** TPM expression profiles from a 2D human neural model for 39 labeled chemicals (29 toxic-class and 10 control-class chemicals), with multiple exposure durations.
- **Used:** A label-free ranking of genes by expression variance. The top 250 genes that also occur in the Version 3 candidate set are summarized in the transcriptomic branch. This expands the transcriptomic evidence beyond the 3D system without treating repeated chemical identities as new independent labels.
- **Not used:** The toxic/control labels and sample-presence indicators are not appended as features. Many chemicals overlap GSE63935, so doing that would create label leakage and inflate apparent sample size.
- **Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7075697/ and https://github.com/finnkuusisto/DevTox2D

## BrainSpan developmental human-brain reference

- **Contains:** Developmental human-brain expression profiles from the Allen Institute, including prenatal stages.
- **Used:** Correlation with prenatal reference expression and maturation-projection features inherited from Version 3.
- **Not used:** BrainSpan samples are not DNT chemical exposures and therefore are not used as toxic/control training examples.
- **Source:** https://www.brainspan.org/

## GSE166297 external organoid transcriptomics

- **Contains:** Chemical-exposure RNA-seq data from a neural organoid/developmental neurotoxicity study.
- **Used:** An external gene-priority table inherited from Version 3, used to emphasize genes reproducibly responsive in another experimental system.
- **Not used:** Its samples are not pooled with GSE63935 as interchangeable training rows because the exposure designs, chemical coverage, and processing differ.
- **Source:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166297

## EPA Carstens integrated DNT-NAM panel

- **Contains:** 92 chemicals and 57 modeled assay endpoints. Endpoints cover human neural progenitor proliferation, cortical and hN2 neurite outgrowth, synapse/neuron maturation, and developmental microelectrode-array activity.
- **Used:** Four independent public-domain targets were defined: any active proliferation endpoint, any active neurite/progenitor endpoint, any active synaptic endpoint, and any active non-cytotoxic MEA endpoint. PubChem-descriptor models transfer these domain probabilities to all GSE63935 chemicals. When an exact chemical match exists, the measured public outcome is blended with the transferred probability. Eleven GSE chemicals match directly.
- **Not used:** Caspase, CellTiter, LDH, and activity-burst cytotoxicity endpoints are excluded from the main domain targets to reduce nonspecific cytotoxicity signals. Missing AC50 values are treated as inactive within the EPA modeled matrix, not as zero potency values.
- **Limitation:** Structure-to-assay transfer is imperfect; its five-fold ROC AUC is reported separately for every public domain.
- **Source:** https://catalog.data.gov/dataset/integrating-data-from-in-vitro-new-approach-methodologies-for-developmental-neurotoxicity

## Harrill high-content DNT assays

- **Contains:** 48,856 concentration-response rows, 21 assay components/endpoints, and 88 sample/control identifiers covering apoptosis, proliferation, neurite outgrowth, synaptogenesis, and viability.
- **Used:** Source-level traceability and confirmation of endpoint definitions represented in the processed Carstens EPA matrix.
- **Not used as an additional independent feature panel:** It substantially overlaps the high-content data already processed into the Carstens matrix. Counting both would give the same experiments extra weight.
- **Source:** https://catalog.data.gov/dataset/data-for-harrill-et-al-testing-for-developmental-neurotoxicity-using-a-suite-of-assays-for

## Brown developmental MEA data

- **Contains:** 990 rows and 26 columns for seven treatments, including developmental day, dose, firing rate, burst, and network-spike measurements.
- **Used:** Source-level validation of the electrical endpoints and chemical identities represented in the EPA MEA evidence.
- **Not used as a second independent electrical feature:** The small panel overlaps the processed EPA/Carstens MEA evidence.
- **Source:** https://catalog.data.gov/dataset/data-for-brown-et-al-mea-developmental-neurotoxicity-screening-manuscript

## ToxCast single-concentration MEA screen

- **Contains:** 4,218 well/assay rows and 31 columns, including sample IDs, endpoint values, hit calls, and final calls.
- **Used:** Traceability for the EPA electrical screen and sample-ID cross-checking through the Carstens mapping.
- **Not directly used:** The file does not provide a complete chemical-name key on its own and overlaps the processed EPA MEA endpoints. Treating it as independent would duplicate the same evidence.
- **Source:** https://catalog.data.gov/dataset/nheerl-mea-toxcast-single-concentration-screening-data

## Cohn oligodendrocyte-development screen

- **Contains:** 1,823 chemicals with normalized oligodendrocyte O1 signal and viability. The publication also identifies cytotoxic chemicals, inhibitors, and drivers of oligodendrocyte development.
- **Used:** Nineteen chemicals match the GSE panel. Sixteen pass the prespecified viability threshold of 0.75 and contribute a continuous disruption score to the progenitor/organization branch.
- **Not used:** Three matched measurements are excluded because low viability makes it impossible to distinguish selective developmental effects from general cytotoxicity. Unmatched chemicals are not treated as negative evidence.
- **Source:** https://catalog.data.gov/dataset/pervasive-environmental-chemicals-impair-oligodendrocyte-development

## PubChem chemical descriptors

- **Contains:** Molecular weight, XLogP, topological polar surface area, hydrogen-bond counts, rotatable bonds, molecular complexity, and formal charge, plus auditable PubChem identifiers/URLs.
- **Used:** Descriptors for the GSE chemicals and 91 of 92 EPA panel chemicals. They support chemical-context features and the four public assay transfer models.
- **Not used:** Polymers, mixtures, proteins, and unresolved chemicals are retained as missing with an exclusion reason; they are not silently replaced by a misleading small molecule.
- **Source:** https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

## Why there is no separate public E/I assay panel

The E/I branch uses RNA markers such as SLC17A7, SATB2, TBR1, GAD2, DLX1, and DLX2. Public atlases can define cell types, but they are not large multi-chemical exposure panels with a common E/I-balance outcome. Atlas cells therefore cannot be counted as extra toxicant examples. This branch should be upgraded later with matched single-cell RNA-seq, interneuron/excitatory-neuron proportions, or functional E/I measurements from a multi-chemical exposure study.
