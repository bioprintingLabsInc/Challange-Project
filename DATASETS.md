# Datasets

The detailed source-by-source audit is maintained in [`DATASET_GUIDE.md`](DATASET_GUIDE.md).

| Dataset | Role | Label use |
|---|---|---|
| GSE63935 | Primary chemical-level RNA, pathways, development, dose, and overall target | Overall toxic/control target |
| GSE126786 / DevTox2D | Label-free high-variance gene prior | Labels excluded |
| BrainSpan | Human developmental reference | No chemical labels |
| GSE166297 | External organoid gene-priority evidence inherited from V3 | Not pooled as training rows |
| EPA Carstens DNT-NAM | Four functional public-domain targets and direct matches | Transfer-model targets only |
| Cohn oligodendrocyte screen | Non-cytotoxic progenitor/organization evidence | Continuous feature evidence |
| PubChem | Physicochemical descriptors and structure provenance | No toxicity labels |

Raw public-source files are not duplicated in this compact working package. `dataset_manifest.csv`
records their roles; preparation scripts define matching, filtering, endpoint construction, and
exclusions. Before sharing externally, add download dates, source-file checksums, and applicable
redistribution terms for every raw dataset.
