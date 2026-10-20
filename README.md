# CEFR-based Morphology-Aware Lexical Complexity Prediction for Kazakh

Source code and curated data for the manuscript:
**"A CEFR-Graded Lexicon and Morphology-Aware Benchmarks for Kazakh Lexical Complexity Prediction"**
submitted to *PeerJ Computer Science*.

- Archived release (DOI): https://doi.org/10.5281/zenodo.19365834
- Dataset mirror: https://huggingface.co/datasets/Gulnur7/kazakh-lexical-complexity-classes

## Cycle 3 (testing and refinement)

Data protocol, run scripts, analysis scripts and held-out results of cycle 3: see [docs/CYCLE3.md](docs/CYCLE3.md).

## Передача проекта - 16 октября 2026

Начните с [инструкции по передаче](docs/HANDOFF.md). Готовые файлы находятся локально в
`artifacts/handoff-2026-10-16/`: краткий PDF, пакет исходников/данных/отчётов и отдельные веса.
Исследовательские версии августа и сентября сохранены отдельно от основной версии.

Ниже сохранено описание исходного проекта. Числа из рукописи не следует считать
повторно подтверждёнными: локальные исходные splits содержат 3 108 / 671 / 671 строк
(4 450 суммарно), а поздние серии используют другие разбиения. Результаты и ограничения
каждой серии перечислены в передаваемом реестре экспериментов.

## Description

This repository provides the first CEFR-graded lexicon for the Kazakh language (4,561 lemma–POS entries across A1–C1), together with the feature extraction pipeline, classical baselines, neural context/morphology encoders, gated fusion model, cross-lingual projection code, and evaluation scripts used to produce the results reported in the manuscript.

## Dataset Information

- **Size:** 4,561 Kazakh lemma–POS entries graded A1–C1 (after cleaning and deduplication: 4,437 unique lemma–POS pairs, partitioned 70/15/15 into train/dev/test).
- **Source:** State-sponsored Kazakh pedagogical handbooks (lexical minimum textbooks for each CEFR level).
- **Encoding:** UTF-8 Kazakh Cyrillic throughout. CSV splits include a UTF-8 byte-order mark for compatibility with Microsoft Excel and Windows tooling.
- **Files:**
  - `dataset/huggingface/kazakh_cefr_lexicon.json` — full lexicon
  - `dataset/processed/combined_lexicon.csv` — cleaned modeling subset
  - `dataset/splits/{train,dev,test}.csv` — stratified 70/15/15 splits

Field descriptions (CSV):

| Field   | Type   | Description                                                    |
|---------|--------|----------------------------------------------------------------|
| `lemma` | string | Kazakh lemma (Cyrillic script, UTF-8)                          |
| `pos`   | string | Part of speech (`NOUN`, `VERB`, `ADJ`, `ADV`, `NUM`, `PRON`, `ADP`, `MODAL`, `INTJ`, `AUX`, `OTHER`) |
| `cefr`  | string | CEFR proficiency level (`A1`, `A2`, `B1`, `B2`, `C1`)          |
| `freq`  | int    | Corpus frequency (Leipzig Kazakh monolingual corpus)           |

## Code Information

```
cefr-lexicon-kaz/
├── config/           # YAML configs (training, paths, crosslingual, suffixes)
├── src/              # Feature extraction, data prep, models, metrics
│   └── models/       # Baselines, context/morph encoders, fusion, CORAL loss
├── scripts/          # Entry points: preprocess, train_*, evaluate
├── cefr/             # Lexicon extraction, morphology helpers, text utilities
├── artifacts/        # Precomputed caches (lemma sentences, frequencies)
└── dataset/          # Processed lexicon, splits, Hugging Face mirror
```

Key files:

- `src/features.py` — 40-dim handcrafted feature extraction (frequency, orthography, morphology, POS, TF-IDF).
- `src/models/dual_encoder.py` — dual-branch architecture (XLM-RoBERTa context + CharCNN/MorphMLP morphology).
- `src/models/fusion.py` — concatenation and sigmoid-gated fusion.
- `scripts/train_dual.py` — training entry point for the dual-encoder / gated fusion model.
- `scripts/evaluate.py` — held-out test evaluation with Accuracy, Macro-F1, MAE, and QWK.

## Usage Instructions

### 1. Environment setup

```bash
conda env create -f environment.yml
conda activate kazakh_cefr_env
```

### 2. Load the dataset

```python
import pandas as pd

train = pd.read_csv('dataset/splits/train.csv', encoding='utf-8-sig')
dev   = pd.read_csv('dataset/splits/dev.csv',   encoding='utf-8-sig')
test  = pd.read_csv('dataset/splits/test.csv',  encoding='utf-8-sig')
```

Or via Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset('Gulnur7/kazakh-lexical-complexity-classes')
```

### 3. Preprocessing

```bash
python scripts/preprocess.py --data-type lexicon
```

### 4. Train classical baselines

```bash
python scripts/train_baselines.py --variant majority
python scripts/train_baselines.py --variant freq_only
python scripts/train_baselines.py --variant full_feature
```

### 5. Train neural models

```bash
# Context-only encoder (XLM-RoBERTa)
python scripts/train_context.py --variant context_only_xlmr

# Dual encoder with concatenation fusion
python scripts/train_dual.py --variant dual_concat

# Dual encoder with gated fusion
python scripts/train_dual.py --variant dual_gated
```

### 6. Evaluation

```bash
python scripts/evaluate.py --model dual_gated --generate-report
```

### 7. Multi-seed ablation (Table 6 in the manuscript)

```bash
for seed in 42 123 456 789 2024; do
  python scripts/train_dual.py --variant dual_gated --seed "$seed"
done
```

## Accuracy-and-robustness autoresearch

The locked series in `config/autoresearch.yaml` separates dataset construction,
candidate evaluation, and the one-shot final test.

```bash
# Rebuild the immutable leakage-grouped 70/15/15 series.
python3 scripts/build_research_splits.py --config config/autoresearch.yaml

# Candidate and ablation commands open train.csv and dev.csv only.
python3 scripts/train_dual.py \
  --research-config config/autoresearch.yaml \
  --proposal-id PROPOSAL_ID --seed 42

# Generate deterministic robustness inputs, run model inference on them, then score.
python3 scripts/evaluate_robustness.py \
  --research-config config/autoresearch.yaml \
  --input dataset/research/ar-accrobust-20260724/dev.csv \
  --output results/autoresearch/ar-accrobust-20260724/robustness-inputs.csv
python3 scripts/train_dual.py \
  --research-config config/autoresearch.yaml \
  --proposal-id PROPOSAL_ID --seed 42 \
  --robustness-input results/autoresearch/ar-accrobust-20260724/robustness-inputs.csv
python3 scripts/evaluate_robustness.py \
  --research-config config/autoresearch.yaml \
  --predictions results/predictions/MODEL_robustness.csv \
  --output ROBUSTNESS_METRICS.json

# Fit one temperature only from five-fold, exactly-once OOF predictions for
# every frozen training row.
python3 scripts/calibrate_predictions.py \
  --research-config config/autoresearch.yaml \
  --proposal-id PROPOSAL_ID \
  --oof-predictions OOF_PREDICTIONS.csv \
  --output CALIBRATED_OOF_PREDICTIONS.csv

# Only an accepted, five-seed champion manifest can consume the test once.
python3 scripts/final_test.py \
  --research-config config/autoresearch.yaml \
  --champion-manifest CHAMPION_MANIFEST.json
```

Each research run records its proposal, revision, changed files, seed,
configuration and protocol hashes, dev metrics, runtime, parameter count, peak
VRAM, artifact paths and checksums, and outcome. Failed and interrupted runs
remain separate immutable records. Test results are never a signal for another
same-series tuning round.

Target spans use NFC-normalized, case-insensitive tokenizer offsets. All
occurrences are pooled when a target repeats. A missing or truncated target
falls back explicitly to target-only encoding; an empty target mask or
non-finite context representation is an error.

Remote runs require verified encrypted-at-rest storage under
`ESAI_RA_ZHANTORE`. Durable outputs are mirrored into the ignored
`remote_mirror/` tree and checksum-verified with
`scripts/mirror_gpu_artifacts.py`.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- scikit-learn ≥ 1.3
- pandas, numpy, PyYAML, tqdm

The baseline dependency list is in `environment.yml`; it is not version-pinned. Use the recorded environment of the selected experiment for reproduction.

Training was performed on NVIDIA A100 (40 GB) GPUs. Classical baselines run on CPU in under one minute; the full dual-encoder with XLM-RoBERTa-base trains in roughly 30 minutes per seed on a single A100.

## Configuration

Hyperparameters are in `config/training.yaml`:

- Epochs: 20
- Batch size: 16
- Learning rate: 2e-5
- Early stopping patience: 7
- Transformer: `xlm-roberta-base`
- Fusion: gated

The Kazakh suffix inventory used for morphological feature extraction is defined in `config/kaz_suffixes.yaml`.

## Methodology (summary)

1. **Lexicon construction:** lemmas and POS tags extracted from CEFR-level pedagogical handbooks, cleaned for Cyrillic validity and length, and deduplicated at the lemma–POS key level.
2. **Feature engineering:** 40-dimensional handcrafted vector covering frequency, orthography, morphology (HFST-based), POS, and TF-IDF signals.
3. **Models:** classical baselines (LR, SVM, RF), context-only XLM-RoBERTa encoder, morphology-only CharCNN + MorphMLP encoder, and dual-encoder with concatenation or sigmoid-gated fusion.
4. **Evaluation:** Accuracy, Macro-F1, MAE, quadratic-weighted kappa on the held-out test split, averaged over five random seeds.
5. **Cross-lingual projection:** Kazakh surface forms projected into Russian (KELLY) and English (EFLLex) CEFR lexicons via open-source machine translation (Tilmash, OPUS-MT) as a silver-label baseline.

Full methodological details are in Sections 3 and 4 of the manuscript.

## Citation

When using this dataset or code, please cite the manuscript once it is published and the Zenodo archive:

```bibtex
@software{yerkebulan2026cefrkaz,
  author  = {Yerkebulan, Gulnur and Akanova, Akerke and Galymzhan, Zhantore and Ospanova, Nazira},
  title   = {Code and Data for: A CEFR-Graded Lexicon and Morphology-Aware Benchmarks for Kazakh Lexical Complexity Prediction},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.19365834},
  url     = {https://doi.org/10.5281/zenodo.19365834}
}
```

## License and Contribution

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code:** MIT License
- Issues and pull requests are welcome.

## Contact

Corresponding authors:

- **Gulnur Yerkebulan** — erkgulnur@mail.ru
- **Akerke Akanova** — akerkegansaj1995@gmail.com
