# CEFR level prediction for Kazakh words – cycle 3

Testing and refinement of a neural model that predicts the CEFR level (A1–C1) of a Kazakh word. This repository
continues [cefr-lexicon-kaz](https://github.com/gyerkebulan/cefr-lexicon-kaz) (code of cycle 2, revision `8803297`) and adds
the data protocol, run scripts, analysis scripts and results of cycle 3. At this stage the model works with single words
(lemmas); phrases are the next step.

## Summary of results

Held-out part (535 words), mean over five networks (ensemble of the five in brackets):

| Version | Accuracy | Within-1 | Macro-F1 | MAE | ECE |
| --- | --- | --- | --- | --- | --- |
| V0 (logistic regression) | 0.320 | 0.587 | 0.297 | 1.351 | 0.111 |
| V3 (two-stream network, gated fusion) | 0.335 (0.342) | 0.701 (0.710) | 0.326 | 1.132 | 0.219 |
| V5 (V4 with up to 10 contexts averaged) | 0.352 (0.370) | 0.689 (0.712) | 0.334 | 1.124 | 0.169 |
| **V6 (final)** | **0.385 (0.391)** | **0.750 (0.763)** | **0.379** | **0.961** | **0.068** |

Accuracy is the share of exactly predicted levels; Within-1 is the share of words predicted within one level of the label;
ECE is the expected calibration error. V6 = XLM-RoBERTa-large + two-stream network with residual fusion + corpus
frequencies v2 + averaging over up to 10 contexts + ensemble of five networks. The full tables (development part, single-factor runs, learning
curve, candidates, per-level F1, error analysis) are in [docs/CYCLE3.md](docs/CYCLE3.md) and `docs/cycle3/`.

## Data

* Source list: 3 563 rows (3 497 unique words, levels A1–C1), one row per “word + part of speech”.
* Grouped, stratified split: train 2 492 / dev 536 / test 535 words, 2 677 groups; a group (same word, linked forms,
  near-identical words, shared context sentence) always lies in one part. All 13 overlap checks equal 0
  (`docs/cycle3/splits/dataset-manifest.json`). Word lists without sentence text: `docs/cycle3/splits/*_words.csv`.
* Contexts: up to 10 corpus sentences per lemma (Leipzig Corpora Collection); no sentence is shared between parts
  (`docs/cycle3/splits/context-files-audit.json`).

## Repository layout

| Path | Content |
| --- | --- |
| `scripts/build_cycle3_dataset.py`, `scripts/make_context_files.py` | grouped split of the word list and multi-context files with a cross-split overlap audit |
| `scripts/run_cycle3.sh` | staged runner (check, data, smoke, reference, v5, factorial, summary) |
| `scripts/cycle3/` | run scripts for each experiment stage and analysis scripts (post-hoc calibration, word-level errors, held-out report) |
| `scripts/train_dual.py` | training and evaluation of the two-stream model (fusion norm included in the optimizer, strict checkpoint loading) |
| `docs/CYCLE3.md` | description of cycle 3 and result tables |
| `docs/cycle3/metrics/` | metrics files of all cycle 3 runs |
| `docs/cycle3/summary/` | summary tables and logs of the analyses |
| `docs/cycle3/splits/` | split manifests and word lists |

Branches: `cycle3-testing` (stage 1: testing of versions V0–V5, factors, contexts, calibration, learning curve, backbone size),
`cycle3-refinement` (stage 2: refinement candidates, final version V6, one-time held-out evaluation), `main` (both merged).

## Reproduction

1. Environment: Python 3, PyTorch 2.5.1, Transformers 4.46.1, scikit-learn, pandas, openpyxl (see `environment.yml` of the base repository).
2. Resources not distributed here: the corpus sentences, the frequency files `lemma_freqs.json` / `surface_freqs.json`
   (resources v2) and model weights. They are built with `scripts/build_corpus_data.py` and `scripts/prepare_handoff_corpus.py`
   from the Leipzig corpora.
3. Build the split and context files: `bash scripts/run_cycle3.sh data` (after `check`), then the stages in `scripts/cycle3/`
   (`run_reference_and_factors.sh`, `run_resources_v2.sh`, `run_learning_curve_backbone_features.sh`, `run_candidates*.sh`,
   `final_evaluation.sh`); analyses are in `scripts/cycle3/*.py`.

## Base repository and citation

Code, lexicon and manuscript materials: https://github.com/gyerkebulan/cefr-lexicon-kaz,
archived release https://doi.org/10.5281/zenodo.19365834, dataset mirror
https://huggingface.co/datasets/Gulnur7/kazakh-lexical-complexity-classes.

The Leipzig Corpora Collection sentences are not redistributed in this repository.
