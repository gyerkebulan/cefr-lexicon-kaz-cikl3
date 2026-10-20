# Cycle 3: testing and refinement on the cleaned 3563-word list

This document describes the third testing/refinement cycle (reports 5 and 6 of the project). It builds on the
cycle 2 code in this repository (`8803297`) and adds the data protocol, run scripts, analysis scripts and
metrics of the cycle. Sentence text from the Leipzig corpus is not included; the split files contain words only.

## Data

* Source list: `words_clean3563.xlsx` (3563 rows, 3497 unique words, levels A1-C1).
* Grouped, stratified split: train 2492 / dev 536 / test 535 words, 2677 groups (same word, linked forms,
  words of 6+ characters at edit distance 1, shared context sentence). All 13 overlap checks are 0
  (`docs/cycle3/splits/dataset-manifest.json`; word lists without sentences in the same folder).
* Contexts: pool v2 (up to 10 sentences per lemma); no sentence is shared between train, dev and test
  (`context-files-audit.json`).
* Builders: `scripts/build_cycle3_dataset.py`, `scripts/make_context_files.py`.

## Runs

All dev results are means over five seeds (42, 123, 456, 789, 2024); metrics files are in
`docs/cycle3/metrics/` (the test runs use the tags `c3tV3`, `c3tV4`, `c3tV5`, `c3tV6` and the
`full_feature` files written after the held-out evaluation).

| Stage | Scripts |
| --- | --- |
| V0, V3, V4 (without frequencies v2) and single-factor runs | `scripts/cycle3/run_reference_and_factors.sh` |
| Resources v2: V4, V5, frequency factor | `scripts/cycle3/run_resources_v2.sh` |
| Learning curve, XLM-R-large, feature sets, context count, temperature | `scripts/cycle3/prepare_learning_curve.py`, `scripts/cycle3/run_learning_curve_backbone_features.sh` |
| Refinement candidates P0-P6 | `scripts/cycle3/run_candidates.sh`, `run_candidates_large.sh`, `run_large_rotation.sh` |
| One-time held-out evaluation | `scripts/cycle3/prepare_heldout_eval.py`, `scripts/cycle3/final_evaluation.sh`, `scripts/cycle3/final_report.py` |
| Analysis | `scripts/cycle3/summarize_runs.py`, `posthoc_analysis.py`, `error_analysis.py` |

The held-out part is evaluated through the ordinary evaluation path with the held-out words placed in the dev slot of
a copy of the data directory. The sealed-test mode of `train_dual.py` was not used: it requires a locked research
protocol and forbids some of the settings used here. The held-out part was evaluated once, after the final configuration
had been fixed on dev.

## Dev results (536 words, five seeds, mean)

| Version | Accuracy | Macro-F1 | MAE | QWK | ECE | Within-1 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 (LR) | 0.304 | 0.284 | 1.300 | 0.309 | 0.117 | 0.608 |
| V1-V2* (reconstruction) | 0.357 | 0.349 | 1.112 | 0.416 | 0.181 | 0.673 |
| V3 | 0.352 | 0.345 | 1.121 | 0.418 | 0.201 | 0.673 |
| V4 | 0.351 | 0.339 | 1.160 | 0.387 | 0.264 | 0.660 |
| V5 (V4 + up to 10 contexts) | 0.372 | 0.353 | 1.090 | 0.441 | 0.151 | 0.690 |
| P0 (residual fusion + frequencies v2) | 0.385 | 0.373 | 1.045 | 0.478 | 0.071 | 0.702 |
| P4 = V6 (P0 + XLM-R-large) | 0.410 | 0.402 | 0.953 | 0.527 | 0.055 | 0.739 |

## Held-out results (535 words)

| Version | Accuracy (5 nets) | Macro-F1 | MAE | QWK | ECE | Within-1 | Accuracy (ensemble) | Within-1 (ensemble) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.320 | 0.297 | 1.351 | 0.252 | 0.111 | 0.587 | - | - |
| V1-V2* | 0.342 | 0.333 | 1.118 | 0.409 | 0.205 | 0.692 | 0.344 | 0.697 |
| V3 | 0.335 | 0.326 | 1.132 | 0.406 | 0.219 | 0.701 | 0.342 | 0.710 |
| V4 | 0.332 | 0.318 | 1.178 | 0.382 | 0.294 | 0.674 | 0.335 | 0.675 |
| V5 | 0.352 | 0.334 | 1.124 | 0.423 | 0.169 | 0.689 | 0.370 | 0.712 |
| V6 | 0.385 | 0.379 | 0.961 | 0.539 | 0.068 | 0.750 | 0.391 | 0.763 |

V6 (XLM-R-large, residual fusion, frequencies v2, up to 10 contexts averaged, ensemble of five networks) is
better than V3 and V4 on Accuracy and Within-1 and better than V5 on Within-1 (paired bootstrap of the
ensembles: +0.050, 95 % CI 0.011 to 0.090); its Accuracy and Macro-F1 gain over V5 (+0.021 and +0.030) is not
confirmed by the interval. F1 of level B1 is lower than for V5 (0.230 against 0.302).

## Not done

The original V1 and V2 were not re-evaluated (the hand-over package has no weights and no full configuration); V1-V2* is a reconstruction from the 30 July 2026 reproduction configuration (XLM-R-base, gated fusion, focal loss gamma 2, no stacking), run by `scripts/cycle3/run_v1v2_reconstruction.sh`. Feature-group
ablation, cross-validation on train+dev, LoRA and stacking were not run. Level C2 is not included.
