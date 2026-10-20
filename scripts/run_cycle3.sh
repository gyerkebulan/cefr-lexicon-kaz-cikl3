#!/usr/bin/env bash
# Cycle 3 runs for reports 5 and 6. Run from the root of the cefr-lexicon-kaz repository (patch applied) on the GPU machine.
# Usage: bash run_cycle3.sh <stage> [seed ...]
# Stages: check | data | smoke | reference | v5 | factorial | summary
set -euo pipefail

DATA=${DATA:-dataset/research/ar-cycle3-clean3563-20261020}
SRC=${SRC:-words_clean3563.xlsx}
CACHE=${CACHE:-artifacts/context/lemma_sentences.json}
POOL=${POOL:-artifacts/corpus-resources-v2-20260823/lemma_sentences.json}
LEGACY=${LEGACY:-dataset/splits}
LF=${LF:-artifacts/corpus-resources-v2-20260823/lemma_freqs.json}
SF=${SF:-artifacts/corpus-resources-v2-20260823/surface_freqs.json}
SEEDS_DEFAULT="42 123 456 789 2024"
PY=${PY:-python}
TOOLS=${TOOLS:-scripts}                  # folder with build_cycle3_dataset.py and make_context_files.py

stage=${1:-check}; shift || true
SEEDS=${*:-$SEEDS_DEFAULT}

v4_flags="--variant dual_residual_gated --label-smoothing-sigma 0 --target-only-sample-weight 3.5 \
  --lemma-frequency-file $LF --surface-frequency-file $SF --ordinal-auxiliary --ordinal-weight 0.5 --mid-class-weight-multiplier 1.5"

run() { echo "+ $*"; "$@"; }

case "$stage" in
check)
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
  $PY -c "import torch,transformers,sklearn,pandas;print('torch',torch.__version__,'transformers',transformers.__version__,'sklearn',sklearn.__version__,'pandas',pandas.__version__,'cuda',torch.cuda.is_available())"
  for f in scripts/train_dual.py scripts/train_baselines.py "$CACHE" "$POOL" "$LF" "$SF" "$SRC" "$TOOLS/build_cycle3_dataset.py"; do
    [ -e "$f" ] && echo "OK      $f" || echo "MISSING $f"; done
  ls ~/.cache/huggingface/hub 2>/dev/null | head -5 || true
  git rev-parse --short HEAD 2>/dev/null || true ;;
data)
  run $PY "$TOOLS/build_cycle3_dataset.py" --source "$SRC" --context-cache "$CACHE" --legacy-splits "$LEGACY" --out-dir "$DATA"
  if [ -e "$POOL" ]; then
    run $PY "$TOOLS/make_context_files.py" --dataset "$DATA" --pool "$POOL" --max-contexts 10
  else
    echo "pool of contexts v2 not found ($POOL): multi-context files are skipped (needed only for V5 and context rotation)"
  fi ;;
smoke)   # one short run to confirm that data and code work together (3 epochs, one seed)
  run $PY scripts/train_dual.py --variant dual_gated --epochs 3 --seed 42 --data-dir "$DATA" --tag c3smoke ;;
reference)   # V3 (gated baseline) and V4 (combined), V0 (LR) for every seed
  for s in $SEEDS; do
    run $PY scripts/train_baselines.py --variant full_feature --seed "$s" --data-dir "$DATA"
    run $PY scripts/train_dual.py --variant dual_gated --epochs 20 --seed "$s" --data-dir "$DATA" --tag c3v3
    run $PY scripts/train_dual.py $v4_flags --epochs 20 --seed "$s" --data-dir "$DATA" --tag c3v4
  done ;;
v5)   # V5: V4 checkpoints evaluated with up to 10 contexts averaged (no retraining)
  for s in $SEEDS; do
    ck="results/models/dual_residual_gated_c3v4_s${s}_ordaux_midw1p5_tow3p5/best_model.pt"
    [ -e "$ck" ] || { echo "no checkpoint $ck"; exit 1; }
    run $PY scripts/train_dual.py $v4_flags --epochs 20 --seed "$s" --data-dir "$DATA" --tag c3v5 \
        --eval-only --eval-context-file "$DATA/dev_contexts.json" --max-eval-contexts 10 --checkpoint-path "$ck"
  done ;;
factorial)   # one factor at a time from V3, to separate the effects that were mixed in the combined measurement
  for s in $SEEDS; do
    b="--epochs 20 --seed $s --data-dir $DATA"
    run $PY scripts/train_dual.py --variant dual_gated $b --label-smoothing-sigma 0 --tag c3f_sigma0
    run $PY scripts/train_dual.py --variant dual_residual_gated $b --tag c3f_residual
    run $PY scripts/train_dual.py --variant dual_gated $b --lemma-frequency-file "$LF" --surface-frequency-file "$SF" --tag c3f_freqv2
    run $PY scripts/train_dual.py --variant dual_gated $b --ordinal-auxiliary --ordinal-weight 0.5 --tag c3f_ordaux
    run $PY scripts/train_dual.py --variant dual_gated $b --mid-class-weight-multiplier 1.5 --tag c3f_midw
    run $PY scripts/train_dual.py --variant dual_gated $b --target-only-sample-weight 3.5 --tag c3f_tow
  done ;;
summary)
  $PY - <<'EOF'
import glob, json, os, statistics as st
rows = {}
for f in glob.glob("results/metrics/*_dev.json"):
    d = json.load(open(f, encoding="utf-8"))
    name = os.path.basename(f)
    tag = next((t for t in ("c3v3", "c3v4", "c3v5", "c3f_sigma0", "c3f_residual", "c3f_freqv2", "c3f_ordaux", "c3f_midw", "c3f_tow", "full_feature") if t in name), None)
    if tag:
        rows.setdefault(tag, []).append(d)
print(f"{'tag':14}{'n':>3}{'Acc':>8}{'MacroF1':>9}{'sd':>8}{'MAE':>8}{'QWK':>8}{'ECE':>8}{'far>=2':>8}")
for tag, ds in sorted(rows.items()):
    g = lambda k: st.mean(x[k] for x in ds)
    sd = st.stdev([x["macro_f1"] for x in ds]) if len(ds) > 1 else float("nan")
    print(f"{tag:14}{len(ds):3}{g('accuracy'):8.4f}{g('macro_f1'):9.4f}{sd:8.4f}{g('ordinal_mae'):8.4f}"
          f"{g('quadratic_weighted_kappa'):8.4f}{g('expected_calibration_error'):8.4f}{g('far_error_rate'):8.4f}")
EOF
  ;;
*) echo "unknown stage: $stage"; exit 2 ;;
esac
