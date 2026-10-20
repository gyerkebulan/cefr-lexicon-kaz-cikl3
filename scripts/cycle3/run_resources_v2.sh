#!/usr/bin/env bash
# Usage: run_part2.sh <gpu> <part> <seeds...>   parts: v4v5 | freq
set -uo pipefail
GPU=$1; PART=$2; shift 2
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
RES=artifacts/corpus-resources-v2-20260823
LF=$RES/lemma_freqs.json
SF=$RES/surface_freqs.json
V4F="--variant dual_residual_gated --label-smoothing-sigma 0 --target-only-sample-weight 3.5 --lemma-frequency-file $LF --surface-frequency-file $SF --ordinal-auxiliary --ordinal-weight 0.5 --mid-class-weight-multiplier 1.5"
mkdir -p ~/c3logs
for s in $SEEDS; do
  b="--epochs 20 --seed $s --data-dir $DATA"
  if [ "$PART" = v4v5 ]; then
    python scripts/train_dual.py $V4F $b --tag c3v4 > ~/c3logs/v4_s$s.log 2>&1
    ck=$(ls results/models/*_c3v4_s${s}_*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$ck" ]; then
      python scripts/train_dual.py $V4F $b --tag c3v5 --eval-only --eval-context-file $DATA/dev_contexts.json \
        --max-eval-contexts 10 --checkpoint-path "$ck" > ~/c3logs/v5_s$s.log 2>&1
    else
      echo "no V4 checkpoint for seed $s" >> ~/c3logs/progress_${PART}_gpu$GPU.txt
    fi
  else
    python scripts/train_dual.py --variant dual_gated $b --lemma-frequency-file $LF --surface-frequency-file $SF --tag c3f_freqv2 > ~/c3logs/f_freqv2_s$s.log 2>&1
  fi
  echo "done $PART seed $s $(date +%T)" >> ~/c3logs/progress_${PART}_gpu$GPU.txt
done
echo "ALL DONE $PART gpu$GPU $(date +%T)" >> ~/c3logs/progress_${PART}_gpu$GPU.txt
