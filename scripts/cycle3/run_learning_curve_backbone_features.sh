#!/usr/bin/env bash
# Usage: run_part3.sh <gpu> <part> <seeds...>   parts: lc | large | eval
set -uo pipefail
GPU=$1; PART=$2; shift 2
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
RES=artifacts/corpus-resources-v2-20260823
V4F="--variant dual_residual_gated --label-smoothing-sigma 0 --target-only-sample-weight 3.5 --lemma-frequency-file $RES/lemma_freqs.json --surface-frequency-file $RES/surface_freqs.json --ordinal-auxiliary --ordinal-weight 0.5 --mid-class-weight-multiplier 1.5"
mkdir -p ~/c3logs
note() { echo "$1 $(date +%T)" >> ~/c3logs/progress3_${PART}_gpu$GPU.txt; }

if [ "$PART" = lc ]; then
  for frac in 25 50 75; do
    for s in $SEEDS; do
      python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir dataset/research/ar-cycle3-lc$frac --tag c3lc$frac > ~/c3logs/lc${frac}_s$s.log 2>&1
      rm -rf results/models/*_c3lc${frac}_s${s}*
      note "done lc$frac seed $s"
    done
  done
elif [ "$PART" = large ]; then
  for s in $SEEDS; do
    python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir $DATA --transformer-model xlm-roberta-large --tag c3large > ~/c3logs/large_s$s.log 2>&1
    rm -rf results/models/*_c3large_s${s}*
    note "done large seed $s"
  done
  for s in $SEEDS; do
    python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir $DATA --feature-selection all --tag c3featall > ~/c3logs/featall_s$s.log 2>&1
    rm -rf results/models/*_c3featall_s${s}*
    python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir $DATA --feature-top-k 20 --tag c3featk20 > ~/c3logs/featk20_s$s.log 2>&1
    rm -rf results/models/*_c3featk20_s${s}*
    note "done feat seed $s"
  done
else
  for s in $SEEDS; do
    ck4=$(ls results/models/*_c3v4_s${s}_*/best_model.pt 2>/dev/null | head -1)
    ck3=results/models/dual_gated_c3v3_s${s}/best_model.pt
    for n in 1 3 5; do
      python scripts/train_dual.py $V4F --epochs 20 --seed $s --data-dir $DATA --tag c3ctx$n --eval-only --eval-context-file $DATA/dev_contexts.json --max-eval-contexts $n --checkpoint-path "$ck4" > ~/c3logs/ctx${n}_s$s.log 2>&1
    done
    for t in 0.8 1.5 2.0 3.0; do
      python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir $DATA --tag c3T${t}v3 --eval-only --temperature $t --checkpoint-path "$ck3" > ~/c3logs/T${t}v3_s$s.log 2>&1
      python scripts/train_dual.py $V4F --epochs 20 --seed $s --data-dir $DATA --tag c3T${t}v5 --eval-only --eval-context-file $DATA/dev_contexts.json --max-eval-contexts 10 --temperature $t --checkpoint-path "$ck4" > ~/c3logs/T${t}v5_s$s.log 2>&1
    done
    note "done eval seed $s"
  done
fi
note "ALL DONE $PART"
