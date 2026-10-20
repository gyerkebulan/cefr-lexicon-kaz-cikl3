#!/usr/bin/env bash
# Usage: run_part.sh <gpu> <part> <seeds...>   parts: ref | fact
# One GPU only (shared machine). Legacy frequencies/contexts; resources v2 not needed.
set -uo pipefail
GPU=$1; PART=$2; shift 2
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
mkdir -p ~/c3logs
for s in $SEEDS; do
  b="--epochs 20 --seed $s --data-dir $DATA"
  if [ "$PART" = ref ]; then
    python scripts/train_baselines.py --variant full_feature --seed $s --data-dir $DATA > ~/c3logs/v0_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_gated $b --tag c3v3 > ~/c3logs/v3_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_residual_gated $b --label-smoothing-sigma 0 --target-only-sample-weight 3.5 \
      --ordinal-auxiliary --ordinal-weight 0.5 --mid-class-weight-multiplier 1.5 --tag c3v4nf > ~/c3logs/v4nf_s$s.log 2>&1
  else
    python scripts/train_dual.py --variant dual_gated $b --label-smoothing-sigma 0 --tag c3f_sigma0 > ~/c3logs/f_sigma0_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_residual_gated $b --tag c3f_residual > ~/c3logs/f_residual_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_gated $b --ordinal-auxiliary --ordinal-weight 0.5 --tag c3f_ordaux > ~/c3logs/f_ordaux_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_gated $b --mid-class-weight-multiplier 1.5 --tag c3f_midw > ~/c3logs/f_midw_s$s.log 2>&1
    python scripts/train_dual.py --variant dual_gated $b --target-only-sample-weight 3.5 --tag c3f_tow > ~/c3logs/f_tow_s$s.log 2>&1
  fi
  echo "done $PART seed $s $(date +%T)" >> ~/c3logs/progress_${PART}_gpu$GPU.txt
done
echo "ALL DONE $PART gpu$GPU $(date +%T)" >> ~/c3logs/progress_${PART}_gpu$GPU.txt
