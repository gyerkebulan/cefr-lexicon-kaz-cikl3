#!/usr/bin/env bash
# Reconstruction of the cycle 1-2 recipe (V1/V2): dual_gated, XLM-R-base, focal loss gamma 2.0, no label smoothing.
# Usage: run_v2.sh <gpu> <seeds...>
set -uo pipefail
GPU=$1; shift
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
T=dataset/research/ar-cycle3-testeval
for s in $SEEDS; do
  python scripts/train_dual.py --variant dual_gated --loss-type focal --focal-gamma 2.0 --epochs 20 --seed $s --data-dir $DATA --tag c3v2star > ~/c3logs/v2star_s$s.log 2>&1
  ck=$(ls results/models/*_c3v2star_s${s}*/best_model.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python scripts/train_dual.py --variant dual_gated --loss-type focal --focal-gamma 2.0 --epochs 20 --seed $s --data-dir $T --tag c3tV2 --eval-only --checkpoint-path "$ck" > ~/c3logs/t_v2_s$s.log 2>&1
  fi
  rm -rf results/models/*_c3v2star_s${s}*
  echo "done v2star seed $s $(date +%T)" >> ~/c3logs/progress_v2star_gpu$GPU.txt
done
