#!/usr/bin/env bash
# Usage: run_p6.sh <gpu> <seeds...>
# Candidate P6: XLM-R-large + residual fusion + frequencies v2 + training-context rotation (single-context dev evaluation).
set -uo pipefail
GPU=$1; shift
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
RES=artifacts/corpus-resources-v2-20260823
R="--variant dual_residual_gated --lemma-frequency-file $RES/lemma_freqs.json --surface-frequency-file $RES/surface_freqs.json --transformer-model xlm-roberta-large --train-context-file $DATA/train_contexts.json"
for s in $SEEDS; do
  python scripts/train_dual.py $R --epochs 20 --seed $s --data-dir $DATA --tag c4p6 > ~/c3logs/c4p6_s$s.log 2>&1
  echo "done p6 seed $s $(date +%T)" >> ~/c3logs/progress4_p6_gpu$GPU.txt
done
