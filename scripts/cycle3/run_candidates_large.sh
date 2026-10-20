#!/usr/bin/env bash
# Usage: run_part4k.sh <gpu> <wait_pid|0> <seeds...>
# Candidate P4 (XLM-R-large, residual fusion, frequencies v2): trained and evaluated on dev with <=10 contexts;
# checkpoints are KEPT for the single final evaluation on the held-out part.
set -uo pipefail
GPU=$1; WAIT=$2; shift 2
SEEDS="$*"
while [ "$WAIT" != "0" ] && kill -0 "$WAIT" 2>/dev/null; do sleep 15; done
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
RES=artifacts/corpus-resources-v2-20260823
R="--variant dual_residual_gated --lemma-frequency-file $RES/lemma_freqs.json --surface-frequency-file $RES/surface_freqs.json --transformer-model xlm-roberta-large"
note() { echo "$1 $(date +%T)" >> ~/c3logs/progress4k_gpu$GPU.txt; }
for s in $SEEDS; do
  python scripts/train_dual.py $R --epochs 20 --seed $s --data-dir $DATA --tag c4p4 > ~/c3logs/c4p4_s$s.log 2>&1
  ck=$(ls results/models/*_c4p4_s${s}*/best_model.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python scripts/train_dual.py $R --epochs 20 --seed $s --data-dir $DATA --tag c4p4v5 --eval-only \
      --eval-context-file $DATA/dev_contexts.json --max-eval-contexts 10 --checkpoint-path "$ck" > ~/c3logs/c4p4v5_s$s.log 2>&1
  fi
  note "done p4 seed $s"
done
note "ALL DONE p4k"
