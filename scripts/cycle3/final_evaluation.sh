#!/usr/bin/env bash
# One-time evaluation of the frozen models on the held-out part (placed in the dev slot of ar-cycle3-testeval).
# Usage: final_eval.sh <gpu> <part> <seeds...>   parts: base | v6
set -uo pipefail
GPU=$1; PART=$2; shift 2
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
T=dataset/research/ar-cycle3-testeval
RES=artifacts/corpus-resources-v2-20260823
LF=$RES/lemma_freqs.json; SF=$RES/surface_freqs.json
V4F="--variant dual_residual_gated --label-smoothing-sigma 0 --target-only-sample-weight 3.5 --lemma-frequency-file $LF --surface-frequency-file $SF --ordinal-auxiliary --ordinal-weight 0.5 --mid-class-weight-multiplier 1.5"
V6F="--variant dual_residual_gated --lemma-frequency-file $LF --surface-frequency-file $SF --transformer-model xlm-roberta-large"
mkdir -p ~/c3logs
note() { echo "$1 $(date +%T)" >> ~/c3logs/progress_final_${PART}_gpu$GPU.txt; }
for s in $SEEDS; do
  if [ "$PART" = base ]; then
    python scripts/train_baselines.py --variant full_feature --seed $s --data-dir $T > ~/c3logs/t_v0_s$s.log 2>&1
    ck3=results/models/dual_gated_c3v3_s${s}/best_model.pt
    python scripts/train_dual.py --variant dual_gated --epochs 20 --seed $s --data-dir $T --tag c3tV3 --eval-only --checkpoint-path "$ck3" > ~/c3logs/t_v3_s$s.log 2>&1
    ck4=$(ls results/models/*_c3v4_s${s}_*/best_model.pt | head -1)
    python scripts/train_dual.py $V4F --epochs 20 --seed $s --data-dir $T --tag c3tV4 --eval-only --checkpoint-path "$ck4" > ~/c3logs/t_v4_s$s.log 2>&1
    python scripts/train_dual.py $V4F --epochs 20 --seed $s --data-dir $T --tag c3tV5 --eval-only --eval-context-file $T/dev_contexts.json --max-eval-contexts 10 --checkpoint-path "$ck4" > ~/c3logs/t_v5_s$s.log 2>&1
  else
    ck6=$(ls results/models/*_c4p4_s${s}*/best_model.pt | head -1)
    python scripts/train_dual.py $V6F --epochs 20 --seed $s --data-dir $T --tag c3tV6 --eval-only --eval-context-file $T/dev_contexts.json --max-eval-contexts 10 --checkpoint-path "$ck6" > ~/c3logs/t_v6_s$s.log 2>&1
  fi
  note "done $PART seed $s"
done
note "ALL DONE $PART"
