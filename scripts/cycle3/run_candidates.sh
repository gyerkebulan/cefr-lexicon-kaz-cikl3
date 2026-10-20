#!/usr/bin/env bash
# Usage: run_part4.sh <gpu> "<parts>" <seeds...>
# Candidate approaches for report 6, each over the base pair R = residual fusion + frequency v2 resources,
# trained, then evaluated on dev with <=10 contexts averaged (as V5). Checkpoints are deleted after evaluation.
set -uo pipefail
GPU=$1; PARTS=$2; shift 2
SEEDS="$*"
cd ~/cefr-lexicon-kaz
source ~/cefr3venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
DATA=dataset/research/ar-cycle3-clean3563-20261020
RES=artifacts/corpus-resources-v2-20260823
R="--variant dual_residual_gated --lemma-frequency-file $RES/lemma_freqs.json --surface-frequency-file $RES/surface_freqs.json"
mkdir -p ~/c3logs
note() { echo "$1 $(date +%T)" >> ~/c3logs/progress4_gpu$GPU.txt; }

run_cfg() {   # name, extra flags, seed
  local name=$1 extra=$2 s=$3
  python scripts/train_dual.py $R $extra --epochs 20 --seed $s --data-dir $DATA --tag c4$name > ~/c3logs/c4${name}_s$s.log 2>&1
  local ck
  ck=$(ls results/models/*_c4${name}_s${s}*/best_model.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    python scripts/train_dual.py $R $extra --epochs 20 --seed $s --data-dir $DATA --tag c4${name}v5 --eval-only \
      --eval-context-file $DATA/dev_contexts.json --max-eval-contexts 10 --checkpoint-path "$ck" > ~/c3logs/c4${name}v5_s$s.log 2>&1
  fi
  rm -rf results/models/*_c4${name}_s${s}*
  note "done $name seed $s"
}

for part in $PARTS; do
  for s in $SEEDS; do
    case $part in
      p0) run_cfg p0 "" $s ;;
      p1) run_cfg p1 "--emd-weight 0.5" $s ;;
      p2) run_cfg p2 "--train-context-file $DATA/train_contexts.json" $s ;;
      p3) run_cfg p3 "--mask-conflicting-train-contexts" $s ;;
      p4) run_cfg p4 "--transformer-model xlm-roberta-large" $s ;;
      p5) run_cfg p5 "--ordinal-auxiliary --ordinal-weight 0.5" $s ;;
    esac
  done
done
note "ALL DONE $PARTS"
