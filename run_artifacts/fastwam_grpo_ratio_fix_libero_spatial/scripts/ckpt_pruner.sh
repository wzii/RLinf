#!/bin/bash
# Checkpoint pruner for FastWAM RL runs.
# Each RLinf checkpoint = global_step_N/actor/{dcp_checkpoint (model+optimizer shards,
# large), model_state_dict/full_weights.pt (consolidated, directly loadable)}.
# To bound disk for save-every-10-steps we: (1) drop the big dcp_checkpoint optimizer
# dir as soon as full_weights.pt is written, (2) keep only the last KEEP checkpoints.
# Usage: ckpt_pruner.sh <checkpoints_dir> [KEEP]
set -u
CKPT_DIR="$1"
KEEP="${2:-3}"
echo "[pruner] watching $CKPT_DIR (keep last $KEEP, drop dcp optimizer dirs)"
while true; do
  if [ -d "$CKPT_DIR" ]; then
    # 1) drop dcp_checkpoint once the consolidated full_weights.pt exists for that step
    for d in "$CKPT_DIR"/global_step_*/actor; do
      [ -d "$d" ] || continue
      if [ -f "$d/model_state_dict/full_weights.pt" ] && [ -d "$d/dcp_checkpoint" ]; then
        echo "[pruner] $(date +%T) dropping optimizer dcp: $d/dcp_checkpoint"
        rm -rf "$d/dcp_checkpoint"
      fi
    done
    # 2) keep only the last KEEP global_step_* dirs
    mapfile -t steps < <(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n)
    n=${#steps[@]}
    if [ "$n" -gt "$KEEP" ]; then
      for ((i=0; i<n-KEEP; i++)); do
        echo "[pruner] $(date +%T) removing old ckpt: ${steps[$i]}"
        rm -rf "${steps[$i]}"
      done
    fi
  fi
  sleep 20
done
