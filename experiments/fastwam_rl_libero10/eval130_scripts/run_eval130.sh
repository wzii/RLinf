#!/bin/bash
# Drive all LIBERO-130 deterministic evals (BASE round15 + RL step10) x 5 suites,
# sequentially, with clean GPU teardown between runs. 100 envs, 50 ep/task.
# Each run -> /workspace/results/<exp>/run.log ; SR read from tensorboard afterwards.
set -uo pipefail

RESUME_DIR=/workspace/results/lib10_rl_n6/lib10_rl_n6/checkpoints/global_step_10
ROUND15=/workspace/fastwam_ckpt/round15_robust_step_005000.pt
CFG=libero_spatial_grpo_fastwam_h240

# name suite horizon rollout_epoch(@100envs -> 50 ep/task)
SUITES=(
  "long libero_10 520 5"
  "spatial libero_spatial 240 5"
  "object libero_object 280 5"
  "goal libero_goal 300 5"
  "short libero_90 240 45"
)
# ckpt order per suite: rl first on 'long' (correctness anchor ~0.90), else base then rl
CKPTS=("base" "rl")

teardown() {
  for r in 1 2 3 4 5; do
    pkill -9 -f train_embodied_agent.py 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f raylet 2>/dev/null || true
    sleep 6
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    echo "  teardown round $r: total GPU MiB used=$used"
    [ "${used:-9999}" -lt 2000 ] && break
  done
  sleep 4
}

run_one() {
  local ckpt=$1 name=$2 suite=$3 hz=$4 ep=$5
  local exp="eval130_${ckpt}_${name}"
  local dir="/workspace/results/${exp}"
  mkdir -p "$dir"
  local extra=""
  [ "$ckpt" = "rl" ] && extra="runner.resume_dir=${RESUME_DIR}"
  echo "=========================================================="
  echo "[$(date +%H:%M:%S)] RUN $exp  suite=$suite hz=$hz epochs=$ep  extra='$extra'"
  echo "=========================================================="
  bash /workspace/run_eval.sh "$CFG" \
    runner.max_steps=0 +runner.eval_at_start=true runner.save_interval=-1 \
    $extra \
    runner.logger.log_path="$dir" runner.logger.experiment_name="$exp" \
    actor.model.checkpoint_path="$ROUND15" \
    env.eval.task_suite_name="$suite" \
    env.eval.max_episode_steps="$hz" env.eval.max_steps_per_rollout_epoch="$hz" \
    env.eval.total_num_envs=100 env.eval.rollout_epoch="$ep" \
    env.train.task_suite_name="$suite" \
    env.train.max_episode_steps="$hz" env.train.max_steps_per_rollout_epoch="$hz" \
    > "${dir}/run.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] DONE $exp rc=$rc  (last log line:)"
  tail -1 "${dir}/run.log"
  teardown
}

# Anchor: RL on long first, then base on long, then the rest base+rl per suite.
run_one rl   long    libero_10      520 5
run_one base long    libero_10      520 5
for entry in "${SUITES[@]:1}"; do
  read -r name suite hz ep <<< "$entry"
  for ckpt in "${CKPTS[@]}"; do
    run_one "$ckpt" "$name" "$suite" "$hz" "$ep"
  done
done
echo "ALL EVAL130 RUNS COMPLETE at $(date +%H:%M:%S)"
