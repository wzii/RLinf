#!/bin/bash
# FastWAM RL (GRPO) launcher on LIBERO (RLinf). Config-driven (libero_variant in yaml).
set -eo pipefail
source /workspace/venv_fw/bin/activate

export REPO_PATH=/workspace/RLinf
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export PYTHONPATH="${REPO_PATH}:/workspace/LIBERO:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/workspace/.hf_home
export FASTWAM_CONFIG_DIR=/workspace/FastWAM/configs

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# NOTE: do NOT enable expandable_segments — its CUDA VMM allocator forces fd-based
# CUDA IPC (pidfd_getfd), which is blocked in this container. The legacy allocator
# uses cudaIpcMemHandle for the actor->rollout weight sync, which works here.
export ROBOT_PLATFORM=LIBERO

CONFIG_NAME="${1:-libero_spatial_grpo_fastwam}"; shift || true
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  "$@"
