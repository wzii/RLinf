#!/bin/bash
# FastWAM LIBERO EVAL launcher (eval-only, no training). Mirrors run_rl.sh but points
# DiffSynth + HF caches at /dev/shm (RAM, 469G free) so the ~11GB T5+VAE backbone
# re-download does NOT touch the tight /workspace disk (8G free). Skip-DiT eval:
# the video DiT is supplied by the checkpoint; only T5+VAE+tokenizer are fetched.
set -eo pipefail
source /workspace/venv_fw/bin/activate

export REPO_PATH=/workspace/RLinf
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export PYTHONPATH="${REPO_PATH}:/workspace/LIBERO:${PYTHONPATH:-}"

# RAM-backed model/cache roots (no /workspace disk cost)
export DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/dev/shm/hf_home
export FASTWAM_CONFIG_DIR=/workspace/FastWAM/configs

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM=LIBERO
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

CONFIG_NAME="${1:-libero_spatial_grpo_fastwam_h240}"; shift || true
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  "$@"
