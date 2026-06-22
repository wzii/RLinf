#!/bin/bash
# FastWAM SFT launcher (RLinf) — libero_spatial, controlled env.
set -eo pipefail
source /workspace/venv_fw/bin/activate

export REPO_PATH=/workspace/RLinf
export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export PYTHONPATH="${REPO_PATH}:/workspace/LIBERO:${PYTHONPATH:-}"

# Wan2.2 model files (VAE/T5/tokenizer) resolve from here.
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME=/workspace/.hf_home

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTWAM_CONFIG_DIR=/workspace/FastWAM/configs

python "${EMBODIED_PATH}/train_vla_sft.py" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name libero_sft_fastwam \
  "$@"
