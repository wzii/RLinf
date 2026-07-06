# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Factory for building the VersaVLA RLinf policy from a Hydra model config.

Unlike openpi (which hand-loads safetensors and patches PaliGemma), VersaVLA is
a standard HuggingFace ``PreTrainedModel`` (``model_type="VersaVLA0"`` with a
Qwen3-VL backbone), so loading is delegated to ``LIBEROVersaVLAAdapter``, which
calls ``VersaVLA0Model.from_pretrained`` internally and wires up the
preprocessors + normalization stats.
"""

from typing import Optional

from omegaconf import DictConfig

from rlinf.models.embodiment.versatile_vla.versa_vla_action_model import (
    VersaVLAConfig,
    VersaVLAForRLActionPrediction,
)


def get_model(cfg: DictConfig, torch_dtype=None) -> VersaVLAForRLActionPrediction:
    # VersaVLA-specific knobs may live under a `versa_vla:` sub-config.
    versa = getattr(cfg, "versa_vla", None) or {}

    def _get(key, default):
        if hasattr(versa, key):
            return getattr(versa, key)
        return getattr(cfg, key, default)

    versa_cfg = VersaVLAConfig(
        model_path=cfg.model_path,
        repo_id=_get("repo_id", "libero_all_pi_v3.0"),
        action_dim=_get("action_dim", 7),
        num_action_chunks=_get("num_action_chunks", 10),
        denoise_steps=_get("denoise_steps", 5),
        device=_get("device", "cuda"),
        target_action_space=_get("target_action_space", "delta"),
        data_mixture_name=_get("data_mixture_name", "LIBERO-ALL-PI_3.0"),
        precision_preset=_get("precision_preset", "bf16"),
    )
    return VersaVLAForRLActionPrediction(versa_cfg)
