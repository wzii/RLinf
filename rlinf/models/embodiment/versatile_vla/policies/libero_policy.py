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
"""Bridge between RLinf's ``LiberoEnv`` observation format and the
``cloudrobovla`` (Versatile VLA) policy request format.

RLinf ``LiberoEnv._wrap_obs`` produces (see ``rlinf/envs/libero/libero_env.py``)::

    {
        "main_images":       tensor [B, H, W, C] uint8,   # agentview, already [::-1,::-1] flipped
        "wrist_images":      tensor [B, H, W, C] uint8,   # wrist, already flipped
        "states":            tensor [B, state_dim],       # eef_pos + quat2axisangle + gripper_qpos
        "task_descriptions": list[str],
    }

The VersaVLA ``infer``/``infer_batch`` request element expects (see
``cloudrobovla/evaluation/libero/code/eval_libero_pi.py::_build_policy_request_element``)::

    {
        "left_third_image": np.uint8 [H, W, C],   # agentview
        "left_wrist_image": np.uint8 [H, W, C],   # wrist
        "state":            np.float [state_dim],
        "repo_id":          str,                  # resolves normalization stats, e.g. "libero_all_pi_v3.0"
        "prompt":           str,                  # task description
    }

Crucially, both sides already apply the ``[::-1, ::-1]`` 180-degree image flip
(RLinf in ``rlinf/envs/libero/utils.py::get_libero_image``), so no extra flip
is needed here. Images stay uint8 HWC and are resized internally by the
VersaVLA ``VisionPreprocessor`` (Qwen ``smart_resize``, factor=28).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """Convert a single image tensor to uint8 HWC numpy array.

    Accepts [H, W, C] or [C, H, W]; float images are scaled to 0-255.
    """
    img = image.detach()
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[-1] > 4:
        # [C, H, W] -> [H, W, C]
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def env_obs_to_versa_items(
    env_obs: dict[str, Any],
    *,
    repo_id: str,
) -> list[dict[str, Any]]:
    """Convert a batched RLinf ``env_obs`` into a list of VersaVLA request items.

    Args:
        env_obs: observation dict from ``LiberoEnv`` (see module docstring).
        repo_id: VersaVLA dataset repo id used to resolve normalization stats
            (e.g. ``"libero_all_pi_v3.0"``).

    Returns:
        List of per-env request dicts, one per environment in the batch.
    """
    main_images = env_obs["main_images"]  # [B, H, W, C]
    wrist_images = env_obs["wrist_images"]  # [B, H, W, C]
    states = env_obs["states"]  # [B, state_dim]
    task_descriptions = env_obs["task_descriptions"]  # list[str]

    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    states = np.asarray(states)

    batch_size = main_images.shape[0]
    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions] * batch_size

    items: list[dict[str, Any]] = []
    for i in range(batch_size):
        items.append(
            {
                "left_third_image": _to_uint8_hwc(main_images[i]),
                "left_wrist_image": _to_uint8_hwc(wrist_images[i]),
                "state": np.asarray(states[i], dtype=np.float32),
                "repo_id": repo_id,
                "prompt": str(task_descriptions[i]),
            }
        )
    return items
