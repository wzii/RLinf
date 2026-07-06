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
"""RLinf ``BasePolicy`` wrapper around the Versatile VLA (``cloudrobovla``) model.

This wraps the already-validated VersaVLA inference path
(``LIBEROVersaVLAAdapter.infer``) behind RLinf's ``BasePolicy`` interface so
that VersaVLA can be driven by RLinf's rollout / eval workers.

Design (Stage A - inference + eval only):
    The wrapper *composes* a ``LIBEROVersaVLAAdapter`` instance (which itself
    loads the ``VersaVLA0Model`` HF checkpoint + preprocessors + normalization
    stats). Inference is delegated to ``adapter.infer_batch``. The underlying
    ``nn.Module`` (``adapter.model``) is exposed so RLinf's FSDP / weight-sync
    machinery can reach the real parameters once Stage B/C land.

    ``default_forward`` (RL training logprob/value) and ``sft_forward`` are
    Stage B/C and intentionally raise ``NotImplementedError`` for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.versatile_vla.policies.libero_policy import (
    env_obs_to_versa_items,
)


@dataclass
class VersaVLAConfig:
    """Subset of RLinf model cfg needed to build the VersaVLA policy."""

    model_path: str = ""
    repo_id: str = "libero_all_pi_v3.0"
    action_dim: int = 7
    num_action_chunks: int = 10
    denoise_steps: int = 5
    device: str = "cuda"
    # Pass-through VersaVLA PolicyRuntimeArgs overrides (optional).
    target_action_space: str = "delta"
    data_mixture_name: str = "LIBERO-ALL-PI_3.0"
    precision_preset: str = "bf16"
    extra_policy_args: dict = field(default_factory=dict)


class VersaVLAForRLActionPrediction(BasePolicy):
    """RLinf policy backed by a VersaVLA ``LIBEROVersaVLAAdapter``.

    The wrapped ``LIBEROVersaVLAAdapter`` owns the actual ``VersaVLA0Model``
    (an ``nn.Module``); this class forwards ``nn.Module`` attribute access to
    it so that ``parameters()`` / ``state_dict()`` / device placement work.
    """

    def __init__(self, cfg: VersaVLAConfig):
        self.cfg = cfg

        # Build the VersaVLA LIBERO adapter. Imported lazily so that RLinf does
        # not require cloudrobovla unless this model is actually used.
        from cloudrobovla.inference.libero.adapter import (
            LIBEROVersaVLAAdapter,
            LIBEROVersaVLAArgs,
        )

        args = LIBEROVersaVLAArgs(
            pretrained_model_path=cfg.model_path,
            device=cfg.device,
            precision_preset=cfg.precision_preset,
            action_chunk=cfg.num_action_chunks,
            original_action_dim=cfg.action_dim,
            denoise_steps=cfg.denoise_steps,
            data_mixture_name=cfg.data_mixture_name,
            target_action_space=cfg.target_action_space,
            **cfg.extra_policy_args,
        )
        # LIBEROVersaVLAAdapter.__init__ loads the model + preprocessors + stats.
        self.adapter = LIBEROVersaVLAAdapter(args=args)
        # The real nn.Module (VersaVLA0Model). Exposed for FSDP/weight sync.
        self.model = self.adapter.model

        self.action_dim = cfg.action_dim
        self.num_action_chunks = cfg.num_action_chunks
        self.repo_id = cfg.repo_id

    # ------------------------------------------------------------------
    # nn.Module compatibility: forward attribute access to the real model so
    # that RLinf workers calling .parameters() / .to() / .state_dict() work.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails. Delegate to the
        # underlying nn.Module. Guard against recursion during __init__.
        adapter = self.__dict__.get("adapter")
        model = self.__dict__.get("model")
        if model is not None and hasattr(model, name):
            return getattr(model, name)
        if adapter is not None and hasattr(adapter, name):
            return getattr(adapter, name)
        raise AttributeError(name)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        self.adapter.model = self.model
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: Optional[dict[str, Any]] = None,
        calculate_logprobs: bool = False,
        calculate_values: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a batch of action chunks from RLinf env observations.

        Returns:
            (chunk_actions, result) where chunk_actions is
            ``[B, num_action_chunks, action_dim]`` and result carries the
            fields RLinf's rollout worker expects (``prev_logprobs``,
            ``prev_values``, ``forward_inputs``). For Stage A (eval-only)
            logprobs/values are zeros since no RL training is performed.
        """
        do_sample = kwargs.pop("do_sample", False)
        # VersaVLA inference is deterministic Euler flow-matching sampling;
        # `do_sample` (stochasticity) is not exposed in Stage A.

        items = env_obs_to_versa_items(env_obs, repo_id=self.repo_id)
        # infer_batch returns a list of {"actions": [1, chunk, dim], ...}
        infer_outputs = self.adapter.infer_batch(items)

        batch_size = len(infer_outputs)
        chunk_actions = np.zeros(
            (batch_size, self.num_action_chunks, self.action_dim),
            dtype=np.float32,
        )
        for i, out in enumerate(infer_outputs):
            acts = np.asarray(out["actions"], dtype=np.float32)  # [1, chunk, dim]
            acts = acts[0]  # [chunk, dim]
            n = min(acts.shape[0], self.num_action_chunks)
            d = min(acts.shape[1], self.action_dim)
            chunk_actions[i, :n, :d] = acts[:n, :d]

        # Stage A: no logprob / value computation. Provide zero placeholders so
        # the rollout worker's downstream code paths do not break during eval.
        zero_lp = torch.zeros(
            (batch_size, self.num_action_chunks, self.action_dim),
            dtype=torch.float32,
        )
        zero_val = torch.zeros((batch_size, 1), dtype=torch.float32)

        result = {
            "prev_logprobs": zero_lp,
            "prev_values": zero_val,
            "forward_inputs": {},
        }
        return chunk_actions, result

    def default_forward(self, **kwargs):
        # Stage C: flow-matching logprob + value head. Not implemented yet.
        raise NotImplementedError(
            "VersaVLA RL training (default_forward) is not implemented yet "
            "(Stage C). Use eval-only mode for now."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"VersaVLA forward_type={forward_type} is not supported."
        )
