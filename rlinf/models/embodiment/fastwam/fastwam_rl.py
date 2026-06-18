# Copyright 2026 The RLinf Authors.
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

"""Flow-SDE reinforcement-learning core for FastWAM's action expert.

FastWAM samples actions with a deterministic flow-matching ODE: ``x_{k+1} =
x_k + delta_k * v(x_k)`` where ``v = dx/dsigma`` is the velocity predicted by the
action expert (conditioned on the frozen video expert's KV cache). For PPO we turn
this ODE into a *flow-SDE* by adding isotropic Gaussian exploration noise at each
denoise step, which makes every step a tractable Gaussian policy:

    x_{k+1} ~ N( mean_k = x_k + delta_k * v(x_k),  (noise_level * sqrt|delta_k|)^2 )

The per-step log-prob is the Gaussian density of the realized ``x_{k+1}``. As in
RLinf's GR00T / OpenPI flow-matching RL heads, we (a) record the full denoise
``chains`` during rollout, (b) sample a single denoise index per trajectory whose
log-prob enters the PPO ratio, and (c) recompute that step's mean/std/log-prob
under the *current* action-expert parameters at train time (``recompute_logprob``).
The 5B video expert is frozen, so its conditioning (KV cache + pooled feature for
the value head) is rebuilt under ``no_grad`` from the stored observation.

This module is intentionally self-contained: it only calls public FastWAM model
methods (``encode_prompt``, ``_encode_input_image_latents_tensor``,
``_append_proprio_to_context``, ``_build_mot_attention_mask``,
``mot.prefill_video_cache``, ``_predict_action_noise_with_cache``,
``infer_action_scheduler``) so behaviour stays aligned with the upstream model.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

_LOG_STD_FLOOR = 1e-4  # avoid log(0) / divide-by-zero in the last (tiny) step


def gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density log N(sample | mean, std^2)."""
    std = std.clamp_min(_LOG_STD_FLOOR)
    return -0.5 * ((sample - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Elementwise differential entropy of N(mean, std^2)."""
    std = std.clamp_min(_LOG_STD_FLOOR)
    return 0.5 * math.log(2 * math.pi * math.e) + torch.log(std)


@torch.no_grad()
def build_action_conditioning(model, input_image: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, action_horizon: int):
    """Mirror ``infer_action``'s setup up to the prefilled video KV cache.

    The video expert is frozen, so this whole computation is ``no_grad`` and can be
    rebuilt identically at train time from the stored observation.

    Returns ``(video_kv_cache, attention_mask, video_seq_len, pooled_video_feat)``
    where ``pooled_video_feat`` ``[1, dim]`` is the value-head input.
    """
    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    timestep_video = torch.zeros(
        (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=model.device
    )
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_horizon,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    pooled = video_pre["tokens"].float().mean(dim=1)  # [1, dim]
    return video_kv_cache, attention_mask, video_seq_len, pooled


def _predict_action_velocity(model, x, timestep_action, context, context_mask, video_kv_cache, attention_mask, video_seq_len):
    """Action-expert velocity prediction (grad-enabled twin of FastWAM's
    ``_predict_action_noise_with_cache``, which is ``@torch.no_grad`` and would
    detach the graph during the training replay)."""
    action_pre = model.action_expert.pre_dit(
        action_tokens=x,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    action_tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return model.action_expert.post_dit(action_tokens, action_pre)


def _step_mean_std(model, x, k, timesteps, deltas, context, context_mask, video_kv_cache, attention_mask, video_seq_len, noise_level):
    """Velocity prediction + flow-SDE mean/std for denoise step ``k``."""
    timestep_action = timesteps[k].unsqueeze(0).to(dtype=x.dtype, device=model.device)
    v = _predict_action_velocity(
        model, x, timestep_action, context, context_mask,
        video_kv_cache, attention_mask, video_seq_len,
    )
    mean = x + deltas[k] * v  # deterministic flow step (== scheduler.step)
    std = torch.full_like(mean, float(noise_level) * math.sqrt(abs(float(deltas[k])) + 1e-12))
    return mean, std


@torch.no_grad()
def flow_sde_rollout(
    model,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    noise_level: float,
    deterministic: bool = False,
):
    """Sample an action chunk via the flow-SDE, recording the denoise chain.

    Returns ``(final_action[1,H,D], info)`` where ``info`` has ``chains``
    ``[1, num_steps+1, H, D]``, ``logp_per_step`` ``[1, num_steps, H, D]``,
    ``denoise_ind`` (int), and ``pooled_video_feat`` ``[1, dim]``.
    """
    video_kv_cache, attention_mask, video_seq_len, pooled = build_action_conditioning(
        model, input_image, context, context_mask, action_horizon
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=model.torch_dtype,
    )
    action_dim = int(model.action_expert.action_dim)
    x = torch.randn((1, action_horizon, action_dim), device=model.device, dtype=model.torch_dtype)

    chains = [x]
    logps = []
    for k in range(num_inference_steps):
        mean, std = _step_mean_std(
            model, x, k, timesteps, deltas, context, context_mask,
            video_kv_cache, attention_mask, video_seq_len, noise_level,
        )
        if deterministic:
            x = mean
            logp = torch.zeros_like(mean)
        else:
            x = mean + std * torch.randn_like(mean)
            logp = gaussian_logprob(x, mean, std)
        chains.append(x)
        logps.append(logp)

    chains_t = torch.stack(chains, dim=1)  # [1, num_steps+1, H, D]
    logp_t = torch.stack(logps, dim=1)  # [1, num_steps, H, D]
    # One sampled denoise index whose log-prob enters the PPO ratio (joint_logprob=False).
    denoise_ind = int(torch.randint(0, num_inference_steps, (1,)).item())
    info = {
        "chains": chains_t,
        "logp_per_step": logp_t,
        "denoise_ind": denoise_ind,
        "pooled_video_feat": pooled,
    }
    return x, info


def recompute_logprob(
    model,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    chains: torch.Tensor,
    denoise_ind: int,
    action_horizon: int,
    num_inference_steps: int,
    noise_level: float,
):
    """Recompute the sampled step's per-element log-prob + entropy under current params.

    Video conditioning is rebuilt under ``no_grad`` (frozen expert); only the action
    expert velocity at step ``denoise_ind`` carries gradients.

    Returns ``(logp[1,H,D], entropy[1,H,D], pooled_video_feat[1,dim])``.
    """
    video_kv_cache, attention_mask, video_seq_len, pooled = build_action_conditioning(
        model, input_image, context, context_mask, action_horizon
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=model.torch_dtype,
    )
    # Enable grad on the replayed input so reentrant gradient checkpointing inside
    # the action expert tracks and produces parameter gradients (the input's own
    # gradient is unused/discarded). Without this, a fully-detached input makes
    # use_reentrant checkpoint skip param-grad computation.
    x_pre = chains[:, denoise_ind].detach().requires_grad_(True)
    x_next = chains[:, denoise_ind + 1]
    mean, std = _step_mean_std(
        model, x_pre, denoise_ind, timesteps, deltas, context, context_mask,
        video_kv_cache, attention_mask, video_seq_len, noise_level,
    )
    logp = gaussian_logprob(x_next, mean, std)
    ent = gaussian_entropy(std)
    return logp, ent, pooled
