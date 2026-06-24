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

FastWAM samples actions with a deterministic flow-matching ODE. With the (Wan2.2)
flow convention ``x_t = (1 - t) * data + t * noise`` and velocity ``v = noise -
data`` predicted by the action expert, one Euler denoise step is

    x_{k+1} = x_k + delta_k * v(x_k),   delta_k = t_{k+1} - t_k < 0  (t: 1 -> 0).

For PPO we turn this ODE into a *flow-SDE* so every denoise step becomes a tractable
Gaussian policy ``x_{k+1} ~ N(mean_k, std_k^2)`` and the realized step has a closed-form
log-prob. Two schemes are supported (``noise_method``):

* ``"flow_sde"`` (default) — the **principled reverse-SDE**, matching RLinf's OpenPI /
  GR00T heads: a time-dependent noise scale ``sigma = noise_level * sqrt(t/(1-t))`` and
  a score/drift correction on the predicted-noise component::

        data_pred  = x - t * v                  # x0
        noise_pred = x + (1 - t) * v            # x1
        mean = mean_ode - noise_pred * sigma^2 * |delta| / (2 t)
        std  = sqrt(|delta|) * sigma

  As ``t -> 0`` (towards data) ``sigma -> 0`` so the final steps are nearly
  deterministic — this preserves the model's marginal while adding controllable
  exploration (see ``docs`` / the FastWAM README for the derivation).

* ``"simple"`` — the legacy scheme: deterministic ODE mean + constant-scale isotropic
  noise ``std = noise_level * sqrt|delta|``. Kept for ablation / backward-compat. This
  does **not** preserve the marginal and keeps injecting noise at the last steps.

As in GR00T / OpenPI we (a) record the full denoise ``chains`` during rollout, (b)
sample a single denoise index *per trajectory* whose log-prob enters the PPO ratio, and
(c) recompute that step's mean/std/log-prob under the *current* parameters at train time
(``recompute_logprob``). The video expert's conditioning (KV cache + pooled feature for
the value head) is rebuilt from the stored observation: under ``no_grad`` when the video
expert is frozen, or with grad when ``train_video_expert`` is set (so its weights and the
value head update through the shared conditioning).

All functions are **batch-native**: ``x`` / ``chains`` carry a leading batch dim ``B``
(``B == 1`` is just the degenerate case) and ``denoise_inds`` is a ``[B]`` LongTensor, so
a whole env batch can be processed in one model forward (see ``FastWAMPolicy``'s batched
path). The module only calls public FastWAM model methods (``encode_prompt``,
``_encode_input_image_latents_tensor``, ``_append_proprio_to_context``,
``_build_mot_attention_mask``, ``mot.prefill_video_cache``,
``forward_action_with_video_cache``, ``infer_action_scheduler``) so behaviour stays
aligned with the upstream model.

NOTE (possible divergence from OpenPI): OpenPI controls the flow time convention itself
(clean linear ``t in [0,1]`` with ``x_t = (1-t)x0 + t x1``). Here we *assume* FastWAM's
scheduler ``timesteps`` are exactly that interpolation coefficient (true for the Wan2.2
flow scheduler, including ``sigma_shift`` which only reparametrises within ``[0,1]``). If
a future FastWAM scheduler fed the DiT a timestep that is *not* the interpolation
coefficient, ``data_pred``/``noise_pred`` would be reconstructed at the wrong ``t`` and
the SDE correction would be biased. ``"simple"`` does not rely on this assumption.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Optional

import torch

_LOG_STD_FLOOR = 1e-4  # avoid log(0) / divide-by-zero in the last (tiny) step
_T_EPS = 1e-4  # keep t away from 0 in sigma = noise_level * sqrt(t/(1-t))
# Cap t near 1 for the sigma term. The first denoise step has t == t_model/num_train
# == 1.0 (Wan2.2 shift=5 schedule starts at the noise endpoint), and with _T_EPS=1e-4
# this gave 1-t_c=1e-4 -> sigma = noise*sqrt(0.9999/1e-4) = noise*100, i.e. an action
# noise std ~2.2 at noise_level=0.15 (action range is [-1,1]) -> the very first step
# randomizes the action and the whole rollout collapses. dexbotic avoids this by using
# timesteps[1] in the denominator at t==1 (1-0.978=0.022). We clamp t to 0.98 so the
# first step's sigma stays in the same ~O(1) range as the rest of the chain.
_SIGMA_T_MAX = 0.98


def gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density log N(sample | mean, std^2)."""
    std = std.clamp_min(_LOG_STD_FLOOR)
    return -0.5 * ((sample - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Elementwise differential entropy of N(mean, std^2)."""
    std = std.clamp_min(_LOG_STD_FLOOR)
    return 0.5 * math.log(2 * math.pi * math.e) + torch.log(std)


def _bcast(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Right-pad ``t``'s dims so it broadcasts against ``ref`` ([B,H,D]).

    Accepts a 0-dim scalar (rollout: same step for the whole batch) or a ``[B]``
    vector (training replay: a different denoise step per trajectory).
    """
    while t.ndim < ref.ndim:
        t = t[..., None]
    return t


def flow_step_mean_std(
    x: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    delta: torch.Tensor,
    noise_level: float,
    noise_method: str = "flow_sde",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step flow-SDE Gaussian ``(mean, std)`` for a velocity ``v`` at noise-time ``t``.

    Args:
        x: current latent ``[B, H, D]``.
        v: predicted velocity ``[B, H, D]`` (``= noise - data``).
        t: *normalized* interpolation coefficient ``s = sigma in [0, 1]`` where
            ``x = (1-s)*data + s*noise`` (0-dim scalar or ``[B]``). NOT the raw scheduler
            timestep (which lives in ``[0, num_train_timesteps]``).
        delta: signed scheduler sigma step ``sigma_{k+1} - sigma_k`` (< 0 when denoising;
            0-dim scalar or ``[B]``), already normalized so ``x + delta*v == scheduler.step``.
        noise_level: exploration scale.
        noise_method: ``"flow_sde"`` (principled reverse-SDE) or ``"simple"`` (legacy).
    """
    # Compute the SDE coefficients in fp32: with bf16, the clamp ``1 - _T_EPS`` rounds
    # back to 1.0 (bf16 spacing near 1 is ~8e-3), making ``1 - t_c == 0`` and
    # ``sigma = noise_level*sqrt(t/(1-t))`` blow up to inf/NaN at the first denoise step
    # (s == 1). fp32 also matches OpenPI, which keeps the value/logprob path in fp32.
    xf = x.float()
    vf = v.float()
    t = _bcast(t.float().to(device=x.device), xf)
    delta = _bcast(delta.float().to(device=x.device), xf)
    abs_delta = delta.abs()
    mean_ode = xf + delta * vf  # deterministic Euler step (== scheduler.step)

    if noise_method == "simple":
        mean = mean_ode
        std = (noise_level * torch.sqrt(abs_delta + 1e-12)).expand_as(mean)
        return mean, std

    if noise_method != "flow_sde":
        raise ValueError(
            f"Unknown noise_method={noise_method!r}; expected 'flow_sde' or 'simple'."
        )

    # Principled reverse-SDE (mirrors OpenPI / GR00T flow_sde).
    noise_pred = xf + (1.0 - t) * vf  # x1 (predicted noise)
    t_c = t.clamp(_T_EPS, _SIGMA_T_MAX)
    sigma = noise_level * torch.sqrt(t_c / (1.0 - t_c))
    mean = mean_ode - noise_pred * (sigma**2 * abs_delta / (2.0 * t_c))
    std = (torch.sqrt(abs_delta) * sigma).expand_as(mean)
    return mean, std


@contextlib.contextmanager
def _maybe_grad(enable: bool):
    """``enable_grad`` when ``enable`` else ``no_grad`` (works inside an outer no_grad)."""
    if enable:
        with torch.enable_grad():
            yield
    else:
        with torch.no_grad():
            yield


def _encode_first_frame_latents(model, input_image: torch.Tensor) -> torch.Tensor:
    """First-frame VAE encode for ``input_image`` ``[B, 3, H, W]`` (values in ``[-1, 1]``).

    FastWAM's ``_encode_input_image_latents_tensor`` hard-requires ``B == 1`` (it indexes
    ``[0]`` and wraps the image in a 1-element list). We replicate it batched: the input is
    a *single* frame (``T == 1``), so the VAE's causal temporal cache never couples samples
    and ``vae.model.encode`` runs as plain batched 3D convs. Falls back to the per-sample
    loop on any error, so correctness never depends on the batched path.

    Returns ``first_frame_latents`` ``[B, z, 1, h, w]`` (same as the upstream B=1 call).
    """
    if input_image.shape[0] == 1:
        return model._encode_input_image_latents_tensor(input_image=input_image)
    try:
        vae = model.vae
        # mirror single_encode: [B,3,1,H,W] on device, then the inner model.encode + scale.
        video = input_image.to(device=model.device).unsqueeze(2)  # [B, 3, 1, H, W]
        return vae.model.encode(video, vae.scale)  # [B, z, 1, h, w]
    except Exception as e:  # noqa: BLE001 — degrade to the proven per-sample path
        from rlinf.utils.logging import get_logger
        get_logger().warning(
            "FastWAM RL: batched VAE encode failed (%s); per-sample fallback.", e)
        return torch.cat(
            [model._encode_input_image_latents_tensor(input_image=input_image[i : i + 1])
             for i in range(input_image.shape[0])],
            dim=0,
        )


def build_action_conditioning(
    model,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    enable_grad: bool = False,
):
    """Mirror ``infer_action``'s setup up to the prefilled video KV cache.

    Batched: ``input_image`` ``[B, C, H, W]``, ``context`` ``[B, L, d]``,
    ``context_mask`` ``[B, L]``. When ``enable_grad`` is False the whole computation runs
    under ``no_grad`` (frozen video expert); when True it carries gradients so the video
    expert (and the value head reading ``pooled``) can be trained.

    Returns ``(video_kv_cache, attention_mask, video_seq_len, pooled_video_feat)`` where
    ``pooled_video_feat`` ``[B, dim]`` is the value-head input.

    NOTE: the MoT attention mask is built once from ``(video_seq_len, action_horizon)``
    and shared across the batch — valid because every sample uses the same image
    resolution and action horizon. Per-sample text padding is handled by ``context_mask``.
    """
    with _maybe_grad(enable_grad):
        first_frame_latents = _encode_first_frame_latents(model, input_image)
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
        pooled = video_pre["tokens"].float().mean(dim=1)  # [B, dim]
    return video_kv_cache, attention_mask, video_seq_len, pooled


def _predict_action_velocity(
    model, x, timestep_action, context, context_mask, video_kv_cache, attention_mask, video_seq_len
):
    """Action-expert velocity prediction (grad-enabled twin of FastWAM's
    ``_predict_action_noise_with_cache``, which is ``@torch.no_grad`` and would detach the
    graph during the training replay). ``timestep_action`` is ``[B]``.

    ``x`` may be fp32 (the SDE math runs in fp32); cast to the model dtype for the forward
    (grad flows through the cast, preserving the reentrant-checkpoint param gradients)."""
    x = x.to(dtype=model.torch_dtype)
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


def _step_mean_std(
    model, x, t_model, s, delta, context, context_mask, video_kv_cache, attention_mask,
    video_seq_len, noise_level, noise_method,
):
    """Velocity prediction + flow-SDE mean/std for one denoise step.

    ``t_model`` is the *raw* scheduler timestep in ``[0, num_train_timesteps]`` fed to the
    action expert (that is what it was trained on). ``s = t_model/num_train_timesteps`` is
    the normalized interpolation coefficient (``x = (1-s)*data + s*noise``) used by the SDE
    math. ``delta`` is the scheduler's already-normalized sigma step (``sigma_{k+1}-sigma_k``,
    so ``mean_ode = x + delta*v`` equals ``scheduler.step``). Each may be a scalar or ``[B]``.
    """
    batch = x.shape[0]
    # Feed the model in its own dtype (x may be fp32 from the SDE math).
    t_action = _bcast_vec(t_model, batch).to(dtype=model.torch_dtype, device=model.device)
    v = _predict_action_velocity(
        model, x, t_action, context, context_mask,
        video_kv_cache, attention_mask, video_seq_len,
    )
    return flow_step_mean_std(x, v, s, delta, noise_level, noise_method)


def _bcast_vec(t: torch.Tensor, batch: int) -> torch.Tensor:
    """Expand a 0-dim ``t`` to ``[batch]`` (the action expert expects a per-sample timestep)."""
    if t.ndim == 0:
        return t.reshape(1).expand(batch)
    return t


@torch.no_grad()
def flow_sde_rollout(
    model,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    noise_level: float,
    noise_method: str = "flow_sde",
    deterministic: bool = False,
):
    """Sample an action chunk via the flow-SDE, recording the denoise chain. Batched.

    Returns ``(final_action[B,H,D], info)`` where ``info`` has ``chains``
    ``[B, num_steps+1, H, D]``, ``logp_per_step`` ``[B, num_steps, H, D]``,
    ``denoise_inds`` ``[B]`` (LongTensor), and ``pooled_video_feat`` ``[B, dim]``.
    """
    video_kv_cache, attention_mask, video_seq_len, pooled = build_action_conditioning(
        model, input_image, context, context_mask, action_horizon, enable_grad=False
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=model.torch_dtype,
    )
    nt = float(getattr(model.infer_action_scheduler, "num_train_timesteps", 1000))
    batch = int(input_image.shape[0])
    action_dim = int(model.action_expert.action_dim)
    # Work in fp32 (chains / logprobs); the model forward casts back to its dtype.
    x = torch.randn((batch, action_horizon, action_dim), device=model.device, dtype=torch.float32)

    chains = [x]
    logps = []
    for k in range(num_inference_steps):
        mean, std = _step_mean_std(
            model, x, timesteps[k], timesteps[k] / nt, deltas[k], context, context_mask,
            video_kv_cache, attention_mask, video_seq_len, noise_level, noise_method,
        )
        if deterministic:
            x = mean
            logp = torch.zeros_like(mean)
        else:
            x = mean + std * torch.randn_like(mean)
            logp = gaussian_logprob(x, mean, std)
        chains.append(x)
        logps.append(logp)

    chains_t = torch.stack(chains, dim=1)  # [B, num_steps+1, H, D]
    logp_t = torch.stack(logps, dim=1)  # [B, num_steps, H, D]
    # One sampled denoise index per trajectory whose log-prob enters the PPO ratio.
    denoise_inds = torch.randint(0, num_inference_steps, (batch,), device=model.device)
    info = {
        "chains": chains_t,
        "logp_per_step": logp_t,
        "denoise_inds": denoise_inds,
        "pooled_video_feat": pooled,
    }
    return x, info


def recompute_logprob(
    model,
    input_image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    chains: torch.Tensor,
    denoise_inds: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    noise_level: float,
    noise_method: str = "flow_sde",
    train_video_expert: bool = False,
):
    """Recompute the sampled step's per-element log-prob + entropy under current params.

    Batched: ``chains`` ``[B, num_steps+1, H, D]`` and a per-sample ``denoise_inds`` ``[B]``.
    Each trajectory replays *its own* denoise step in a single batched action-expert
    forward (per-sample timestep), exactly like OpenPI/GR00T's tensor-``idx`` path.

    The video conditioning is rebuilt with grad iff ``train_video_expert`` (so the video
    expert + value head train through it); otherwise under ``no_grad`` (frozen expert,
    only the action expert at ``denoise_inds`` carries gradients).

    Returns ``(logp[B,H,D], entropy[B,H,D], pooled_video_feat[B,dim])``.
    """
    video_kv_cache, attention_mask, video_seq_len, pooled = build_action_conditioning(
        model, input_image, context, context_mask, action_horizon, enable_grad=train_video_expert
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=model.torch_dtype,
    )
    batch = int(chains.shape[0])
    if denoise_inds.ndim == 0:
        denoise_inds = denoise_inds.reshape(1).expand(batch)
    denoise_inds = denoise_inds.to(device=chains.device, dtype=torch.long)
    arange = torch.arange(batch, device=chains.device)

    # Gather the per-sample (x_k, x_{k+1}) and (t_k, delta_k).
    x_pre = chains[arange, denoise_inds]  # [B, H, D]
    x_next = chains[arange, denoise_inds + 1]
    t = timesteps[denoise_inds]  # [B]
    delta = deltas[denoise_inds]  # [B]

    # Enable grad on the replayed input so reentrant gradient checkpointing inside the
    # action expert tracks and produces parameter gradients (the input's own gradient is
    # unused/discarded). Without this, a fully-detached input makes use_reentrant
    # checkpoint skip param-grad computation.
    x_pre = x_pre.detach().requires_grad_(True)
    nt = float(getattr(model.infer_action_scheduler, "num_train_timesteps", 1000))
    mean, std = _step_mean_std(
        model, x_pre, t, t / nt, delta, context, context_mask,
        video_kv_cache, attention_mask, video_seq_len, noise_level, noise_method,
    )
    logp = gaussian_logprob(x_next, mean, std)
    ent = gaussian_entropy(std)
    return logp, ent, pooled
