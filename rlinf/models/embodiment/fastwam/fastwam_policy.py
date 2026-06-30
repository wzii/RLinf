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

"""RLinf policy wrapper around the upstream FastWAM world-action model.

FastWAM (``Fast-WAM: Do World Action Models Need Test-time Future Imagination?``)
is a Wan2.2 video-diffusion world model with a flow-matching action expert. This
module adapts its self-contained inference / training API to RLinf's
:class:`~rlinf.models.embodiment.base_policy.BasePolicy` contract so the model can
be evaluated (LIBERO / LIBERO-Plus) and SFT-trained through RLinf workers.

The wrapper deliberately *reuses* upstream FastWAM code (``infer_action`` for
rollout, ``training_loss`` for SFT, ``FastWAMProcessor`` for normalization) rather
than re-implementing the model, so behaviour matches the official repository.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.fastwam import fastwam_rl as R
from rlinf.utils.logging import get_logger

logger = get_logger()

# Upstream FastWAM. The prompt template and gripper helper are vendored here as
# small constants to avoid importing the (non-packaged) ``experiments`` scripts.
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip the gripper open/close sign (matches FastWAM ``invert_gripper_action``)."""
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def _center_crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize + center crop to (height, width).

    Mirrors ``experiments/libero/eval_libero_single.py::_center_crop_resize`` so
    the pixels handed to the model match the official LIBERO evaluation exactly.
    """
    from PIL import Image

    pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


class FastWAMPolicy(nn.Module, BasePolicy):
    """Adapter exposing FastWAM through RLinf's embodied policy interface.

    Args:
        model: An instantiated upstream ``fastwam.models.wan22.fastwam.FastWAM``.
        processor: A ``FastWAMProcessor`` whose normalizer is already populated
            from ``dataset_stats.json`` (action/state min-max stats).
        infer_cfg: Resolved inference hyper-parameters (see :class:`FastWAMInferConfig`).
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        infer_cfg: "FastWAMInferConfig",
        value_head: Optional[nn.Module] = None,
        rl_cfg: Optional[dict] = None,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.processor = processor
        self.infer_cfg = infer_cfg
        # Optional RL head + config (see fastwam_rl). When `value_head` is set the
        # policy is RL-capable: predict_action_batch samples via the flow-SDE and
        # returns prev_logprobs/prev_values, and default_forward replays the chain.
        self.value_head = value_head
        self.rl_cfg = dict(rl_cfg or {})
        # RL-capable (flow-SDE sampling + logprobs) whenever RL is enabled, *independent*
        # of whether a value head exists. Critic-free GRPO (adv_type=grpo, loss_type=actor)
        # runs with value_head=None; PPO/GAE (actor_critic) additionally attaches one.
        # NB: FastWAM's natural value features are inadequate for a critic -- the pre-DiT
        # video "tokens" are a single Conv3d of the *current frame's* VAE latent, with no
        # task/proprio/world-model conditioning (text+proprio live in `context`, fused only
        # inside the DiT cross-attention) -- so GRPO is the recommended algorithm here.
        self._rl_enabled = bool(self.rl_cfg.get("enabled", False)) or value_head is not None
        self._rl_noise_level = float(self.rl_cfg.get("noise_level", 0.1))
        # "flow_sde" = principled reverse-SDE (default, matches OpenPI/GR00T);
        # "simple" = legacy ODE-mean + constant-scale noise. See fastwam_rl.
        self._rl_noise_method = str(self.rl_cfg.get("noise_method", "flow_sde"))
        # Where to inject SDE noise during training rollout (see flow_sde_rollout):
        # "single" (default, OpenPI-aligned) = one stochastic denoise step per rollout,
        # rest are ODE steps; "full" = every step stochastic (legacy full-chain SDE).
        # Eval always runs the pure ODE regardless of this flag.
        self._rl_sde_sampling = str(self.rl_cfg.get("sde_sampling", "single"))
        # Train the (otherwise frozen) video expert through the rebuilt conditioning.
        self._rl_train_video_expert = bool(self.rl_cfg.get("train_video_expert", False))
        # Process the whole env batch in one model forward when True; fall back to the
        # per-env loop on any error (e.g. variable-length text context). See _rl_* below.
        self._rl_batched = bool(self.rl_cfg.get("batched", True))
        # Cap the rollout sub-batch through the frozen 5B video forward so peak memory is
        # bounded independent of how many envs a worker holds (lets total_num_envs scale).
        self._rl_rollout_chunk = int(self.rl_cfg.get("rollout_chunk", 8))

        # Cache the (single) merged action/state keys used by the processor.
        self._state_key = processor.shape_meta["state"][0]["key"]
        self._action_key = processor.shape_meta["action"][0]["key"]
        # Per-camera target HxW from shape_meta (e.g. [3,224,224] -> 224,224).
        self._cam_hw = [
            (int(m["shape"][1]), int(m["shape"][2]))
            for m in processor.shape_meta["images"]
        ]
        self._num_cameras = int(processor.num_output_cameras)
        self._video_saves = 0  # count of predicted-future clips saved so far

    def _save_future_video(self, frames) -> None:
        """Save a predicted future-video clip (list of PIL frames) as an MP4."""
        import os

        from fastwam.utils.video_io import save_mp4

        out_dir = self.infer_cfg.future_video_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"future_{self._video_saves:04d}.mp4")
        save_mp4(frames, path, fps=8)
        self._video_saves += 1

    # ------------------------------------------------------------------ utils
    @property
    def device(self):
        return getattr(self.model, "device", next(self.model.parameters()).device)

    @property
    def torch_dtype(self):
        return getattr(self.model, "torch_dtype", torch.bfloat16)

    def _build_input_image(self, main_img, wrist_img) -> torch.Tensor:
        """Build the FastWAM input frame: per-cam crop/resize, h-concat, [-1, 1]."""
        main_np = main_img.detach().cpu().numpy() if torch.is_tensor(main_img) else np.asarray(main_img)
        if self._num_cameras == 1:
            h, w = self._cam_hw[0]
            rgb = _center_crop_resize(main_np, width=w, height=h)
        else:
            h0, w0 = self._cam_hw[0]
            h1, w1 = self._cam_hw[1]
            primary = _center_crop_resize(main_np, width=w0, height=h0)
            wrist_np = (
                wrist_img.detach().cpu().numpy()
                if torch.is_tensor(wrist_img)
                else np.asarray(wrist_img)
            )
            wrist = _center_crop_resize(wrist_np, width=w1, height=h1)
            if self.infer_cfg.concat_multi_camera == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                rgb = np.concatenate([primary, wrist], axis=1)
        x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=self.device, dtype=self.torch_dtype)
        x = x * (2.0 / 255.0) - 1.0
        return x

    def _normalize_proprio(self, state_vec) -> torch.Tensor:
        state = state_vec.detach().cpu() if torch.is_tensor(state_vec) else torch.as_tensor(state_vec)
        state = state.to(dtype=torch.float32).reshape(-1)
        batch = {"state": {self._state_key: state.unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self._state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        action = action.to(dtype=torch.float32, device="cpu")
        return normalizer.backward(action).numpy()

    # ------------------------------------------------------------------ RL helpers
    @property
    def is_rl(self) -> bool:
        """RL-capable when RL is enabled (flow-SDE rollout), with or without a critic."""
        return self._rl_enabled

    @property
    def has_value_head(self) -> bool:
        return self.value_head is not None

    def _value(self, pooled: torch.Tensor) -> torch.Tensor:
        """Value-head call, casting input to the head's param dtype; returns fp32 ``[B,1]``."""
        head_dtype = next(self.value_head.parameters()).dtype
        return self.value_head(pooled.to(head_dtype)).float()

    def _build_rl_conditioning(self, main_img, wrist_img, state, task):
        """Per-env conditioning for the flow-SDE: (image, context, context_mask)."""
        image = self._build_input_image(main_img, wrist_img)
        proprio = self._normalize_proprio(state).to(
            device=self.device, dtype=self.torch_dtype
        )
        context, context_mask = self.model.encode_prompt(
            DEFAULT_PROMPT.format(task=str(task))
        )
        context, context_mask = self.model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio
        )
        return image, context, context_mask

    def _build_rl_conditioning_batch(self, main_images, wrist_images, states, tasks):
        """Stack per-env conditioning into a single batched (image, context, mask).

        Raises if the text-context lengths differ across the batch (then the caller
        falls back to the per-env loop). In practice FastWAM's ``encode_prompt`` pads to
        a fixed T5 length, so this holds — the existing rollout path already relies on it
        (it ``torch.cat``s the per-env contexts into ``forward_inputs``).
        """
        imgs, ctxs, masks = [], [], []
        for i in range(len(tasks)):
            image, context, context_mask = self._build_rl_conditioning(
                main_images[i], None if wrist_images is None else wrist_images[i],
                states[i], tasks[i],
            )
            imgs.append(image)
            ctxs.append(context)
            masks.append(context_mask)
        lens = {int(c.shape[1]) for c in ctxs}
        if len(lens) != 1:
            raise ValueError(
                f"variable text-context lengths {sorted(lens)} cannot be batched"
            )
        return (
            torch.cat(imgs, dim=0),
            torch.cat(ctxs, dim=0),
            torch.cat(masks, dim=0),
        )

    def _postprocess_action(self, action_lat: torch.Tensor) -> np.ndarray:
        """Denormalize a [1,T,D] action latent and apply LIBERO gripper convention."""
        action = self._denormalize_action(action_lat)[0]  # [T, D]
        action[..., -1] = action[..., -1] * 2 - 1
        action = _invert_gripper_action(action)
        if self.infer_cfg.binarize_gripper:
            action[..., -1] = np.sign(action[..., -1])
        return action[: self.infer_cfg.num_action_chunks]

    @torch.no_grad()
    def _rl_predict_action_batch(self, env_obs: dict, mode: str = "eval", **kwargs):
        """RL rollout: flow-SDE sampling with per-step log-probs + value, recording
        the denoise chain + observation needed to replay it during training."""
        main_images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images")
        states = env_obs["states"]
        tasks = env_obs["task_descriptions"]
        batch_size = int(main_images.shape[0])
        deterministic = mode != "train"

        # Batched rollout in sub-batches of `_rl_rollout_chunk` so peak memory (the frozen
        # 5B video forward) stays bounded no matter how many envs a worker holds -- this is
        # what lets us scale total_num_envs for GRPO stability. Fall back to a per-env loop
        # on ANY error (e.g. an upstream op that still rejects B>1).
        if self._rl_batched and batch_size > 1:
            try:
                chunk = max(1, int(self._rl_rollout_chunk))
                conds = []
                for s in range(0, batch_size, chunk):
                    e = min(s + chunk, batch_size)
                    conds.append(self._build_rl_conditioning_batch(
                        main_images[s:e],
                        None if wrist_images is None else wrist_images[s:e],
                        states[s:e], tasks[s:e],
                    ))
                return self._rl_rollout_from_conds(conds, deterministic)
            except Exception as e:  # noqa: BLE001 — degrade to the safe per-env path
                logger.warning(
                    "FastWAM RL: batched rollout failed (%s); falling back to per-env loop.",
                    e,
                )

        conds = [
            self._build_rl_conditioning(
                main_images[i], None if wrist_images is None else wrist_images[i],
                states[i], tasks[i],
            )
            for i in range(batch_size)
        ]
        return self._rl_rollout_from_conds(conds, deterministic)

    def _rl_rollout_from_conds(self, conds, deterministic):
        """Run flow-SDE rollout over a list of (image, context, mask) groups and assemble
        the RLinf rollout result. Each group may carry any batch size (1 for per-env).

        Eval (``deterministic=True``) skips every training-replay tensor. The rollout
        worker discards the result dict on the eval path (``mode="eval" -> actions, _``),
        so collecting per-step log-probs/values and cat-ing chains/image/context across
        all envs is pure waste -- and at high eval env counts that cat is what OOMs.
        On the eval path we return ``actions`` plus an empty result.
        """
        cfg = self.infer_cfg
        nchunks = cfg.num_action_chunks
        chunks, logps, values = [], [], []
        f_chains, f_di, f_img, f_ctx, f_mask = [], [], [], [], []
        f_nn = []  # per-step injected-noise L2 norm (debug report)
        for image, context, context_mask in conds:
            action_lat, info = R.flow_sde_rollout(
                self.model, image, context, context_mask,
                action_horizon=cfg.action_horizon,
                num_inference_steps=cfg.num_inference_steps,
                noise_level=self._rl_noise_level,
                noise_method=self._rl_noise_method,
                deterministic=deterministic,
                sde_sampling=self._rl_sde_sampling,
            )
            b = int(image.shape[0])
            if not deterministic:
                # Training only: gather the PPO/GRPO replay tensors.
                di = info["denoise_inds"].to(torch.long)  # [b]
                ar = torch.arange(b, device=di.device)
                # Each trajectory's log-prob at its own sampled denoise step.
                logps.append(info["logp_per_step"][ar, di][:, :nchunks, :].float())
                if self.has_value_head:
                    values.append(self._value(info["pooled_video_feat"]))
                f_chains.append(info["chains"])
                f_di.append(di)
                f_nn.append(info["inject_noise_norm"].float())  # [b]
                f_img.append(image)
                f_ctx.append(context)
                f_mask.append(context_mask)
            for j in range(b):
                chunks.append(self._postprocess_action(action_lat[j : j + 1]))

        actions = np.stack(chunks, axis=0).astype(np.float32)
        if deterministic:
            # Eval: the rollout worker discards the result -- skip the replay tensors.
            return actions, {}
        result = {
            "prev_logprobs": torch.cat(logps, dim=0),
            # None for critic-free GRPO; the rollout worker tolerates a None value.
            "prev_values": torch.cat(values, dim=0) if self.has_value_head else None,
            "forward_inputs": {
                "chains": torch.cat(f_chains, dim=0),
                "denoise_inds": torch.cat(f_di, dim=0),
                "inject_noise_norm": torch.cat(f_nn, dim=0),  # [B] (debug report)
                "image": torch.cat(f_img, dim=0),
                "context": torch.cat(f_ctx, dim=0),
                "context_mask": torch.cat(f_mask, dim=0),
            },
        }
        return actions, result

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval", **kwargs):
        """Predict an action chunk for a batch of LIBERO observations.

        Args:
            env_obs: dict from :class:`~rlinf.envs.libero.libero_env.LiberoEnv`
                with ``main_images`` / ``wrist_images`` ``[B,H,W,3]`` uint8,
                ``states`` ``[B,8]`` and ``task_descriptions`` list[str].

        Returns:
            (actions, result): ``actions`` is ``np.ndarray`` of shape
            ``[B, num_action_chunks, action_dim]`` ready for ``env.chunk_step``;
            ``result`` is an (empty) metadata dict for the eval path.
        """
        if self.is_rl:
            return self._rl_predict_action_batch(env_obs, mode=mode, **kwargs)

        cfg = self.infer_cfg
        main_images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images")
        states = env_obs["states"]
        task_descriptions = env_obs["task_descriptions"]
        batch_size = int(main_images.shape[0])

        chunks = []
        for i in range(batch_size):
            image = self._build_input_image(
                main_images[i], None if wrist_images is None else wrist_images[i]
            )
            proprio = self._normalize_proprio(states[i])
            prompt = DEFAULT_PROMPT.format(task=str(task_descriptions[i]))

            common = dict(
                prompt=prompt,
                input_image=image,
                action_horizon=cfg.action_horizon,
                proprio=proprio,
                negative_prompt=cfg.negative_prompt,
                text_cfg_scale=cfg.text_cfg_scale,
                num_inference_steps=cfg.num_inference_steps,
                sigma_shift=cfg.sigma_shift,
                seed=cfg.seed,
                rand_device=cfg.rand_device,
                tiled=cfg.tiled,
            )
            # World-model imagination: also decode the predicted future video for
            # env 0 (capped) when enabled. infer_joint denoises video latents + VAE
            # decode (slower); the fast action-only path is used otherwise.
            do_viz = (
                cfg.visualize_future_video
                and cfg.future_video_dir is not None
                and i == 0
                and self._video_saves < cfg.max_video_saves
            )
            if do_viz:
                pred = self.model.infer_joint(
                    num_video_frames=cfg.num_video_frames,
                    test_action_with_infer_action=False,
                    **common,
                )
                self._save_future_video(pred["video"])
            else:
                pred = self.model.infer_action(**common)
            action = self._denormalize_action(pred["action"])[0]  # [T, D]

            # FastWAM dataloader maps gripper to (0=close, 1=open); undo that and
            # restore LIBERO's (-1=open, +1=close) convention before execution.
            action[..., -1] = action[..., -1] * 2 - 1
            action = _invert_gripper_action(action)
            if cfg.binarize_gripper:
                action[..., -1] = np.sign(action[..., -1])

            chunks.append(action[: cfg.num_action_chunks])

        actions = np.stack(chunks, axis=0).astype(np.float32)
        return actions, {}

    # ------------------------------------------------------------------ SFT
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"FastWAMPolicy does not support {forward_type}.")

    def sft_forward(self, data: Any = None, **kwargs):
        """Supervised flow-matching loss over a FastWAM sample batch.

        ``data`` is a dict produced by FastWAM's ``RobotVideoDataset`` collate
        (``video``, ``action``, ``context``, ``context_mask`` and optional
        ``proprio`` / ``*_is_pad``). Returns a dict with at least ``loss``.
        """
        if data is None:
            raise ValueError("FastWAMPolicy.sft_forward requires `data`.")
        loss, metrics = self.model.training_loss(data)
        out = {"loss": loss}
        # Surface video / action sub-losses for logging (worker reads .item()).
        if isinstance(metrics, dict):
            if "loss_video" in metrics:
                out["dynamics_loss"] = torch.as_tensor(metrics["loss_video"])
            if "loss_action" in metrics:
                out["action_loss"] = torch.as_tensor(metrics["loss_action"])
        return out

    def default_forward(
        self,
        forward_inputs: Any = None,
        compute_logprobs: bool = True,
        compute_values: bool = True,
        compute_entropy: bool = False,
        **kwargs,
    ):
        """RL training forward: replay the stored denoise chain and recompute the
        sampled step's log-prob (+ entropy) and the state value under current params.

        Returns ``{"logprobs":[B,nchunks,D], "values":[B,1], "entropy":[B,nchunks,D]}``
        matching RLinf's PPO actor contract (see EmbodiedFSDPActor.train_micro_batch).
        """
        if not self.is_rl:
            raise NotImplementedError(
                "FastWAMPolicy.default_forward requires RL to be enabled; SFT uses sft_forward."
            )
        if forward_inputs is None:
            raise ValueError("FastWAMPolicy.default_forward requires `forward_inputs`.")
        cfg = self.infer_cfg
        nchunks = cfg.num_action_chunks
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        image = forward_inputs["image"]
        context = forward_inputs["context"]
        context_mask = forward_inputs["context_mask"]
        batch_size = int(chains.shape[0])

        want_values = compute_values and self.has_value_head

        def _run(sl: slice):
            logp_b, ent_b, pooled_b = R.recompute_logprob(
                self.model,
                input_image=image[sl],
                context=context[sl],
                context_mask=context_mask[sl],
                chains=chains[sl],
                denoise_inds=denoise_inds[sl],
                action_horizon=cfg.action_horizon,
                num_inference_steps=cfg.num_inference_steps,
                noise_level=self._rl_noise_level,
                noise_method=self._rl_noise_method,
                train_video_expert=self._rl_train_video_expert,
            )
            val_b = self._value(pooled_b) if want_values else None
            return (
                logp_b[:, :nchunks, :].float(),
                ent_b[:, :nchunks, :].float(),
                val_b,
            )

        use_batched = self._rl_batched and batch_size > 1
        logps = ents = values = None
        if use_batched:
            try:
                lp, ent, val = _run(slice(0, batch_size))
                logps, ents, values = [lp], [ent], [val]
            except Exception as e:  # noqa: BLE001 — degrade to the safe per-env path
                logger.warning(
                    "FastWAM RL: batched default_forward failed (%s); "
                    "falling back to per-env loop.", e,
                )
                use_batched = False
        if not use_batched:
            logps, ents, values = [], [], []
            for b in range(batch_size):
                lp, ent, val = _run(slice(b, b + 1))
                logps.append(lp)
                ents.append(ent)
                values.append(val)

        out = {"logprobs": torch.cat(logps, dim=0)}
        # The PPO/GRPO denominator is the behavior log-prob returned by rollout.
        # Do not synthesize prev_logprobs here; doing so would silently make
        # ratio == 1 and remove the behavior-policy importance correction.
        # values omitted for critic-free GRPO (values entries are None).
        if want_values:
            out["values"] = torch.cat(values, dim=0)
        if compute_entropy:
            out["entropy"] = torch.cat(ents, dim=0)
        return out

    def gradient_checkpointing_enable(self, **kwargs):
        # FastWAM toggles gradient checkpointing through its DiT configs
        # (``use_gradient_checkpointing`` / ``mot_checkpoint_mixed_attn``); this is a
        # no-op so RLinf's FSDP manager can call it uniformly.
        return
