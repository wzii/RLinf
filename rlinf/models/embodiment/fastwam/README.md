# FastWAM in RLinf

[FastWAM](https://github.com/yuantianyuan01/FastWAM) (*Fast-WAM: Do World Action
Models Need Test-time Future Imagination?*) is a Wan2.2 video-diffusion world model
with a flow-matching action expert. This package adapts FastWAM's own implementation
to RLinf so it can be **evaluated** (LIBERO / LIBERO-Plus) and **SFT-trained** through
RLinf's standard workers, by wrapping the upstream package rather than reimplementing it.

## Layout

| File | Purpose |
|------|---------|
| `fastwam_policy.py` | `FastWAMPolicy(BasePolicy)` — `predict_action_batch`→`infer_action`, `sft_forward`→`training_loss` |
| `__init__.py` | `get_model` — composes FastWAM's Hydra configs, builds model + processor, loads checkpoint |
| `../../../data/datasets/fastwam/` | SFT dataloader wrapping FastWAM `RobotVideoDataset` |

Registered in `rlinf/config.py` (`SupportedModel.FASTWAM`, `EMBODIED_MODEL`),
`rlinf/models/__init__.py` (`_build_fastwam`), and dispatched in
`rlinf/workers/sft/fsdp_vla_sft_worker.py`.

## Prerequisites

- Install FastWAM as a package (`pip install -e /path/to/FastWAM`) so `import fastwam` works.
- Wan2.2 VAE + T5 are fetched by DiffSynth on first model build. The default converted
  -safetensors mirror is **ModelScope-only**; configs set `model.redirect_common_files=false`
  to use the original `Wan-AI/Wan2.2-TI2V-5B` `.pth` files on HuggingFace.
- Eval uses `skip_dit_load_from_pretrain=true`: the released FastWAM checkpoint's `mot`
  provides the trained video+action experts, so the 20GB Wan DiT is **not** downloaded.

## Evaluation (LIBERO + LIBERO-Plus)

```bash
# weights
huggingface-cli download yuanty/fastwam libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json --local-dir /workspace/checkpoints/fastwam

# LIBERO
MUJOCO_GL=egl bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

# LIBERO-Plus  (needs the liberoplus package + its assets.zip, and ImageMagick)
LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
  bash evaluations/run_eval.sh libero libero_spatial_fastwam_plus_eval
```

Set `rollout.model.checkpoint_path` / `dataset_stats_path` (configs default to
`/workspace/checkpoints/fastwam/...`). `num_action_chunks` is the executed replan
length; FastWAM predicts `action_horizon` (=`num_frames-1`) per inference.

## SFT

```bash
# 1) LeRobot LIBERO data + 2) precomputed T5 text embeddings
python FastWAM/scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 \
  'data.train.dataset_dirs=[/path/to/libero_spatial_no_noops_lerobot]' \
  data.train.text_embedding_cache_dir=/workspace/data/text_embeds_cache/libero \
  model.redirect_common_files=false

bash examples/sft/run_vla_sft.sh libero_sft_fastwam
```

SFT initialises the MoT from the released checkpoint (`skip_dit_load_from_pretrain`),
so only VAE+T5 + the 12GB checkpoint are needed (no 20GB DiT / ActionDiT backbone).
Only the MoT experts (+ proprio encoder) are trained (`freeze_non_dit`).

**Multi-GPU note:** the full MoT is ~6B trainable params, and the MoT performs *manual*
cross-expert attention (accessing DiT-block internals rather than `block.forward`),
which is incompatible with per-block FSDP2 auto-wrap. Full-MoT SFT therefore needs
multi-GPU FSDP with a whole-model wrap policy; on a single GPU it OOMs. See
`smoke_sft.py` for a single-GPU action-expert smoke that exercises the full
dataloader → `sft_forward` → `training_loss` → backward path.

## RL (PPO, flow-SDE)

PPO trains the action expert (and optionally the video expert) by turning FastWAM's
deterministic flow-matching ODE into a *flow-SDE* — each denoise step becomes a
tractable Gaussian whose realized step has a closed-form log-prob. The core lives in
`fastwam_rl.py`; `FastWAMPolicy` wires it to RLinf's PPO actor (`predict_action_batch`
→ rollout with `prev_logprobs`/`prev_values`/`forward_inputs`; `default_forward` →
replay + recompute under current params). Launch:

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_fastwam
```

Options (`model/fastwam_rl.yaml` → `rl:`):

| key | default | meaning |
|-----|---------|---------|
| `noise_method` | `flow_sde` | `flow_sde` = principled reverse-SDE (time-dependent `sigma=noise_level·√(t/(1-t))` + score/drift correction), **matching RLinf's OpenPI/GR00T heads**; `simple` = legacy ODE-mean + constant-scale noise. |
| `noise_level` | `0.1` | exploration scale. |
| `train_video_expert` | `false` | also update the 5B video expert (value-head gradient flows through the now grad-enabled video conditioning). Much heavier — multi-GPU. `freeze_video_expert: false` is an alias. |
| `batched` | `true` | run the whole env batch in one model forward; auto-falls back to a per-env loop (with a warning) on any error. |

### Differences vs OpenPI / GR00T (and known caveats)

* **Single denoise step in the PPO ratio.** OpenPI/GR00T store `denoise_inds` as
  `[B, num_steps]` and can do `joint_logprob` over multiple steps; FastWAM samples **one**
  step per trajectory (`denoise_inds` is `[B]`). Higher variance but unbiased; this is the
  upstream FastWAM design, preserved here.
* **Flow-time convention is assumed, not controlled.** OpenPI owns its flow schedule
  (clean linear `t∈[0,1]`, `x_t=(1-t)x0+t·x1`). Here we assume FastWAM's scheduler
  `timesteps` are exactly that interpolation coefficient (true for the Wan2.2 flow
  scheduler incl. `sigma_shift`, which only reparametrises within `[0,1]`). If a future
  scheduler fed the DiT a timestep that is *not* the interpolation coefficient, the
  `data_pred`/`noise_pred` reconstruction — and thus the SDE drift correction — would be
  biased. `noise_method: simple` does not depend on this assumption. *(verified: the
  `flow_sde` mean/std match an independent OpenPI-formula reference to machine precision.)*
* **Value head reads the frozen (or now-trainable) pooled pre-DiT video tokens**, not the
  action-expert/VLM suffix like OpenPI's `value_after_vlm`/`value_vlm_mode`. Shallower
  critic; when `train_video_expert` is on, the value gradient *does* reach the video
  expert through `pooled`.
* **bf16 throughout.** OpenPI casts `suffix_out`/value to fp32; FastWAM keeps chains and
  the value head in bf16 (the policy casts to the model dtype). Logprob/value precision is
  therefore lower — a candidate follow-up is fp32 chains + fp32 value head.
* **Batched path is unverified end-to-end here** (no `fdtwam`/GPU in this dev box). It
  assumes the upstream model methods (`pre_dit`, `prefill_video_cache`,
  `forward_action_with_video_cache`, `post_dit`) accept a leading batch dim and that the
  per-sample timestep `[B]` is honoured — the same assumptions OpenPI's batched path makes.
  Any failure degrades gracefully to the proven per-env loop with a logged warning. The
  shared MoT attention mask is built once from `(video_seq_len, action_horizon)`, valid
  because every sample shares image resolution and horizon; per-sample text padding is
  carried by `context_mask` (and the existing rollout already required uniform context
  length to `torch.cat` `forward_inputs`).
* **Recompute rebuilds the video KV cache every micro-step.** When `train_video_expert`
  is off this is `no_grad` (cheap-ish); when on it carries grad and is expensive. OpenPI
  caches the prefix KV; FastWAM's cache is not serialised across rollout→train, so it is
  recomputed.
