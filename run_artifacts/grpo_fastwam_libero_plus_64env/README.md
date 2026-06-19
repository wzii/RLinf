# FastWAM critic-free GRPO on LIBERO-Plus — run artifacts

Run that validated the FastWAM GRPO code (branch `feat/fastwam-RL`) end-to-end with
the official released checkpoint (`yuanty/fastwam libero_uncond_2cam224.pt`).

## Files
- `run_stdout.log` — full training stdout (Ray workers, per-step metrics).
- `metrics.log` — RLinf metric log.
- `tensorboard/events.out.tfevents.*` — TensorBoard scalars (`tensorboard --logdir tensorboard`).
- `run_config.yaml` — the exact config used.

## Setup
- Model: FastWAM (Wan2.2 5B video expert FROZEN + 1B action expert trained), bf16.
- Algorithm: critic-free GRPO (`adv_type=grpo`, `loss_type=actor`), principled fp32 flow-SDE actor.
- Env: LIBERO-Plus (`libero_variant=plus`, `perturbation_suffix=all`), 64 envs = 8 groups x 8.
- Batched rollout (`rollout_chunk=8`) + batched training (`micro_batch_size=8`), 4x A100-80GB.

## Result (50 GRPO steps)
- Healthy: no divergence / OOM / fallbacks; peak GPU 71 GB / 80 GB; ~142 s/step.
- `success_once`: range 0.11–0.69, overall mean 0.357, early-half 0.346 -> late-half 0.369
  (measurable upward drift, noisy due to the aggressive step size approx_kl~0.5).
- The end-of-run 88 GB checkpoint save filled the disk and was discarded (training itself
  completed all 50 steps).

## Note
Rollout sampling uses the global RNG (unseeded), so per-step `success_once` varies run-to-run.
