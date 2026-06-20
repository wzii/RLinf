# FastWAM SFT correctness verification on LIBERO-spatial — run artifacts

Experiment to verify that **RLinf's FastWAM SFT pipeline is correct**, by fine-tuning the
released FastWAM checkpoint on its *own* LIBERO training data and checking that it does
**not** move: a correct SFT pipeline applied to an already-converged checkpoint should show
near-flat loss, small non-drifting gradients, negligible weight change, and unchanged eval
success rate.

Branch under test: `feat/fastwam-RL` (code identical to the known-good
`run-logs/fastwam-grpo-libero-plus`). Hardware: 4× A100 80GB **PCIe** (no NVLink).

## Files
- `README.md` — this writeup.
- `sft_metrics.tsv` — per-step SFT metrics (step, action_loss, dynamics_loss, grad_norm, lr, loss) for all 1000 steps.
- `tensorboard/events.out.tfevents.*` — SFT TensorBoard scalars.
- `run_config.yaml` — the exact SFT config used.
- `eval_baseline_original_ckpt.txt` — RLinf LIBERO-spatial eval of the **original** released ckpt (500 episodes).
- `eval_sft1k_ckpt.txt` — RLinf LIBERO-spatial eval of the **SFT-1k** ckpt (500 episodes, identical fixed init states).
- `SR_comparison.txt` — side-by-side SR summary.

## Setup
- Model: FastWAM (Wan2.2-TI2V-5B video expert + 1B action expert via MoT), bf16.
  Init weights = released `yuanty/fastwam libero_uncond_2cam224.pt`. VAE/T5/tokenizer from
  the local `Wan-AI/Wan2.2-TI2V-5B` files (no extra downloads; `redirect_common_files=false`).
- Data: LeRobot `yuanty/LIBERO-fastwam` **libero_spatial** only (434 episodes), T5 text
  embeddings precomputed via FastWAM `scripts/precompute_text_embeds.py`.
- SFT: RLinf FSDP2 pipeline (`examples/sft/.../libero_sft_fastwam.yaml`), `freeze_non_dit=true`
  → only the MoT experts (~6.0B trainable) + proprio encoder are trained.
  **1000 steps**, lr 1e-5 cosine (5% warmup), `micro_batch_size=4`, `global_batch_size=16`
  (the throughput sweet spot; mbs 1–4 all ~36 s/step, peak ~74 GB/80 GB). ~10 h wall-clock.
- Checkpoint recovered from the FSDP DCP shards (the consolidated save filled the disk) into a
  native FastWAM `.pt` (`{mot, proprio_encoder}`); key-set verified identical to the released ckpt.

## Result — SFT pipeline is CORRECT

All four expected signals confirmed:

| signal | expectation | measured |
|---|---|---|
| training loss | ~flat (no learning) | mean **0.0814** (std 0.021); first-100 mean **0.0804** → last-100 **0.0810**; linear slope **−9.8e-7/step** (≈ −0.001 over 1000 steps) |
| grad_norm | ~0 / no drift | mean **0.135** (std 0.053); first-100 **0.1358** → last-100 **0.1368** (no drift) |
| weight change | negligible | ‖SFT − released‖ / ‖released‖ over the MoT = **0.0012 (0.12%)** after 1000 steps |
| eval success rate | unchanged | see below |

Eval (RLinf LIBERO-spatial, **500 episodes** = 50 trials × 10 tasks, identical fixed init states):

| ckpt | success_once | success_at_end |
|---|---|---|
| original (released) | **0.904** | 0.878 |
| SFT 1k | **0.888** | 0.852 |
| Δ | −0.016 (≈ −1.2σ) | −0.026 (≈ −1.8σ) |

The deltas are within binomial noise (σ ≈ 1.3–1.5 pt at n=500); rollout sampling is unseeded.

**Conclusion:** fine-tuning the converged checkpoint on its own data for 1000 steps leaves the
loss flat, gradients at a small stochastic floor (the flow-matching objective is inherently
noisy per random timestep/noise — its *expectation* is ~0 at the optimum), the weights almost
unchanged (0.12%), and the task success rate statistically unchanged. This is exactly the
behavior of a correct SFT data + loss pipeline. **RLinf's FastWAM SFT is verified correct.**

## Note on the RL follow-up (not in this artifact)
A GRPO RL run on LIBERO-spatial was attempted from the same checkpoint but is impractically
slow **on this specific instance** (~17 min/step) — diagnosed to the hardware, not the code:
this box is A100 **PCIe** with **NVLink inactive** (`nvidia-smi nvlink --status` = "all links
inActive"; measured inter-GPU P2P **7–11 GB/s** vs NVLink ~300–600 GB/s). FastWAM's forced
whole-model FSDP wrap all-gathers ~13 GB every forward/backward, which is negligible over the
reference's SXM/NVLink (~31–74 s/step training, see `run-logs/fastwam-grpo-libero-plus`) but
dominates over PCIe. Code + resolved config are byte-identical to that known-good run; only the
interconnect differs. RL should be run on an A100-SXM (NVLink) instance.
