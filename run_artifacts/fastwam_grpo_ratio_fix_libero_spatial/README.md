# FastWAM SFT + GRPO RL in RLinf on 4×A100-SXM4 (NVLink) — run report

Experiments run 2026-06-22 on a **4×A100-SXM4-80GB, NVLink active (NV12, ~300 GB/s)** Vast.ai
instance — the fast counterpart to the earlier **4×A100-PCIe** box (NVLink inactive, ~5 GB/s)
whose run-logs live in `run-logs/fastwam-sft-verify-libero-spatial` and
`run-logs/fastwam-grpo-libero-plus`.

Code: branch `feat/fastwam-RL` (unchanged); only configs + launch/util scripts added.

---

## 1. Environment

- venv `/workspace/venv_fw` (py3.11, torch 2.7.1+cu128). `pip install -e` FastWAM + RLinf;
  RLinf/LIBERO fork (`--no-deps`) + robosuite 1.4.1 + sim deps; LIBERO-Plus (`RLinf/LIBERO-plus`).
- **Wan2.2-TI2V-5B backbone from HuggingFace** (`hf download Wan-AI/Wan2.2-TI2V-5B`, ~32 GB:
  VAE + T5 + umt5 tokenizer + DiT), symlinked into the DiffSynth path
  (`DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints`, `redirect_common_files=false`). **No ModelScope.**
- FastWAM model builds end-to-end (6.7 B; VAE/T5 from the HF `.pth`). LIBERO renders via EGL.

---

## 2. SFT timing — **~24× faster than the PCIe box**

Same config as the PCIe run-log (`libero_sft_fastwam`, mbs=4, gbs=16, freeze_non_dit, full MoT trained):

| | 4×A100 **PCIe** (run-log) | 4×A100 **SXM/NVLink** (here) |
|---|---|---|
| **per-step (steady)** | ~36 s/step | **median 1.49 s/step** (min 1.35 s) |
| peak GPU mem | ~62–74 GB | ~62 GB |
| loss / grad_norm | ~0.08 / ~0.13 | identical |

**~24× faster.** This confirms the PCIe diagnosis exactly: FastWAM's forced whole-model FSDP wrap
all-gathers the 13 GB model every fwd/bwd; over PCIe (~5 GB/s) that dominates, over NVLink it's
negligible and the run becomes compute-bound.

(After SFT, the LeRobot dataset was deleted to free disk for RL.)

---

## 3. RL configuration — the memory & disk journey

FastWAM's **whole-model FSDP wrap** (its MoT does manual cross-expert attention, incompatible with
per-block wrap) means the full 6.7 B model is all-gathered (~13 GB contiguous buffer) every
fwd/bwd. On 4×80 GB with actor+rollout+env **co-located**, this made memory tight. Findings:

- **`total_num_envs` is the training-memory driver** (the full collected rollout buffer of
  video-latent forward-inputs stays GPU-resident during training). 128 envs OOMs at ~81 GB;
  96 envs is unreliable (per-step peak *varies* with rollout content, 75–81 GB, OOMs intermittently).
- **Valid env counts are multiples of 32** (group_size 8 × 4 GPUs), so the choices are 32/64/96/128.
  **64 envs (mbs=8, gbs=64, reshard_after_forward=True) is the reliable max ≈ 75 GB peak.**
- **`gbs` = full-batch GRPO update** (= total_num_envs; FastWAM produces one training sample per
  trajectory — a single flow-SDE denoise step — so `chunk_level`, not token-level, is the correct
  unit; token-level would be a no-op for batch size and mismatch the joint diffusion sampling).
- **Eval env count is a *hidden* OOM driver:** with `val_check>0` the workers allocate the eval envs
  at init (sized by `eval.total_num_envs`), held through training. eval=100 made even 64-env training
  OOM. Fix: **eval = 20 envs × 5 epochs = 100 episodes** (≤ train → sizes by 64).
- **Checkpoints are 46–86 GB each** (the 12.4 B RL policy's consolidated fp32 `full_weights.pt` +
  optimizer DCP). They do **not** fit a 159 GB disk alongside 73 GB of models — `save_interval=10`
  filled the disk and crashed `torch.save`. Since the eval signal is in-process (no disk) and the
  trained policy degrades (below), **checkpoint saving was turned off**; the **eval trend is the result**.

**Final RL config (both suites):** GRPO (critic-free, flow-SDE actor, group_size=8), 64 train envs,
mbs=8, gbs=64, reshard_after_forward=True, lr=3e-5, clip 0.2; eval 100 ep (20×5) every 10 steps; 50 steps.

---

## 4. Results

### 4a. libero_spatial (standard) — **GRPO degrades the policy**

Eval `success_once` on 100 episodes (10 trials × 10 tasks), vs released-ckpt baseline SR₀ ≈ **0.904**:

| step | 10 | 20 | 30 | 40 |
|---|---|---|---|---|
| **success_once** | **0.97** | 0.81 | 0.81 | **0.74** |
| success_at_end | 0.95 | 0.74 | 0.79 | 0.70 |

The SR spikes above baseline at step 10 then **steadily collapses to 0.74** (−0.16 below baseline by
step 40). `grad_norm` is 183 at step 1 (clipped to 100) then 4–77; `approx_kl` ranges 0.02–0.27 (mean
0.117). **Conclusion: FastWAM GRPO at the reference hyperparameters does not improve libero_spatial —
it destabilizes the already-converged policy.** This is consistent with the prior observation that
the reference run's training SR was not clearly increasing. (The step-50 eval + final ckpt were lost
to the disk-full crash, but the 4-point trend is unambiguous.)

### 4b. libero_plus (spatial) — see below

LIBERO-Plus `all`/`light`/`add`/`tb` perturbations crash the RL run with `KeyError` on
dynamically-generated object/texture/scene variants (a LIBERO-Plus package bug in its on-the-fly
variant registration — *not* disk, *not* our code; single-process env creation works but the
broad multi-worker RL sampling hits the broken subset). Plus was run with **`perturbation_suffix=language`**
(instruction perturbations, no new assets). [RESULT — filled in after the run]

### 4c. Baselines (released ckpt) — [filled in]

---

## 5. Honest caveats

- **The reference GRPO run-log was *not* validated** (no proper eval; training SR not clearly rising) —
  so this run was the actual test, and the answer (for libero_spatial) is **GRPO does not help here**.
- 50 steps is short and the reward is sparse binary task success; a different lr / noise_level / longer
  horizon might behave differently, but the *degradation* trend is a clear negative signal at these settings.
- Checkpoints not saved (disk-infeasible at 46–86 GB); the deliverable is the eval trends + logs.

---

## 6. THE KEY FINDING: the GRPO "degradation" was an importance-ratio bug (now fixed)

The libero_spatial "GRPO degrades" result (4a) is **an artifact of a bug**, not a property of GRPO.

**Symptom:** `actor/ratio` swung 0.25–1.03 (systematically <1) and `approx_kl` up to 0.27 **before any
gradient step** — impossible for correct PPO/GRPO with `update_epoch=1` (the ratio must be ≈1 at the
first update because old==new policy).

**Diagnosis (quantitative):**
- Single-process rollout→recompute is **bit-exact** (ratio = 1.0000, logp diff = 0). So bf16 precision
  and the video-cache rebuild cause **zero** in-process error — the code is correct.
- FastWAM uses the **rollout worker's** bf16 log-prob as the "old" log-prob. The rollout worker (plain
  bf16, per-env batches) and the actor worker (FSDP bf16, batched) are **separate processes** whose bf16
  forwards disagree at ~bf16 precision (~8e-3).
- FastWAM's flow-SDE chunk-level log-prob is a **70-element sum** (10 chunks × 7 dims) evaluated at the
  **sampled** action (near the rollout mean = the Gaussian tail). A controlled test: an 8e-3 (bf16-scale)
  shift → chunk ratio mean **0.62**, range 0.08–3.18 — matching the observed run. Systematically <1 because
  any cross-process perturbation moves the sampled action into the tail → lower recomputed log-prob.
- **Why SFT is unaffected:** SFT minimizes ‖v_pred−v_true‖² (8e-3 error → ~6e-5 loss change, negligible);
  RL's exp-of-a-sum ratio amplifies the same error. It's RL-vs-SFT, not model-specific.
- **Why GR00T/ABOT are unaffected:** `fsdp_actor_worker.py` recomputes the "old" log-prob on the learner
  for those models but **not** for FastWAM.

**Fix (2 edits, on `feat/fastwam-RL`):**
1. `FastWAMPolicy.default_forward` returns `prev_logprobs = logprobs.detach()` — the actor's own in-process
   recompute as the "old" log-prob.
2. `EmbodiedFSDPActor.train_micro_batch` reads `output_dict["prev_logprobs"]` for `SupportedModel.FASTWAM`.

So "old" and "new" come from the same in-process forward → `ratio≡1` at the first update (correct for
`update_epoch=1`; for >1 recompute once before the inner loop). **Verified:** `ratio_abs` (mean|ratio−1|)
0.33→**0.0000**, `approx_kl` 0.205→**0.0000**, `clip_fraction` 0.116→**0.0000**, across all steps of the
re-run.

**Result with the fix (libero_spatial, in progress):** the collapse is gone — eval `success_once` is
**stable** (step10 0.84, step20 0.84) instead of the buggy slide 0.97→0.81→0.81→0.74, with `success_at_end`
rising 0.65→0.77. It is **stable but ~6pt below the 0.904 baseline** so far — i.e. the fix removes the
spurious gradient (no more collapse), but GRPO at these hyperparameters (`lr=3e-5`, loose `clip_grad=100`)
does not yet beat an already-near-optimal policy; a gentler-hyperparameter run is the natural follow-up.
See `EVAL_SUMMARY.txt` and the run logs.
