---
name: fastwam-rlinf-sft-verify
description: "Experiment to verify RLinf's FastWAM SFT correctness on LIBERO spatial"
metadata: 
  node_type: memory
  type: project
  originSessionId: aec8b9f4-9c46-475a-a584-588b491eaa06
---

Goal (started 2026-06-19): experimentally verify RLinf's FastWAM SFT is correct. Train FastWAM ckpt with FastWAM-libero SFT data via RLinf SFT pipeline for 1k steps on **libero_spatial only**, save ckpt, eval. Expected (sanity): grad_norm ≈ 0, loss ≈ flat, 1k-ckpt SR ≈ original ckpt SR (because SFT on the same data from a converged ckpt should not move it).

Key facts:
- Env: 4× A100 80GB, fresh box, nothing installed (no torch). Workspace is a 160G volume.
- /workspace/RLinf (RLinf v0.3.0, has fastwam integration), /workspace/FastWAM (upstream, `pip install -e`), /workspace/fastwam_ckpt == /workspace/fastwam_libero (identical HF ckpt repos: libero_uncond_2cam224.pt 12G + _dataset_stats.json). NOTE: fastwam_libero is NOT data — both dirs are just ckpts. Need LeRobot dataset yuanty/LIBERO-fastwam (libero_spatial only) separately.
- /workspace/wan2.2 = Wan-AI/Wan2.2-TI2V-5B repo (Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, DiT safetensors, google/umt5-xxl tokenizer). Use these for T5/VAE — no extra download.
- torch: 2.7.1+cu128 (fastwam pin; works on A100 cc8.0, driver max CUDA 13.2). No flash_attn needed.
- Model files resolved via DIFFSYNTH_MODEL_BASE_PATH/<model_id>/<file>. Need symlinks: $BASE/Wan-AI/Wan2.2-TI2V-5B -> /workspace/wan2.2 ; $BASE/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl -> /workspace/wan2.2/google/umt5-xxl. configs use redirect_common_files=false (use .pth not safetensors).
- SFT config: examples/sft/config/libero_sft_fastwam.yaml (+ model/fastwam.yaml). checkpoint_path default /workspace/checkpoints/fastwam/libero_uncond_2cam224.pt. freeze_non_dit=true -> trains mot (=model.dit, ~6B params, video+action experts) + proprio_encoder. Needs precomputed T5 text embeds cache (FastWAM/scripts/precompute_text_embeds.py) + LeRobot data dir.
- SFT launch: bash examples/sft/run_vla_sft.sh libero_sft_fastwam. Loss/grad_norm logged to tensorboard. Worker: rlinf/workers/sft/fsdp_vla_sft_worker.py.
- Checkpoint format: native fastwam .pt = {"mot":..., "proprio_encoder":...}. RLinf saves with save_full_model_weights=true -> <save>/model_state_dict/full_weights.pt (full state dict of FastWAMPolicy, keys prefixed model.mot.* / model.proprio_encoder.*). Convert -> native .pt for eval.
- Eval: RLinf evaluations/run_eval.sh libero libero_spatial_fastwam_eval (config evaluations/libero/libero_spatial_fastwam_eval.yaml, default only 4 envs - increase for meaningful SR). Needs RLinf/LIBERO fork (github.com/RLinf/LIBERO) installed + MUJOCO_GL=egl. checkpoint_path / dataset_stats_path under rollout.model.

PROGRESS (as of 2026-06-20):
- Env built in /workspace/venv_fw (py3.11, torch 2.7.1+cu128). Launchers: /workspace/run_sft.sh, /workspace/run_eval.sh, /workspace/convert_ckpt.py.
- Batch tuned: mbs=4/gbs=16 → ~36s/step (compute saturates at mbs≈4; mbs1-4 all ~36s/step), peak ~62-74GB/80GB. 1k steps ≈ 10h.
- BASELINE SR_0 (original ckpt, 500 episodes = 50 trials×10 tasks): success_once=0.904, success_at_end=0.878. Saved /workspace/results/SR_baseline.txt.
- SFT 1k RUNNING (bg): mbs4/gbs16, lr 1e-5 cosine + warmup 5%, save_full_model_weights=True. Early steps: loss~0.065-0.09 (flat, stochastic diffusion noise), grad_norm~0.10 (steady, no drift) → consistent with model-at-optimum. Logs /workspace/results/sft_1k/run.log (tqdm uses \r; use tr '\r' '\n').
- 1k ckpt will save to /workspace/results/sft_1k/libero_sft_fastwam_1k/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt → convert via convert_ckpt.py to native .pt → eval with run_eval.sh (total_num_envs=20 rollout_epoch=25 = 500 ep) → SR_1k.
- Eval metric: success_once is primary SR. Eval ~35min/500ep.
- NOTE on expectation: grad_norm "~0" is really "small & stochastic, no systematic drift" (~0.1 for 6B bf16 diffusion model); loss "flat" = no downward trend (per-step variance is inherent to flow-matching random t/noise). Real verdict = SR preserved.

RESULTS (2026-06-20):
- SFT 1k DONE. Loss over all 1000 steps: mean 0.0814 std 0.0207; first-100 mean 0.0804 vs last-100 0.0810 (NO drift); loss slope -9.8e-7/step (=-0.001 total). grad_norm mean 0.135 std 0.053, first-100 0.1358 vs last-100 0.1368 (no drift). => loss几乎不变 + grad_norm小且无漂移. CONFIRMED.
- CKPT SAVE GOTCHA: full_weights.pt save ran OUT OF DISK (160G volume; wan2.2 32G + fastwam_ckpt 23G + fastwam_libero 23G dup + dcp 58G). full_weights truncated. RECOVERED via /workspace/dcp_to_fastwam.py: selectively loads model tensors (mot+proprio) from dcp_checkpoint shards (skips 48G optimizer) -> native /workspace/checkpoints/fastwam_sft1k/sft1k.pt (12G). KEY-SET MATCH exact. Then deleted dcp_checkpoint (58G) to free space. Disk is TIGHT - watch it.
- WEIGHT DELTA: ||sft1k - released||/||released|| (mot) = 0.0012 (0.12%). Model barely moved => strong proof SFT pipeline correct.
- SR_1k eval: running/done (see /workspace/results/eval_sft1k). Compare to SR_0=0.904.
- RL READY (not launched until SR confirms): config /workspace/RLinf/examples/embodiment/config/libero_spatial_grpo_fastwam.yaml (GRPO, save_interval=50, env.train rollout_epoch=1 x total_num_envs=128 = 128 traj/step, group_size=8, init from released ckpt). Launcher /workspace/run_rl.sh. May OOM at 128 envs (32/GPU) - tune down to 64envs x2epoch if so.

VERDICT: RLinf FastWAM SFT CORRECT. SR_0=0.904 vs SR_1k=0.888 (success_once; -1.2σ, within noise); loss flat; grad_norm ~0.135 no drift; weight delta 0.12%.

RL PHASE (libero_spatial GRPO):
- CRITICAL FIX: expandable_segments:True breaks actor->rollout CUDA IPC weight sync (uses VMM fd-based IPC -> pidfd_getfd which is BLOCKED in this container: "Operation not permitted" even on self). REMOVED expandable_segments from run_rl.sh -> legacy allocator uses cudaIpcMemHandle -> works. (Keep it for SFT single-process; NEVER for multi-worker RL weight sync here.)
- GRPO config /workspace/RLinf/examples/embodiment/config/libero_spatial_grpo_fastwam.yaml: critic-free (add_value_head=False), noise_method=flow_sde, save_interval=50, group_size=8, mbs=8 gbs=64. Launcher /workspace/run_rl.sh.
- 128 envs step1: success_once=0.70 (train), grad_norm=20.8, PEAK MEM 76/80GB (training phase; rollout only 26GB), STEP TIME ~42min (run_training 37min, GRPO recompute-dominated). mbs can't grow (training peak binding).
- User chose to PUSH to total_num_envs=256 (256 traj/update, ~80min/step). Relaunched. Watch rollout-phase OOM (64 envs/GPU). Fallback if OOM: 128 envs x rollout_epoch=2 (same 256 batch, rollout stays 26GB).
- RL run dir /workspace/results/rl_grpo. First ckpt at step 50 (~67h at 256 envs).

RL SLOWNESS ROOT CAUSE (2026-06-20, important):
- RL training was ~14-30x slower than the reference run (origin/run-logs/fastwam-grpo-libero-plus, run_artifacts/grpo_fastwam_libero_plus_64env: run_training ~31-74s/step, ~3min/step total at 64 env).
- My earlier "128-env per-env fallback" diagnosis was WRONG: my 64-env was ALSO slow (>17min training, no fallback warning). Slowness is SYSTEMIC, training-specific (rollout matched ref: 56s vs 61s).
- TRUE CAUSE = HARDWARE INTERCONNECT. This box = A100 80GB PCIe, 2 NUMA nodes, GPU0/1<->GPU2/3 = SYS (cross-socket). Measured inter-GPU bw = 5.4 GB/s (all pairs). Reference = A100-SXM4 NVLink ~600 GB/s (~100x faster). Whole-model FSDP wrap (forced: MoT manual attention) all-gathers 13GB (6.7B params) every fwd/bwd -> ~2.5s/gather x many -> dominates training. Rollout has model resident (no FSDP gather) so it matched.
- I made ZERO RLinf code edits (git diff empty); only added config libero_spatial_grpo_fastwam.yaml + run_rl.sh. pidfd fix = allocator env var only (no correctness impact). So no regression introduced.
- MITIGATION being tested: reshard_after_forward=False in the grpo config (keep 6.7B resident after forward -> all-gather ~once/step not per-microbatch; only ~1B action expert has grads so reduce-scatter is small). Run /workspace/results/rl_grpo64b (exp libero_spatial_grpo_fastwam_64b). If it helps, RL is feasible here; else hardware-limited.

Link: [[fastwam-rlinf-sft-verify]]
