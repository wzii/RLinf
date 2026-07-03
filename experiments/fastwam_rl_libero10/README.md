# FastWAM GRPO RL on LIBERO-10 (round15 start, noise0.6)

Critic-free flow-SDE GRPO fine-tuning of a FastWAM policy on the LIBERO-10 suite,
starting from the round15 SFT checkpoint. Full write-up in
[`FASTWAM_RL_CONCLUSIONS_2026-06-20.md`](FASTWAM_RL_CONCLUSIONS_2026-06-20.md).

## Result

Deterministic full eval (500 ep, libero_10, fixed reset states → zero measurement variance):

| | base (500ep) | step10 (500ep) | Δ |
|---|---|---|---|
| **overall** | 0.836 | **0.904** | **+0.068** |
| head[2,3,9] | 0.740 | 0.867 | +0.127 |
| sat[0,1,5,7,8] | 0.952 | 0.908 | −0.044 |

Headroom (low-base) tasks drive the gain (task2 0.54→0.90, task4 0.66→0.98, task6 0.72→0.92);
saturated tasks hold. Third independent confirmation that unsaturated-start RL genuinely improves.

## Config

round15 SFT start · libero_10 · GRPO `noise_level=0.6` · lr 2e-5 · clip 1.0 ·
no-std (`grpo_norm_by_std=false`, Dr.GRPO-style) · 32×8 = 256 traj · ep520 · 10 steps ·
`bucket_syncer` weight sync · `per_rollout_report=true`.

train SR: 0.836→0.879→0.906→0.891→0.898→**0.781**→0.816→0.844→0.910. The step5 dip
self-heals by step8; analysis shows it is GRPO on-policy gradient variance, not noise
(see §6 of the conclusions doc).

## LIBERO-130 generalization eval (2026-07-03)

Both checkpoints evaluated on the **full LIBERO-130 benchmark** (all 5 suites, all 50
init-states/task = 6500 ep/ckpt), each suite at its proper max-horizon, 100 parallel
envs, deterministic ordered reset states → zero measurement variance. Base and RL are
compared at the identical 100-env config (absolute SRs can differ ±~0.02 from 20-env
runs due to forward numerics; the Δ is clean).

| Suite | tasks | horizon | base SR | RL SR | Δ |
|---|--:|--:|--:|--:|--:|
| Spatial | 10 | 240 | 0.962 | 0.972 | +0.010 |
| Object | 10 | 280 | 0.972 | 0.912 | −0.060 |
| Goal | 10 | 300 | 0.956 | 0.962 | +0.006 |
| **Long (libero_10, RL-trained)** | 10 | 520 | 0.852 | 0.906 | **+0.054** |
| Short (libero_90) | 90 | 240 | 0.914 | 0.850 | −0.065 |
| **LIBERO-130 (task-weighted)** | **130** | — | **0.921** | **0.877** | **−0.044** |

**Single-suite RL specialization/forgetting tradeoff.** GRPO on libero_10 gives a real,
headroom-driven gain **on the trained suite** (+0.054; task2 0.56→1.00, task4 0.68→0.96)
and holds spatial/goal, but the action expert specializes toward libero_10's long-horizon
behavior and **mildly forgets** the rest: object −0.060 and libero_90 −0.065 (54/90 tasks
worse; biggest task40 0.88→0.36). Because libero_90 is 90 of the 130 tasks, forgetting
dominates the aggregate → full-130 SR drops −0.044. The libero_10 anchor 0.906 ≈ the
training-time full eval 0.904, validating the resume→eval pipeline end-to-end.

Full per-suite/per-task numbers in [`EVAL130_base_vs_rl.md`](EVAL130_base_vs_rl.md).

### Reproduce

```bash
bash eval130_scripts/run_eval130.sh          # drives all 10 runs (base+RL × 5 suites)
python eval130_scripts/collect_sr.py         # rebuilds the table from tensorboard
```

Eval-only recipe: `runner.max_steps=0 +runner.eval_at_start=true`, RL ckpt via
`runner.resume_dir=.../global_step_10`, base via `actor.model.checkpoint_path=round15`.
Episodes = `env.eval.rollout_epoch × total_num_envs` (100 envs → rollout_epoch 5 for
10-task suites, 45 for libero_90 → exactly 50/task). Backbone note: the DiffSynth-Studio
converted-safetensors repo is 404, so FastWAM `configs/model/fastwam.yaml` needs
`redirect_common_files: false` to fetch T5+VAE from `Wan-AI/Wan2.2-TI2V-5B`; pre-download
the T5 single-process (concurrent FSDP ranks race on it).

## Files

- `FASTWAM_RL_CONCLUSIONS_2026-06-20.md` — full experiment write-up (all runs, §1–6)
- `lib10_rl_n6_FINAL.log` — training + final full-eval log
- `per_rollout_report_rank0.txt` — per-rollout debug report sample (1 of 4 FSDP ranks)
- `EVAL130_base_vs_rl.md` — LIBERO-130 base-vs-RL report (per-suite + per-task)
- `eval130_logs/` — the 10 eval run logs + driver log
- `eval130_scripts/` — `run_eval.sh`, `run_eval130.sh`, `extract_sr.py`, `collect_sr.py`

## Checkpoint

RL checkpoint (global_step_10) is on the Hub:
[`HardToFindAGoodUserName/fastwam-rl-libero10`](https://huggingface.co/HardToFindAGoodUserName/fastwam-rl-libero10)
(`dcp_checkpoint/` format — model + optimizer, resumable via RLinf `load_checkpoint`).
