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

## Files

- `FASTWAM_RL_CONCLUSIONS_2026-06-20.md` — full experiment write-up (all runs, §1–6)
- `lib10_rl_n6_FINAL.log` — training + final full-eval log
- `per_rollout_report_rank0.txt` — per-rollout debug report sample (1 of 4 FSDP ranks)

## Checkpoint

RL checkpoint (global_step_10) is on the Hub:
[`HardToFindAGoodUserName/fastwam-rl-libero10`](https://huggingface.co/HardToFindAGoodUserName/fastwam-rl-libero10)
(`dcp_checkpoint/` format — model + optimizer, resumable via RLinf `load_checkpoint`).
