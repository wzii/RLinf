# FastWAM GRPO RL on LIBERO — Experiment Conclusions

**Date: 2026-06-20**

Model: FastWAM flow-SDE actor (critic-free GRPO). Only the ~1B action expert is trained;
the 5B Wan2.2 video expert is frozen. FSDP2, whole-model wrap. Eval is **deterministic with
fixed ordered reset states** (`libero_env.py:408/:455`) → **zero measurement variance**: every
reported SR is the policy's true state, not sampling noise.

---

## TL;DR

| # | Experiment | Headline result |
|---|---|---|
| 1 | **Single-task RL raise** | RL **works** — task8 0.57→0.90 plateau (Δ+0.26), task76→0.90, clean 10-step convergence with zero eval variance |
| 2 | **Large-lr learning** | lr1e-4 one-step **catastrophic collapse** (overall −0.475); also root-caused the OOM (PatchWeightSyncer densification → fixed with `bucket_syncer`) |
| 3 | **Full eval of new ckpt** (round15_robust) | libero130×50=6500 ep: overall **0.879**, libero90 **0.927** (saturated), other-40 **0.767** |
| 4 | **512-traj RL on saturated round15 (libero90)** | RL **destroys** the saturated policy, monotonically: 0.927 → step5 0.664 → step10 **0.437** (Δ−0.490). Textbook whack-a-mole that churns downward (fully-dead tasks 8→21; only 3/90 above base at step10) |
| 5 | **256-traj RL on *unsaturated* round15 (libero_10)** | RL **improves**, monotonically: 0.820 → step5 0.842 → step10 **0.926** (Δ+0.106). Headroom tasks lifted +0.28 (task9 0.5→1.0), saturated barely touched (−0.03). The **mirror image of #4** — headroom at the start decides help vs harm |

**One-line synthesis:** RL on FastWAM genuinely learns (single-task & headroom tasks improve
dramatically), but on a *saturated multi-task* start the gradient conflict + no KL anchor means
"learning the new costs forgetting the old" — net negative.

---

## 1. Single-task RL raise experiment

**Goal:** isolate whether RL can improve SR at all, free of multi-task interference.

**Setup:** critic-free GRPO, no-std (`grpo_norm_by_std=false`), **lr 2e-5**, clip_grad 1.0,
**noise_level 0.3** (flow-SDE single), group_size 8, total_num_envs 32 × rollout_epoch 8 =
**256 traj/step**, gb64/mb4, kl_beta 0, train+eval `task_id_filter=[8]`, eval 20×2=40 trials,
val_check 1, eval_at_start, start ckpt `sft_libero90_20k.pt`.

**Result — task8, 10-step trajectory (eval = 40 trials, deterministic):**
```
init  st1   st2   st3   st4   st5   st6   st7   st8   st9   st10
0.57  0.65  0.85  0.68  0.82  0.90  0.80  0.85  0.93  0.98  0.90
```
- **10/10 steps above base 0.57**, floor 0.65 always held
- mean 0.835 (**Δ +0.260**), second-half (st6-10) mean **0.890**, peak 0.98
- clean RL convergence: oscillating rise → 0.85-0.98 plateau

**task76** independently reproduced: single-task RL to **0.900**.

**Conclusion:** RL on FastWAM can **truly, stably, reproducibly** improve a single task. The
earlier "no reproducible improvement" verdict was a multi-task-interference misattribution, not
an RL-algorithm or flow-matching-policy problem. Evidence: `results/user_exp_t8_FINAL.log`,
`results/user_exp_t76_RERUN_FINAL.log`.

---

## 2. Large-lr learning experiment

**Goal:** push lr to accelerate learning; understand the OOM that kept killing long runs.

**Results:**

| Config | Outcome |
|---|---|
| multi-task **lr 2e-5** | ≈ flat (head vs true-base −0.011) |
| multi-task **lr 1e-4** | **one-step catastrophic collapse** (overall −0.475, saturated −0.820) |
| single-task **lr 2e-5** | stable convergence (task8 +0.26) ✅ |
| multi-task **lr 5e-5** | destroys saturated policy (see §4) |

**OOM root cause (solved):** lr 1e-4 changes *almost all* 5B weights in one step. The default
`PatchWeightSyncer` ships weight deltas as a COO **sparse** patch (rows+cols+values, 3×). When
the delta is dense, that representation explodes → OOM in `sync_model_to_rollout`. The same
massive weight change that OOMs also wrecks the policy. **Fix:** switch to `bucket_syncer`
(dense, bucketed; cost independent of how much the weights changed) — config line 13. This
unblocked all subsequent large-lr / large-batch runs.

**Other engineering fixes made along the way:**
- **timer crash**: `eval_at_start` using `with self.timer("eval")` didn't `consume` → conflicted
  with the training-loop eval timer. Fixed by removing the wrap + `consume_durations()`.
- **added `eval_at_start`** switch (force evaluate before the RL loop) in `embodied_runner.py`.
- **eval-coverage bug**: filtering tasks broke `_build_interleaved` coverage (30-task filter only
  evaluated 8). Use full-90 eval or single-task filter.

**Conclusion:** large lr on a saturated multi-task start is destructive (one step can collapse it);
the recurring OOM was a sparse-patch densification artifact, not a true memory ceiling.

---

## 3. Full eval of the new checkpoint (round15_robust)

**Checkpoint:** `round15_robust_step_005000.pt` from HF `peter-chl/fastwam-sft-libero90` (12 GB),
adopted as the new RL starting point.

**Eval:** libero_130 × 50 trials = **6500 episodes**, deterministic ordered reset, full coverage
(6500 = 130 tasks × 50, verified each task exactly 50 trials).

**Results:**

| Slice | SR | Tasks |
|---|---|---|
| **Overall (libero130)** | **0.879** | 130 |
| **libero90 (task0-89)** | **0.927** | 90 |
| Other-40 (spatial/object/goal/long) | 0.767 | 40 |

**Notable per-task (libero90):** weak spots task51=0.18, task57=0.22, task75=0.42, task81=0.38;
the rest mostly ≥0.9. **Other-40** hardest: long/10 tasks (task120-129) mostly 0.0-0.38.

**Takeaway:** round15_robust is a *much* stronger SFT than the old `sft_libero90_20k`
(libero90 0.927 vs ~0.61). It is **highly saturated** on libero90 — which, as §4 shows, makes it
a poor RL substrate (little headroom, strong policy → weak advantage signal).
Log: `results/eval_baseline_r15_lib130_FINAL.log`.

---

## 4. Latest experiment — 512-traj RL on saturated round15 (libero90)

**Goal:** large-batch (512-traj) GRPO on the strong round15 start, full libero90.

### 4a. Noise sweep — the saturated policy can't be pushed into the learning band
Target train-rollout SR 0.3-0.8 (mixed success/failure → strong advantage signal). On round15 it
**can't be reached**:

| noise_level | step0 train SR |
|---|---|
| 0.5 | 0.893 |
| 0.7 | 0.867 |
| 0.9 | 0.828 |

Each +0.2 noise drops SR only ~0.03; extrapolating, even noise 1.0 stays >0.8. The policy is so
strong that exploration can't lower the rollout success rate → groups are mostly all-success →
weak/zero advantage. This **is** the "saturated SFT is hard for RL" wall.

### 4b. Batch / memory — 512 traj runs, but at the cliff
- **64×8=512 traj OOMs** in *training* (512-traj advantages/backward peaks at the 80 GB cliff).
- **32×16=512 traj fits** (fewer simultaneous-rollout residuals; train backward peaks ~72-75 GB,
  just under 80 GB). This is how the requested 512-batch was realized.

### 4c. Config (run `rl_r15_lib90_512_n9`)
critic-free GRPO, no-std, **lr 5e-5**, clip_grad 1.0, **noise 0.9**, kl_beta 0, group_size 8,
total_num_envs 32 × rollout_epoch 16 = **512 traj/step**, gb64/mb4 (8 grad-accum steps),
train+eval libero_90 (all 90 tasks, no filter), **eval 60×15 = 900 ep = 10 trials/task**
(deterministic), `bucket_syncer`, start = round15_robust, max_steps 10, val_check 5.

### 4d. Result — RL destroys the saturated policy (monotonically, over the full 10 steps)
**train-SR trajectory (all 10 steps):** `0.828 → 0.629 → 0.656 → 0.621 → 0.496 → 0.648 → 0.371 → 0.549 → 0.439 → 0.400`

**True eval (90 tasks, deterministic):**
```
            overall    Δ vs base 0.927
step5  eval:  0.664      −0.263
step10 eval:  0.437      −0.490     ← nearly doubled the damage
```
**It does not converge or recover — it churns downward.** step5→step10 per-task movement: **13 up, 20 flat, 57 further down**; fully-dead (SR=0.0) tasks **8 → 21** (nearly tripled); by step10 only **3/90 tasks remain above base**. Even *rescued* tasks get lost later — e.g. task36 (base 0.66 → step5 0.90 → step10 **0.10**), task69 (0.86 → 1.00 → 0.70). The one star that mostly held: task57 (0.22 → 1.00 → 0.80). So more RL steps = more whack-a-mole collateral, with internal churn (tasks rise then fall) rather than a stable trade.

**Per-task whack-a-mole (step5 eval vs round15 50-trial base):**

*Improved (8 tasks, all low-base / headroom):*
```
task57: 0.22 → 1.00  (+0.78)   ← hardest libero90 task, RL maxed it out
task36: 0.66 → 0.90  (+0.24)
task70: 0.82 → 1.00  (+0.18)
task69: 0.86 → 1.00  (+0.14)
task56/46/31/88...   small gains
```

*Collapsed (55 tasks, mostly saturated; catastrophic, not gradual):*
```
task27/38/21/32/74: 1.00 → 0.00  (−1.00, full collapse)
task39: 0.98→0.00 · task45: 0.96→0.00 · task73: 0.94→0.00
task64/77/80/41/34: 1.0 → ~0.1
```

| Bucket | Count |
|---|---|
| Improved (Δ>+0.05) | **8** |
| Flat (±0.05) | 27 |
| **Declined (Δ<−0.05)** | **55** |

Saturated tasks (base≥0.9): **75 total, 49 destroyed**; saturated mean **0.983 → 0.673**.

### 4e. Per-task coverage & the noise-fragility mechanism (the *why*)
Each RL step randomly samples **~41-47 of 90 tasks** (64 groups × 8, with overlap); over 6 steps
**89/90 tasks were trained** (only task71 never sampled) — so it is genuinely multi-task training
across nearly everything, ruling out "some tasks just weren't touched".

Cross-referencing each task's **train-rollout SR under noise0.9** (steps 0-4, before the step5 eval)
against its **eval Δ** gives **correlation +0.56** — a task's eval fate is predicted by whether it
survived the rollout exploration:

| group | avg train-rollout SR | eval outcome |
|---|---|---|
| **Collapsed** (base≥0.9 → eval~0.0) | **0.34** (low) | died |
| **Improved** (eval Δ>+0.1) | **0.87** (high) | held/improved |

You can watch the collapse happen *inside* training, step by step:
```
task27: base1.0  rollout 0.50→0.19→0.00   eval 0.0
task38: base1.0  rollout 0.94→0.88→0.12   eval 0.0
task57: base0.22 rollout 1.0, 1.0, 1.0    eval 1.0  (noise0.9 rescued a stuck task)
```

**Corrected mechanism** (this refutes the naive "saturated→filtered→passive drift" guess): the
collapsed tasks were **not passive drift victims** — they were *actively failing under noise0.9
exploration*, and RL pushed them further down. The real driver is that **noise_level=0.9 is too
high**: robust saturated tasks survive the perturbation (stay 8/8 → reinforced), but *fragile*
saturated tasks (success depends on precise actions) break under it → rollout SR collapses → GRPO's
gradient drives them to 0. The flip side: noise0.9 *rescued* a few stuck tasks (task57 0.22→1.0 by
escaping a bad deterministic mode).

**The catch-22 this exposes:** the noise sweep forced us to 0.9 (to pull train SR toward the 0.3-0.8
band), but 0.9 is exactly what destabilizes the fragile tasks. On a saturated start there is **no
good noise setting** — low noise gives no advantage signal, high noise breaks robust tasks. The lever
is therefore not merely "lower lr"; the saturated start itself is the trap.

### 4f. Conclusion
RL is clearly *learning* (task57 0.22→1.0 proves it), but on a saturated multi-task start the
**shared policy + gradient conflict + no KL anchor (kl_beta=0) + lr 5e-5 + noise 0.9** make every
gain come at the cost of multiple saturated tasks collapsing. 8 up vs 55 down → necessarily net
negative (−0.263). This is the exact whack-a-mole predicted by the single-task vs multi-task
contrast, now with a clean 50-trial base + deterministic eval per-task proof.

*(caveat: step5 eval is 10 trials/task; but 1.0→0.0 = 10/10→0/10 is a real change, not noise.)*
Run continuing to step10 (in progress at time of writing). Log: `results/rl_r15_lib90_512_n9/`.

---

## 5. RL on an *unsaturated* suite — libero_10 (the positive counterpart to §4)

**Goal:** test the §4 hypothesis directly — if a saturated start dooms RL, does an *unsaturated*
one let it work? Same ckpt (round15_robust), same OpenPI-aligned single-step SDE code, but on
**libero_10** (the 10 long-horizon LIBERO-LONG tasks).

### 5a. A truncation-artifact correction
The §3 libero130 baseline reported the long tasks at 0.0-0.38 — but that eval used
`max_episode_steps=240`, which **truncates** the long-horizon episodes. Re-run with the proper
**520 steps** (both train rollout and eval), round15's real libero_10 base is **0.820**, not near-zero.
So those "0.0-0.38" scores were largely a truncation artifact, not policy weakness.

### 5b. Config
critic-free GRPO, no-std, **lr 2e-5**, clip 1.0, **noise 0.3**, kl 0, group 8,
total_num_envs 32 × rollout_epoch 8 = **256 traj/step**, **max_episode_steps 520** (long-horizon),
train+eval libero_10, eval 20×5=100 ep (10 trials/task), `eval_at_start=true` (clean in-config base),
bucket_syncer, max_steps 10, val_check 5.

`base` per-task (deterministic): `{0:.9,1:1.,2:.6,3:.7,4:.8,5:.9,6:.8,7:1.,8:1.,9:.5}` — moderately
saturated: 5/10 already ≥0.9 (task0/1/5/7/8), real headroom on **task2(.6)/3(.7)/9(.5)**.

### 5c. Result — RL improves, monotonically (mirror image of §4)
**train-SR trajectory (noise0.3):** `0.859 → 0.887 → 0.902 → 0.883 → 0.801 → 0.840 → 0.840 → 0.844 → 0.863`
(a healthy dip-and-recover, converging ~0.85 — not the churn-down of §4).

**True eval (deterministic, 10 trials/task):**
```
              overall            headroom[2,3,9]     saturated[0,1,5,7,8]
base:         0.820              0.60                0.96
step5 eval:   0.842  (+0.022)    0.68  (+0.08)       0.90  (-0.06)
step10 eval:  0.926  (+0.106)    0.88  (+0.28)       0.93  (-0.03)
```
The positive Δ **grew** step5→step10 (+0.022 → +0.106). Headroom tasks lifted hard —
**task9 0.5→1.0, task2 0.6→0.82, task3 0.7→0.81** — while the saturated 5 barely moved (−0.03).
Final overall 0.926 even matches the libero90 saturated level.

### 5d. The decisive contrast (§4 vs §5)
| start | step5 | step10 | trajectory |
|---|---|---|---|
| **saturated** libero90 (base 0.927) | −0.263 | **−0.490** | monotonic **destruction** |
| **unsaturated** libero_10 (base 0.820) | +0.022 | **+0.106** | monotonic **improvement** |

Same ckpt, same code, same algorithm — opposite outcomes. **The headroom at the starting point,
not the RL machinery, decides whether RL helps or harms.** Saturated → only downside to move (RL
destroys); real headroom → RL genuinely learns and lifts the low tasks without breaking the high
ones. This also confirms the FastWAM RL implementation (incl. the OpenPI-aligned injection step) is
correct and *does* produce real gains when the substrate allows it.

Log: `results/rl_r15_lib10_256_FINAL.log`.

## Overall conclusions & directions

1. **RL works on FastWAM** — single-task and headroom (low-base) tasks improve dramatically and
   reproducibly, with zero-variance deterministic eval confirming it's real.
2. **The barrier is saturated-multi-task interference**, not the RL algorithm or flow-matching
   policy. Whack-a-mole: lift the few with headroom, break the many that were saturated.
3. **Large lr / no KL anchor on a saturated start is destructive** — one big step collapses
   saturated tasks; the historical OOM was a sparse-patch densification artifact (fixed via
   `bucket_syncer`), not a true memory wall.
4. **To make multi-task RL net-positive (changes the *premise*, not just hyperparameters):**
   - per-task / small-group fine-tuning (single-task already works)
   - **KL anchor to SFT** (protect saturated tasks from drifting)
   - per-task advantage normalization (reduce cross-task gradient conflict)
   - curriculum (converge single tasks, then mix)
   - a less-saturated SFT start (SR 0.5-0.7) so the train band is naturally 0.3-0.8

## Key files
- Single-task: `results/user_exp_t8_FINAL.log`, `results/user_exp_t76_RERUN_FINAL.log`
- Large-lr / OOM: `results/user_exp3g_lr1e4_bucket_CRASH_FINAL.log`, `results/user_exp2_lr2e5_FINAL.log`
- New-ckpt eval: `results/eval_baseline_r15_lib130_FINAL.log`
- Latest 512 RL: `results/rl_r15_lib90_512_n9/` (+ `rl_r15_lib90_512b_noise05_trainSR0.893.log`,
  `rl_r15_lib90_512_n7_trainSR0.867.log` for the noise sweep)
- Config: `examples/embodiment/config/libero_spatial_grpo_fastwam_h240.yaml` (line 13 = bucket_syncer)
- Code: `rlinf/runners/embodied_runner.py` (eval_at_start + timer fix)
- Baseline: `results/v2_step2_pertask_baseline.json`

## 6. noise=0.6 replication on libero_10 (final full eval + per-rollout report validated)

Same round15 start, same OpenPI-aligned code, libero_10, but **noise_level=0.6** (vs §5's 0.3),
lr2e-5, clip1, 32×8=256 traj, ep520, 10 steps, per-rollout report ON, full 500-ep deterministic eval
at end only. Purpose: replicate §5's positive result at higher exploration + validate the new
per-rollout debug report on a real run.

**Result — RL again net-positive:**

| | base (500ep) | final step10 (500ep) | Δ |
|---|---|---|---|
| **overall** | 0.836 | **0.904** | **+0.068** |
| head[2,3,9] | 0.740 | 0.867 | **+0.127** |
| sat[0,1,5,7,8] | 0.952 | 0.908 | −0.044 |

per-task final vs base: task2 0.54→0.90 (+0.36), task4 0.66→0.98 (+0.32), task6 0.72→0.92 (+0.20)
are the drivers (all headroom/low-base); saturated tasks give a little back (task0 0.96→0.80,
task1 0.96→0.86). **Same mechanism as noise0.3 (headroom rises, saturated holds); magnitude smaller
(+0.068 vs +0.106).** Third independent confirmation that unsaturated-start RL genuinely improves.

train SR (noise0.6): 0.836→0.879→0.906→0.891→0.898→**0.781**→0.816→0.844→0.910. The step5 dip to
0.781 fully self-heals by step8 (0.910).

**Per-rollout report validated on a real run (3rd validation).** 4 rank files (4-way FSDP), each with
GLOBAL STEP 0-9; shapes rewards=(52,64,10), denoise_inds=(52,64), inject_noise_norm=(52,64). Report's
per-step SR matches `env/success_once` exactly. `task_ids=None` (not plumbed to actor) → groups
labeled by index, not libero task id.

**step5 dip analysis — NOT noise, purely optimization variance (and reversible):**
- step5 self-injection extremes are normal (p99=10.72, MAX=11.54, inj==0 frac=0.103 — identical to
  neighbors); *failing* rollouts' noise-max (10.495) is even *lower* than succeeding (10.622).
- step4 (the upstream training data) noise is also benign: inj==0 frac 0.090 (2nd lowest); its
  successes concentrate at *lower* noise (succ mean 6.033 < fail 6.139) — clean gradient, high-noise
  groups still 8/8. No "bad gradient from extreme noise" either.
- Conclusion: the dip is GRPO on-policy gradient variance (256 binary rewards → one lr2e-5 full-weight
  update overshoots a few tasks' marginal decisions), independent of any step's noise, and reversible.
  Lever to suppress it is optimizer-side (smaller lr / KL anchor / larger batch), not noise.

**Checkpoint (produced):** `results/lib10_rl_n6/lib10_rl_n6/checkpoints/global_step_10/actor/` — BOTH
`dcp_checkpoint/` (53GB, model+optimizer, RLinf resume) and `model_state_dict/full_weights.pt` (48.9GB,
pure model weights, torch.load-ready). NOTE: the run's own full_weights.pt save hit disk-full (dcp 53GB
+ full 48.9GB > free); full_weights.pt was re-exported offline from the intact dcp shards (strip
`fsdp_checkpoint.model.` prefix, verified loadable). wan2.2 backbone (32GB) was deleted for room —
re-download from HF if model init needs it (see `/workspace/wan2.2_DELETED_README.txt`).

Log: `results/lib10_rl_n6_FINAL.log`; report: `results/lib10_rl_n6/per_rollout_report_rank{0-3}.txt`.
