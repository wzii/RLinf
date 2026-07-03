#!/usr/bin/env python
"""Collect LIBERO-130 eval SRs from all eval130_{base,rl}_{suite} tensorboard dirs.
Prints a per-suite base-vs-RL table and writes a markdown report. Missing runs -> '-'."""
import os, glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS = "/workspace/results"
REPORT = "/workspace/results/EVAL130_base_vs_rl.md"
# display order: the 4 "atomic/generalization" suites, long (trained), short(90)
SUITES = [("spatial","libero_spatial",240,10),
          ("object","libero_object",280,10),
          ("goal","libero_goal",300,10),
          ("long","libero_10",520,10),
          ("short","libero_90",240,90)]
SUITE_LABEL = {"spatial":"Spatial","object":"Object","goal":"Goal",
               "long":"Long (libero_10, RL-trained)","short":"Short (libero_90)"}

def load(run):
    tb = os.path.join(RESULTS, run, "tensorboard")
    if not glob.glob(os.path.join(tb, "events.out.tfevents.*")): return None
    ea = EventAccumulator(tb, size_guidance={"scalars": 0}); ea.Reload()
    return {t: ea.Scalars(t)[-1].value for t in ea.Tags().get("scalars", [])}

def sr(d, k="eval/success_once"):
    return d.get(k) if d else None

def wmean(items):
    if not items: return None
    return sum(v*n for v,n in items)/sum(n for _,n in items)

rows, overall = [], {"base":[], "rl":[]}
per_suite = {}
for name, ts, hz, ntask in SUITES:
    b = load(f"eval130_base_{name}"); r = load(f"eval130_rl_{name}")
    bs, rs = sr(b), sr(r)
    per_suite[name] = (b, r, bs, rs, ntask, hz)
    d = (rs-bs) if (bs is not None and rs is not None) else None
    rows.append((name, hz, ntask, bs, rs, d))
    if bs is not None: overall["base"].append((bs, ntask))
    if rs is not None: overall["rl"].append((rs, ntask))

# ---- console ----
print(f"{'suite':8} {'hz':>4} {'base':>7} {'rl':>7} {'delta':>7}")
print("-"*40)
for name,hz,nt,bs,rs,d in rows:
    fb=f"{bs:.3f}" if bs is not None else "  -  "
    fr=f"{rs:.3f}" if rs is not None else "  -  "
    fd=f"{d:+.3f}" if d is not None else "  -  "
    print(f"{name:8} {hz:>4} {fb:>7} {fr:>7} {fd:>7}")
mb,mr=wmean(overall["base"]),wmean(overall["rl"])
print("-"*40)
print(f"LIBERO-130 task-weighted: base={mb and round(mb,4)} rl={mr and round(mr,4)}")

# ---- markdown ----
def f3(x): return f"{x:.3f}" if x is not None else "—"
def fd(x): return f"{x:+.3f}" if x is not None else "—"
L=[]
L.append("# FastWAM GRPO — LIBERO-130 eval: RL step10 vs base (round15)\n")
L.append("Deterministic full eval, **all 50 init-states/task**, per-suite max-horizon, "
         "**100 parallel envs** (25/GPU), zero measurement variance. The RL ckpt was "
         "trained **only on libero_10 (Long)**; the other four suites test "
         "generalization / forgetting.\n")
L.append("| Suite | tasks | horizon | base SR | RL SR | Δ |")
L.append("|---|---:|---:|---:|---:|---:|")
for name,hz,nt,bs,rs,d in rows:
    L.append(f"| {SUITE_LABEL[name]} | {nt} | {hz} | {f3(bs)} | {f3(rs)} | {fd(d)} |")
L.append(f"| **LIBERO-130 (task-weighted)** | **130** | — | **{f3(mb)}** | **{f3(mr)}** | **{fd((mr-mb) if (mb and mr) else None)}** |")
L.append("")
L.append("**Notes.** SR = `success_once` (task solved at any step). Absolute SRs are at "
         "100-env batch and can differ ±~0.02 from 20-env measurements due to forward "
         "numerics; base and RL are compared at the **identical 100-env config**, so Δ is clean. "
         "RL ckpt = `global_step_10` (dcp, HF `HardToFindAGoodUserName/fastwam-rl-libero10`), "
         "loaded via RLinf `resume_dir`; base = `round15_robust_step_005000.pt`.\n")
# per-task: full for 10-task suites, biggest movers for libero_90
L.append("## Per-task success_once\n")
for name, ts, hz, ntask in SUITES:
    b,r,bs,rs,nt,hz2 = per_suite[name]
    if b is None and r is None: continue
    L.append(f"### {SUITE_LABEL[name]} — base {f3(bs)} / RL {f3(rs)} / Δ {fd((rs-bs) if (bs is not None and rs is not None) else None)}")
    tasks=[]
    for i in range(nt):
        bk=b.get(f"eval/success_once_task{i}") if b else None
        rk=r.get(f"eval/success_once_task{i}") if r else None
        if bk is None and rk is None: continue
        dd=(rk-bk) if (bk is not None and rk is not None) else None
        tasks.append((i,bk,rk,dd))
    if nt<=10:
        L.append("| task | base | RL | Δ |"); L.append("|---:|---:|---:|---:|")
        for i,bk,rk,dd in tasks:
            L.append(f"| {i} | {f3(bk)} | {f3(rk)} | {fd(dd)} |")
    else:
        movers=sorted([t for t in tasks if t[3] is not None], key=lambda t:t[3])
        L.append(f"_90 tasks; showing 10 biggest regressions and 10 biggest improvements._\n")
        L.append("| task | base | RL | Δ |"); L.append("|---:|---:|---:|---:|")
        for i,bk,rk,dd in movers[:10]+movers[-10:]:
            L.append(f"| {i} | {f3(bk)} | {f3(rk)} | {fd(dd)} |")
    L.append("")
open(REPORT,"w").write("\n".join(L))
print(f"\nwrote {REPORT}")
