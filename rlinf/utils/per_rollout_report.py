"""Per-rollout debugging report for FastWAM flow-SDE GRPO.

Opt-in (config flag ``runner.per_rollout_report``, default off). Generated at the
actor's ``compute_advantages_and_returns`` from the assembled ``rollout_batch``:

    Global Step N
      Group g | task=<idx or group-proxy> | SR=k/group_size
        inject-step seq (group-shared): [s_0, s_1, ...]   (per decision-step)
        rollout r | reward=<sum> | success/fail | noise-norm seq: [n_0, n_1, ...]

NOTES on the data model (verified from the FastWAM rollout code):
 * The noise-injection denoise step is chosen PER decision-step (one ``predict_action_batch``
   call), so each trajectory has a *sequence* of inject steps (~52 for a 520-step libero_10
   episode). The 8 group members are batched together at each step → they share the same
   inject-step sequence; only the injected gaussian *value* (hence ``inject_noise_norm``)
   differs per rollout. That is why inject-step is reported at the GROUP level and noise-norm
   at the ROLLOUT level.
 * ``task_id`` is env-side and is not currently plumbed to the actor; we report the group
   index as a proxy (task is constant within a group). Pass ``task_ids`` if available.

The whole thing is wrapped by the caller in try/except so a shape mismatch can never break
training. On the FIRST call it dumps the raw tensor shapes (``_SHAPES_LOGGED``) so the exact
layout can be validated against a real run and the formatting finalized.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

_SHAPES_LOGGED = False


def _to_traj_major(t: torch.Tensor, n_traj: int) -> Optional[torch.Tensor]:
    """Return ``t`` reshaped to ``[n_traj, seq]`` (seq may be 1).

    Finds the dimension equal to ``n_traj`` (the trajectory/batch dim) and flattens the
    rest into a sequence dim. Returns None if no dim matches ``n_traj``.
    """
    if t is None:
        return None
    t = t.detach().float().cpu()
    if t.ndim == 1:
        return t.view(n_traj, 1) if t.shape[0] == n_traj else None
    # locate the trajectory dim
    dims = list(t.shape)
    if n_traj not in dims:
        # last resort: if total divides n_traj, fold
        if t.numel() % n_traj == 0:
            return t.reshape(n_traj, -1)
        return None
    bdim = dims.index(n_traj)
    t = t.movedim(bdim, 0)  # [n_traj, ...]
    return t.reshape(n_traj, -1)


def _fmt_seq(x: torch.Tensor, maxn: int = 64, intish: bool = False) -> str:
    vals = x.tolist()
    if intish:
        body = ",".join(str(int(round(v))) for v in vals[:maxn])
    else:
        body = ",".join(f"{v:.2f}" for v in vals[:maxn])
    more = "" if len(vals) <= maxn else f",…(+{len(vals)-maxn})"
    return f"[{body}{more}]"


def write_per_rollout_report(
    rollout_batch: dict[str, Any],
    group_size: int,
    global_step: int,
    out_dir: str,
    task_ids: Optional[torch.Tensor] = None,
    shard_id: int = 0,
) -> Optional[str]:
    """Build the hierarchical per-rollout report string and append it to a file.

    Returns the path written, or None if the required tensors were absent. Designed to be
    called inside a try/except by the actor so it can never disrupt training.
    """
    global _SHAPES_LOGGED

    rewards = rollout_batch.get("rewards")
    fwd = rollout_batch.get("forward_inputs", {}) or {}
    denoise = fwd.get("denoise_inds")
    nnorm = fwd.get("inject_noise_norm")
    if rewards is None:
        return None

    rewards = rewards.detach().float().cpu()
    # rewards expected [n_chunk_step, B, num_action_chunks] -> B is dim 1 if 3D, else infer.
    if rewards.ndim >= 2:
        B = rewards.shape[1]
    else:
        B = rewards.shape[0]
    if B % group_size != 0:
        # can't form clean groups; bail (report stays best-effort)
        B = (B // group_size) * group_size
        if B == 0:
            return None
    n_groups = B // group_size

    # per-trajectory scalar reward (sum over time + chunks) and success (any positive reward)
    rsum = rewards.reshape(rewards.shape[0], rewards.shape[1], -1)[:, :B] if rewards.ndim >= 3 else rewards[:, :B].unsqueeze(-1)
    per_traj_reward = rsum.sum(dim=(0, 2))          # [B]
    per_traj_success = (rsum.amax(dim=(0, 2)) > 0.5)  # [B] bool

    di = _to_traj_major(denoise, B)   # [B, seq] or None
    nn = _to_traj_major(nnorm, B)     # [B, seq] or None
    tid = None
    if task_ids is not None:
        tid = task_ids.detach().cpu().reshape(-1)
        tid = tid[:B] if tid.shape[0] >= B else None

    os.makedirs(out_dir, exist_ok=True)
    # one file per rank/shard: under FSDP each rank holds a disjoint slice of the groups,
    # so per-rank files are both race-free and complete (read all shards for every group).
    path = os.path.join(out_dir, f"per_rollout_report_rank{shard_id}.txt")
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"GLOBAL STEP {global_step}  |  rank/shard {shard_id}  |  "
        f"{n_groups} group(s) x {group_size} rollouts  (B={B})"
    )
    if not _SHAPES_LOGGED:
        lines.append(
            f"[shapes] rewards={tuple(rewards.shape)} "
            f"denoise_inds={tuple(denoise.shape) if denoise is not None else None} "
            f"inject_noise_norm={tuple(nnorm.shape) if nnorm is not None else None} "
            f"task_ids={tuple(task_ids.shape) if task_ids is not None else None}"
        )
        _SHAPES_LOGGED = True
    for g in range(n_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        succ = per_traj_success[sl]
        sr = float(succ.float().mean())
        task_label = f"task={int(tid[g * group_size])}" if tid is not None else f"task~group{g}"
        # inject-step sequence is group-shared; take the group's first member
        inj_seq = ""
        if di is not None:
            inj_seq = "  inject-steps=" + _fmt_seq(di[g * group_size], intish=True)
        lines.append(f"  ┌─ Group {g} | {task_label} | SR={int(succ.sum())}/{group_size}={sr:.2f}{inj_seq}")
        for r in range(group_size):
            idx = g * group_size + r
            ok = "✓succ" if bool(per_traj_success[idx]) else "✗fail"
            nseq = ""
            if nn is not None:
                nseq = " noise‖·‖seq=" + _fmt_seq(nn[idx])
            lines.append(
                f"  │   rollout {r} | reward={float(per_traj_reward[idx]):.2f} | {ok}{nseq}"
            )
        lines.append("  └" + "─" * 40)
    lines.append("")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    return path
