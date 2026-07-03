#!/usr/bin/env python
"""Dump eval/* scalars (SR + per-task) from a run's tensorboard event files.
Usage: python extract_sr.py <run_dir_or_tb_dir> [<run_dir2> ...]
Prints a JSON blob per run: {overall_success, per_task: {...}, all_tags: {...}}."""
import sys, os, json, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_event_dirs(root):
    dirs = set()
    for p in glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True):
        dirs.add(os.path.dirname(p))
    return sorted(dirs)

def load_run(root):
    tags = {}
    for d in find_event_dirs(root):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags().get("scalars", []):
            evs = ea.Scalars(t)
            # take the LAST recorded value (eval_at_start logs at step -1; only one point)
            tags[t] = evs[-1].value
    return tags

for root in sys.argv[1:]:
    tags = load_run(root)
    evaltags = {k: v for k, v in tags.items() if k.lower().startswith("eval")}
    # heuristics for overall success tag
    overall = None
    for key in ("eval/success_once", "eval/success", "eval/success_rate",
                "eval/mean_success", "eval/successes", "eval/sr", "eval/return"):
        if key in tags:
            overall = tags[key]; break
    out = {
        "run": root,
        "overall_success": overall,
        "eval_tags": evaltags,
        "n_tags": len(tags),
    }
    print(json.dumps(out, indent=2, default=float))
