import re, sys
log=sys.argv[1]
raw=open(log,encoding="utf-8",errors="ignore").read().replace("\r","\n")
# The eval table starts at a line containing "Evaluation" (boxed header). Grab the
# success metrics in the window after each Evaluation header until the next box edge.
evals=[]
idxs=[m.start() for m in re.finditer(r"Evaluation\s*[─-]*\s*\n", raw)] or [m.start() for m in re.finditer(r"Evaluation", raw)]
for i in idxs:
    win=raw[i:i+1500]
    if "num_trajectories=100" not in win: continue
    so=re.search(r"success_once=([0-9.]+)", win)
    ae=re.search(r"success_at_end=([0-9.]+)", win)
    rt=re.search(r"return=([0-9.]+)", win)
    if so: evals.append((so.group(1), ae.group(1) if ae else "?", rt.group(1) if rt else "?"))
# de-dup consecutive identical
seen=[]; 
for e in evals:
    if not seen or seen[-1]!=e: seen.append(e)
print(f"eval trend ({len(seen)} evals, 100 ep each):")
for k,(so,ae,rt) in enumerate(seen):
    print(f"  eval #{k+1} (step {10*(k+1)}): success_once={so}  success_at_end={ae}  return={rt}")
print("baseline SR_0 (released ckpt) = 0.904 success_once / 0.878 success_at_end")
