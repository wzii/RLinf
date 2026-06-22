import subprocess,time
mx=[0,0,0,0]
while True:
    try:
        out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().split()
        for i,v in enumerate(out[:4]): mx[i]=max(mx[i],int(v))
        open("/workspace/results/peak_run.txt","w").write(f"peak MiB: {mx} max={max(mx)} ({max(mx)/1024:.1f} GB) t={time.strftime('%T')}\n")
    except Exception: pass
    time.sleep(3)
