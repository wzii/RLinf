import torch, math
from omegaconf import OmegaConf
from rlinf.models.embodiment.fastwam import get_model
import rlinf.models.embodiment.fastwam.fastwam_rl as R
cfg = OmegaConf.create({
  "model_type":"fastwam","precision":"bf16","is_lora":False,
  "model_path":"/workspace/checkpoints/fastwam/libero_uncond_2cam224.pt",
  "checkpoint_path":"/workspace/checkpoints/fastwam/libero_uncond_2cam224.pt",
  "dataset_stats_path":"/workspace/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json",
  "action_dim":7,"num_action_chunks":10,"action_horizon":32,"num_inference_steps":10,
  "binarize_gripper":True,"text_cfg_scale":1.0,"negative_prompt":"","sigma_shift":None,"seed":None,
  "rand_device":"cpu","tiled":False,"add_value_head":False,
  "rl":{"enabled":True,"noise_level":0.1,"noise_method":"flow_sde","freeze_video_expert":True,"train_video_expert":False,"batched":True},
  "fastwam":{"config_dir":None,"config_name":"sim_libero","overrides":["model.redirect_common_files=false"]}})
policy = get_model(cfg, torch_dtype=torch.bfloat16).cuda().eval()
m = policy.model
torch.manual_seed(0)
B=16; H=policy.infer_cfg.action_horizon
img = (torch.rand(B,3,224,224,device="cuda",dtype=torch.bfloat16)*2-1)
ctx = torch.zeros(B,128,4096,device="cuda",dtype=torch.bfloat16)
cmask = torch.ones(B,128,device="cuda",dtype=torch.bool)
ni=cfg.num_inference_steps; nchunks=cfg.num_action_chunks; ar=torch.arange(B,device="cuda")
with torch.no_grad():
    _, info = R.flow_sde_rollout(m, img, ctx, cmask, H, ni, 0.1, "flow_sde", deterministic=False)
chains=info["chains"]; di=info["denoise_inds"]
prev = info["logp_per_step"][ar,di][:, :nchunks, :].float()

def ratio_for(recompute_dtype):
    # temporarily view the model in a given dtype by toggling torch_dtype the forward casts to
    old = m.torch_dtype
    try:
        m.torch_dtype = recompute_dtype
        if recompute_dtype==torch.float32:
            m_fp = m.float()  # cast params to fp32
        with torch.no_grad():
            logp,_,_ = R.recompute_logprob(m, img, ctx, cmask, chains, di, H, ni, 0.1, "flow_sde", train_video_expert=False)
    finally:
        m.torch_dtype = old
        if recompute_dtype==torch.float32:
            m.to(torch.bfloat16)
    r = torch.exp((logp[:, :nchunks, :].float() - prev).sum(dim=(1,2)))
    return r

r_bf16 = ratio_for(torch.bfloat16)
print(f"recompute bf16 (same as rollout): ratio mean={r_bf16.mean():.4f} min={r_bf16.min():.4f} max={r_bf16.max():.4f}")
# Now perturb rollout vs recompute precision: rollout was bf16, recompute in fp32 (simulates cross-process compute diff)
r_fp32 = ratio_for(torch.float32)
print(f"recompute fp32 (rollout was bf16): ratio mean={r_fp32.mean():.4f} min={r_fp32.min():.4f} max={r_fp32.max():.4f}")
print(f"  -> a bf16<->fp32 mean shift makes ratio swing this much (this is the cross-process effect)")
