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
# the sampled x_next at di, and its std (from the rollout). Estimate (x-mean)/std magnitude:
with torch.no_grad():
    logp,_,_ = R.recompute_logprob(m, img, ctx, cmask, chains, di, H, ni, 0.1, "flow_sde", train_video_expert=False)
base = logp[:, :nchunks, :].float()
print(f"control (identical): ratio={torch.exp((base-prev).sum((1,2))).mean():.4f}")
# Now simulate a cross-process mean shift: perturb x_next within the chain by relative eps, recompute
for eps in [1e-3, 4e-3, 8e-3]:  # 8e-3 ~ bf16 mantissa spacing near 1
    chains2 = chains.clone()
    x_next = chains2[ar, di+1]
    chains2[ar, di+1] = x_next + eps*torch.randn_like(x_next)*x_next.abs().clamp_min(0.1)
    with torch.no_grad():
        lp2,_,_ = R.recompute_logprob(m, img, ctx, cmask, chains2, di, H, ni, 0.1, "flow_sde", train_video_expert=False)
    r = torch.exp((lp2[:, :nchunks,:].float()-prev).sum((1,2)))
    print(f"eps={eps:.0e} (bf16-scale x shift): ratio mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
