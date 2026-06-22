import torch, numpy as np
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
B=8; H=policy.infer_cfg.action_horizon; 
img = torch.rand(B,3,224,224,device="cuda",dtype=torch.bfloat16)*2-1
# build a context via the model's text path is complex; use zeros context of right shape
ctx = torch.zeros(B,128,4096,device='cuda',dtype=torch.bfloat16)
cmask = torch.ones(B,128,device='cuda',dtype=torch.bool)
ni=cfg.num_inference_steps
with torch.no_grad():
    _, info = R.flow_sde_rollout(m, img, ctx, cmask, H, ni, 0.1, "flow_sde", deterministic=False)
chains=info["chains"]; di=info["denoise_inds"]
nchunks=cfg.num_action_chunks
ar=torch.arange(B,device='cuda')
prev = info["logp_per_step"][ar,di][:, :nchunks, :].float()   # rollout logp at sampled step
# recompute with the SAME model + inputs + chains + denoise_inds (no params changed)
with torch.no_grad():
    logp,ent,pooled = R.recompute_logprob(m, img, ctx, cmask, chains, di, H, ni, 0.1, "flow_sde", train_video_expert=False)
recomp = logp[:, :nchunks, :].float()
diff = (recomp - prev)
ratio = torch.exp(diff.sum(dim=(1,2)))   # chunk-level ratio like the loss
print("=== SINGLE-PROCESS rollout->recompute consistency ===")
print(f"per-element logp diff: mean={diff.mean().item():.4e} std={diff.std().item():.4e} max|.|={diff.abs().max().item():.4e}")
print(f"chunk-level ratio exp(sum diff): mean={ratio.mean().item():.4f} min={ratio.min().item():.4f} max={ratio.max().item():.4f}")
print("(ratio should be ~1.0 if rollout & recompute are consistent)")
