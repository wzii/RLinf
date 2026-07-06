# Versatile VLA (cloudrobovla / VersaVLA) 移植到 RLinf 方案

> 目标:把 `versatile_vla`(包名 `cloudrobovla`,核心模型 `VersaVLA0Model`)的**模型推理、训练、LIBERO 评测**接入 RLinf 的分布式 RL 栈。
>
> 移植以 RLinf 已有的 **openpi**(同样是 flow-matching VLA)和 **libero** env 为对照模板,但二者都需要改动。
>
> 深度:A(推理 + LIBERO 评测)→ B(SFT/DAgger)→ C(RL: PPO/GRPO),分阶段实现。
> 变体:先做 base `VersaVLA0Model`,其它变体见末尾「从 ckpt 识别变体」。

---

## 0. 一句话架构对照

| 维度 | versatile_vla (`cloudrobovla`) | RLinf |
|---|---|---|
| VLM backbone | **Qwen3-VL**(`Qwen3VLForConditionalGeneration`) | openpi=PaliGemma、openvla=Prismatic、gr00t/starvla=各自。**无 Qwen3-VL 系** |
| Action head | flow-matching DiT/MMDiT,输出连续动作 chunk | openpi=pi0 flow-matching(**最接近的模板**) |
| 推理接口 | `policy.infer(obs)->{"actions":[1,chunk,dim]}` | `BasePolicy.predict_action_batch(env_obs)->(actions, result)` |
| 训练 | HF `transformers.Trainer` + DeepSpeed,loss=VLM_CE+FM_MSE | RLinf 自己的 `runners/embodiment` (RL) 或 `runners/sft` (SFT) |
| LIBERO env | 单进程 `OffScreenRenderEnv` + websocket policy server | 向量化 `LiberoEnv`(`rlinf/envs/libero/`),已自带 step/reset/chunk_step/eval 统计 |

**结论**:LIBERO env 直接复用 RLinf 的;模型本体(`VersaVLA0Model`)直接 `from_pretrained` 加载(标准 HF);核心工作是把 versatile_vla 的 policy 包成 RLinf 的 `BasePolicy`,并注册进 RLinf 的工厂/action 后处理分发。**不要**把 versatile_vla 自带的 `rl/rlt`、`rl/expo_ft`、websocket server、`eval_libero_pi.py`、`third_party/libero` 搬过来。

---

## 1. 已确认的接口契约(两边对齐结果)

### 1.1 RLinf BasePolicy 必须实现 (`rlinf/models/embodiment/base_policy.py:32`)
```python
class BasePolicy(ABC):
    def default_forward(self, **kwargs)            # 训练:返回 {"logprobs","entropy","values"}
    def predict_action_batch(self, env_obs=None, **kwargs) -> (actions, result)
        # actions: np.ndarray/tensor [B, chunk, dim]
        # result: {"prev_logprobs","prev_values","forward_inputs"}
    def sft_forward(self, data, ...)               # 可选:SFT/DAgger
```
调用点:
- rollout:`rlinf/workers/rollout/hf/huggingface_worker.py:505` 调 `model.predict_action_batch(env_obs=env_obs, **kwargs)`
- actor 训练:actor worker 调 `model.default_forward(compute_logprobs=...)`

### 1.2 RLinf LiberoEnv 产出的 env_obs 格式 (`rlinf/envs/libero/libero_env.py:641` `_wrap_obs`)
```python
{
  "main_images":   tensor [B, H, W, C],   # 256x256, 已做 [::-1,::-1] 翻转
  "wrist_images":  tensor [B, H, W, C],   # 256x256, 已翻转
  "states":        tensor [B, 8],         # eef_pos(3)+quat2axisangle(3)+gripper_qpos(2)
  "task_descriptions": list[str],
}
```
> ✅ **关键风险已排除**:RLinf 的 `get_libero_image`/`get_libero_wrist_image`(`rlinf/envs/libero/utils.py:79,94`)**已经做了** `[::-1,::-1]` 翻转,与 versatile_vla 的 `eval_libero_pi.py:220` 完全一致。`quat2axisangle` 也一致。**obs 桥接层不用补翻转**。

### 1.3 versatile_vla 期望的输入 (policy `infer` 的 `item`)
```python
{
  "left_third_image": np.uint8 [256,256,3],   # agentview
  "left_wrist_image": np.uint8 [256,256,3],   # wrist
  "state": np.float [8],
  "repo_id": str,                              # 如 "libero_all_pi_v3.0",用于查 stats
  "prompt": str,                               # task description
}
```
→ 桥接:RLinf env_obs → versatile_vla item,见 §3.2。

### 1.4 versatile_vla 推理内部路径 (`versa_vla_base.py:1036` `infer`)
```
build_model_input(item)          # 用 VersaVLA0RequestPreprocessor + UnifiedDatasetCollator
→ _select_actions(request_item)  # versa_vla_model.py:1064 _sample_actions
    1 次 VLM forward 拿 last_hidden_state (Qwen3-VL,用 .model.forward(inputs_embeds=),非 .generate())
    5 步 Euler 去噪,每步 action_head.predict_velocity(vlm_latents, state, x_t, t, masks)
→ _postprocess_actions          # versa_vla_base.py:982,反归一化 (x+1)/2*(max-min)+min
→ _project_response_actions     # 取 original_action_dim(通常 7)
返回 {"actions": np.float32 [1, chunk, 7], "state":..., "policy_time_use":...}
```

---

## 2. 需要改动的文件清单(对照 openpi + libero)

| # | 文件 | 改动 | 必需阶段 |
|---|---|---|---|
| 1 | `rlinf/config.py` | 注册 `SupportedModel.VERSA_VLA` + 加入 `EMBODIED_MODEL` | A |
| 2 | `rlinf/models/__init__.py` | 注册 versa_vla 工厂 `_build_versa_vla` | A |
| 3 | `rlinf/models/embodiment/versatile_vla/` (新建) | policy 实现(见 §3) | A |
| 4 | `rlinf/workers/rollout/hf/huggingface_worker.py:461,478` | 把 VERSA_VLA 加入两个 `SupportedModel in [...]` 列表(决定 kwargs) | A |
| 5 | `rlinf/envs/action_utils.py` | libero/isaaclab/polaris/calvin/metaworld 的 action 后处理加 versa_vla 分支(见 §4) | A |
| 6 | `examples/embodiment/config/model/versa_vla.yaml` (新建) | 模型配置 | A |
| 7 | `examples/embodiment/config/libero_spatial_ppo_versa_vla.yaml` (新建) | LIBERO+versa_vla 实验配置 | A |
| 8 | `rlinf/models/embodiment/versatile_vla/sft_forward` | SFT loss(CE+FM_MSE) | B |
| 9 | `rlinf/models/embodiment/versatile_vla/default_forward` 完整版 | flow-matching logprob + value head | C |
| 10 | `rlinf/workers/sft/fsdp_vla_sft_worker.py:33,147` | versa_vla SFT worker 分支 | B |
| 11 | FSDP wrap 配置 / actor worker | versa_vla 模块结构识别 | B/C |
| 12 | 依赖:`requirements/install.sh` 或 pyproject | 加 `cloudrobovla` 为可选依赖 | A |

> **每处「很可能需要的改动」**:openpi 的工厂靠 `paligemma_with_expert` 加载和 LoRA tag(`rlinf/models/__init__.py:297`),versa_vla 是 Qwen3-VL + 独立 action head,结构不同,**LoRA/wrap 逻辑要重写**(不能照抄 openpi 的 `module_to_lora = model.paligemma_with_expert.paligemma`)。action 后处理 openpi 对 libero **不做什么**(见 `action_utils.py:72` 的 prepare_actions_for_libero 里 openpi 不在列表),versa_vla 大概率也不需要(它的 gripper 在反归一化时已处理),但**需验证 gripper 符号约定**。

---

## 3. 核心实现:`rlinf/models/embodiment/versatile_vla/`

### 3.1 目录结构(照搬 openpi)
```
versatile_vla/
  __init__.py                  # get_model(cfg, torch_dtype) 工厂
  versa_vla_action_model.py    # class VersaVLAForRLActionPrediction(VersaVLA0Model, BasePolicy)
  policies/
    libero_policy.py           # env_obs ↔ versatile_vla item 桥接
```

### 3.2 `policies/libero_policy.py`(桥接,照搬 openpi/policies/libero_policy.py 的 LiberoInputs/LiberoOutputs 思路)
```python
def env_obs_to_versa_item(env_obs):
    # RLinf env_obs → versatile_vla infer 的 item
    main = env_obs["main_images"].permute(0,3,1,2) if needed  # 看 dtype/布局
    return {
        "left_third_image": to_uint8_hwc(main),
        "left_wrist_image": to_uint8_hwc(env_obs["wrist_images"]),
        "state": env_obs["states"],
        "repo_id": cfg.repo_id,                 # "libero_all_pi_v3.0"
        "prompt": env_obs["task_descriptions"], # list[str]
    }

def versa_actions_to_chunk(actions):  # [1,chunk,7] → RLinf 要的 [B,chunk,dim]
    ...
```
> ⚠️ 改动点:openpi 的 `LiberoInputs` 产出的 key 是 `observation/image`、`observation/wrist_image`、`prompt`,且 pad 到 224、补 zero wrist。versa_vla 的 key 是 `left_third_image`/`left_wrist_image`/`state`/`repo_id`/`prompt`,尺寸 256,**不补零**。两边预处理管线不同,不能共用,要按 versa_vla 的 `VisionPreprocessor`(内部 `smart_resize` factor=28)走。

### 3.3 `versa_vla_action_model.py`(三个方法)

**A 阶段 — `predict_action_batch`**
```python
class VersaVLAForRLActionPrediction(VersaVLA0Model, BasePolicy):
    def __init__(self, config, ..., repo_id, action_dim=7, num_action_chunks=10, add_value_head=False):
        super().__init__(config)
        # 复用 cloudrobovla 的预处理/反归一化工具:
        #   VersaVLA0RequestPreprocessor, VersaVLA0Normalizer, VersaVLA0DatasetProcessor
        # 这些在 LIBEROVersaVLAAdapter(VersaVLA0Policy) 里已组装好
        ...

    @torch.no_grad()
    def predict_action_batch(self, env_obs=None, do_sample=True, **kwargs):
        item = env_obs_to_versa_item(env_obs)              # §3.2
        # 复用 BaseVersaVLAPolicy 的 build_model_input + _select_actions + _postprocess_actions
        # (直接委托给一个内部 VersaVLA0Policy 实例,或把那几个方法搬进来)
        actions = self._policy.infer(item)["actions"]      # [1,chunk,7]
        chunk_actions = torch.as_tensor(actions.reshape(-1, num_action_chunks, action_dim))
        forward_inputs = {...}  # 供 default_forward 重算 logprob 用
        return chunk_actions, {"prev_logprobs":..., "prev_values":..., "forward_inputs": forward_inputs}
```
> 关键:`VersaVLA0Policy`(`versa_vla_policy.py:654`)已经把 `_load_model`/`load_preprocess_tools`/`infer` 全包好了。**最省事的做法是组合而非继承**——在 `VersaVLAForRLActionPrediction` 里持有一个 `LIBEROVersaVLAAdapter` 实例做推理,模型权重共享。但要满足 RLinf 的 FSDP/权重同步,最终得让 action head + vlm_backbone 是 `nn.Module` 的直接子模块(见 §3.4)。

**C 阶段 — `default_forward`(flow-matching logprob)**
- 对照 `openpi_action_model.py:674` 的 joint logprob:每个 denoise step 采一次,ODE-SDE mix sampling。
- versa_vla 的原语:`prepare_flow_matching_inputs`(`flow_matching_utils.py:9`,产 `x_t,t_discretized,u_t`)+ `action_head.predict_velocity(vlm_latents, state, x_t, t, masks)`(`flow_matching_action_head.py:840`)。
- 难点:把"给定一条轨迹算它在 flow-matching 下的 logprob"实现出来。openpi 已有参考实现,versa_vla 的 FM action head 接口更简单(predict_velocity 直接给速度场),需要自己包一层概率密度 / score。
- value head:照搬 `openvla_action_model.py:493` 的 `ValueHead`,挂在 VLM last_hidden_state 上。

**B 阶段 — `sft_forward`(CE + FM_MSE)**
- 直接复用 versatile_vla 原始训练 loss(`versa_vla_trainer.py:636` `_build_total_loss`):
  - VLM loss = next-token CE(ignore_index=-100),`versa_vla_model.py:680`
  - action loss = masked MSE(`action_pred` vs `u_t=noise-actions`),按 `mask.sum()` 归一化(**不是 element 数,要原样照搬否则 loss scale 不同**)
- flow-matching target 由 trainer 预算(`_prepare_action_diffusion_inputs`,`versa_vla_trainer.py:552`),注入 `action_samples/action_timesteps/action_targets`。

### 3.4 `__init__.py` 工厂(照搬 openpi/__init__.py 的结构,但加载逻辑不同)
```python
def get_model(cfg: DictConfig, torch_dtype=None):
    from cloudrobovla.models.vlas.versa_vla import VersaVLA0Model, VersaVLA0ModelConfig
    # 1. from_pretrained 加载(标准 HF,model_type="VersaVLA0")
    model = VersaVLA0Model.from_pretrained(cfg.model_path, torch_dtype=torch_dtype)
    # 2. 装上 RL 用的辅助(VersaVLAForRLActionPrediction 包装 + value head + processor/stats)
    # 3. 加载 merged stats(LIBERO-ALL-PI_3.0_merged_stats.*.json)用于反归一化
    return model
```
> ⚠️ 改动点:openpi 工厂(`openpi/__init__.py:22`)要处理 safetensors/FSDP checkpoint 多路径、`paligemma_with_expert.to_bf16`、`setup_wrappers(transforms)`。versa_vla 是标准 `from_pretrained`,**加载更简单**;但它的 transforms(`VersaVLA0RequestPreprocessor`/`VersaVLA0Normalizer`)不是 openpi 的 `transforms.Group` 模式,**不能照搬 setup_wrappers**,要在 policy 层自己组装(参考 `VersaVLA0Policy.load_preprocess_tools`,`versa_vla_base.py:199`)。

---

## 4. action 后处理改动 (`rlinf/envs/action_utils.py`)

`prepare_actions_for_libero`(`:66`):openpi 不在改 gripper 的列表里(它直接输出已处理好的)。versa_vla 的 gripper 在 `_postprocess_actions` 反归一化时已处理(min_max + 可选 `SemanticBinarySliceNormalizer` 阈值),**大概率也不需要额外改**。但 LIBERO 的 `LIBERO_DUMMY_ACTION=[0,0,0,0,0,0,-1]`(gripper=-1 开)和 versa_vla 输出的 gripper 符号约定**必须验证一致**(versa_vla `target_action_space="delta"`,gripper 在 delta 空间)。

→ 起步:把 VERSA_VLA 加进 `prepare_actions_for_libero` 的"不改"路径(即不加进那个 OPENVLA 列表),然后跑 eval 看成功率为零时优先查这里。

---

## 5. 分阶段落地顺序

### 阶段 A:推理 + LIBERO 评测(最小可验证)
1. `config.py` 注册 `SupportedModel.VERSA_VLA = SupportedModel.register("versa_vla", force=True)`,加进 `EMBODIED_MODEL`(`:117` 的 set)
2. `models/__init__.py` 加 `_build_versa_vla` + `register_model(..., category="embodied")`
3. 新建 `rlinf/models/embodiment/versatile_vla/`,实现 `get_model` + `predict_action_batch`(组合 `LIBEROVersaVLAAdapter`)+ `policies/libero_policy.py` 桥接
4. `huggingface_worker.py:461,478` 两处列表加 VERSA_VLA
5. `action_utils.py` 加 versa_vla 分支(先"不改")
6. `config/model/versa_vla.yaml` + `config/libero_spatial_ppo_versa_vla.yaml`(可先 `only_eval: True`)
7. **验证**:用 RLinf eval runner 在 libero_spatial 跑 50 trials/task,对照 versatile_vla 原生 `eval_libero_pi.py` 的成功率。成功率对齐 = 桥接正确。成功率 0 → 先查图像 dtype/布局、gripper 符号、repo_id/stats 文件路径。

### 阶段 B:SFT/DAgger
8. 实现 `sft_forward`(复用 versatile_vla 的 CE+FM_MSE,原样搬 masked-MSE 归一化)
9. 数据集:对接 LeRobot v3(versatile_vla 的 `UnifiedLeRobotDataset`,RLinf `data/` 也支持 LeRobot,需对齐 schema/mixture registry)
10. `fsdp_vla_sft_worker.py:33,147` 加 versa_vla 分支
11. FSDP wrap:识别 `vlm_backbone`/`action_head` 子模块

### 阶段 C:RL (PPO/GRPO)
12. `default_forward` 完整版:flow-matching logprob(对照 openpi `:674` joint logprob)
13. value head 接入
14. 用 `examples/embodiment/config/libero_spatial_ppo_openpi_pi05.yaml` 为模板写 ppo config
15. 验证:rollout→reward→advantage→actor update 全链路

---

## 6. 依赖与运行环境

- **`cloudrobovla` 作为可选依赖**装进 RLinf 环境(`pip install -e /home/ziyi/versatile_vla`),新 policy `import cloudrobovla...` 复用其 model/preprocessor/stats。**不要复制代码进 RLinf**。
- **LIBERO env 用 RLinf 的**(`rlinf/envs/libero/`),不引 versatile_vla 的 `third_party/libero` 和它的双 venv(server venv + `evaluation/libero/.venv`)。RLinf 自己的 libero import 已支持 standard/pro/plus。
- 运行:`MUJOCO_GL=egl bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_versa_vla`

---

## 7. 关键风险与验证点

| 风险 | 验证方法 |
|---|---|
| obs 图像翻转 | ✅ 已确认两边都做 `[::-1,::-1]`,无需补 |
| 图像 dtype/布局 (HWC vs CHW, uint8 vs float) | eval 成功率为 0 时首查;RLinf env_obs 是 tensor[H,W,C],versa_vla 要 uint8 HWC |
| gripper 符号约定 | 对照 `LIBERO_DUMMY_ACTION` 和 versa_vla delta 空间输出 |
| stats 文件路径/repo_id | `_merged_stats.*.json` 要能被 `VersaVLA0Normalizer` 找到 |
| FSDP wrap 识别 Qwen3-VL 子模块 | 阶段 B/C,可能要写自定义 wrap policy |
| flow-matching logprob 正确性 | 阶段 C,对照 openpi 实现逐 step 核验 |

---

## 8. 从 ckpt 识别是哪个变体

每个 versa_vla 变体的 `config.json` 里 `model_type` 字段就是变体名。**看 ckpt 目录下的 `config.json`**:

```bash
# 假设 ckpt 在 /path/to/ckpt/
python -c "import json; print(json.load(open('/path/to/ckpt/config.json'))['model_type'])"
```

| `model_type` 值 | 变体 | policy 类 | 对应 RLinf 要实现的 |
|---|---|---|---|
| `VersaVLA0` | **base**(本次先做) | `VersaVLA0Policy` / `LIBEROVersaVLAAdapter` | `VersaVLAForRLActionPrediction` |
| `VersaVLA3D` | 3D 点云 | `VersaVLA3DPolicy` | 需额外 3D mix 输入 |
| `VersaVLAForce` | 力反馈 | `VersaVLAForcePolicy` | 需 force-torque 历史 |
| `VersaVLADepth` | 深度辅助 | (depth) | 需 depth head |
| `VersaVLADANext` | DA-Next SE(3) | (da_next) | 需 DA-Next backbone |
| `VersaVLAAdaptive` | 自适应 Qwen3 expert | `VersaVLAAdaptivePolicy` | action expert 不同 |
| `VersaVLALayerwise` | layerwise Qwen3 | `VersaVLALayerwisePolicy` | action expert 不同 |
| `VersaVLADualAdaptive` | 双流自适应 | `VersaVLADualAdaptivePolicy` | |
| `VersaVLATriStreamAdaptive` | 三流 | `VersaVLATriStreamAdaptivePolicy` | |
| `VersaVLAMoT` | Mixture-of-Transformers | `VersaVLAMoTPolicy` | |
| (`VersaVLAMoH`/`Mem` 系列) | 不是 PreTrainedModel,是 runtime 包装 | `VersaVLAMoHPolicy`/`MemPolicy` | 需特殊构造 |

**base 模型内部还有 action head 大小区分**(`config.json` 的 `action_expert_variant` 字段):
- `fm_dit_160m` / `fm_dit_340m`(默认)/ `fm_dit_590m` — flow-matching DiT
- `mmdit_180m/220m/340m/490m` — MMDiT
- `layerwise_qwen3_*` / `adaptive_qwen3_*` — 这两个会让 `model_type` 变成对应的 Layerwise/Adaptive

所以判断流程:
1. 看 `config.json` 的 `model_type` → 定变体大类
2. 若是 `VersaVLA0`,再看 `action_expert_variant` → 定 action head 类型/大小
3. 若是 `VersaVLAForce`/`3D`/`Depth`/`DANext` → base 之上的扩展,需要额外输入/head
4. 若是 `MoH`/`Mem` 系列(无标准 `model_type`,是 runtime 包装)→ 需特殊处理,建议最后做

**本次只先做 `VersaVLA0` + `fm_dit_*` / `mmdit_*`**(base 的两种 action head,都走 `predict_velocity` 接口,实现一致)。
