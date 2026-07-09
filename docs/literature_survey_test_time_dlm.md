# 纯文本 DLM Test-Time 跨步信息利用：详细文献调研

> **调研起点**：[1] Han et al., *Soft-Masked Diffusion Language Models*, arXiv:2510.17206, ICLR 2026  
> **范围**：纯文本 Masked DLM 的 inference / test-time 方法（2025–2026），聚焦跨步信息保留与 unmask 调度  
> **更新**：2026-07-08（含 ICML 2026 专节）

---

## 目录

1. [背景：MDLM 标准 Inference 及其信息瓶颈](#1-背景mdlm-标准-inference-及其信息瓶颈)
2. [问题形式化与研究维度](#2-问题形式化与研究维度)
3. [Survey 综述论文](#3-survey-综述论文)
4. [基座模型与 Decoding 基础设施](#4-基座模型与-decoding-基础设施)
5. [纵向机制：跨步信息保留与传递（详细）](#5-纵向机制跨步信息保留与传递详细)
6. [横向机制：Unmask 顺序与优先级调度（详细）](#6-横向机制unmask-顺序与优先级调度详细)
7. [Test-Time Scaling：搜索、集成与 Step-Level 组合（详细）](#7-test-time-scaling搜索集成与-step-level-组合详细)
8. [Latent / 表示层 Test-Time 方法](#8-latent--表示层-test-time-方法)
9. [训练侧方法（定义 Inference 上下界）](#9-训练侧方法定义-inference-上下界)
10. [机制对比与文献演化时间线](#10-机制对比与文献演化时间线)
11. [统一框架与研究空白](#11-统一框架与研究空白)
12. [推荐实验路径（LLaDA-8B）](#12-推荐实验路径llada-8b)
13. [完整论文索引与参考文献](#13-完整论文索引与参考文献)
14. [ICML 2026 录取 DLM 论文专节（扩展）](#14-icml-2026-录取-dlm-论文专节扩展)

---

## 1. 背景：MDLM 标准 Inference 及其信息瓶颈

### 1.1 Masked DLM 的基本生成流程

Masked Diffusion Language Model（MDLM）将文本生成建模为离散扩散的 reverse process：

**Forward（训练时）**：以概率 `τ` 将 token 替换为 `[MASK]`：
```
q(x_τ | x_0) = Cat(x_τ; α_τ x_0 + (1-α_τ) m)
```
其中 `α_τ` 为 noise schedule（如 linear: `α_t = 1 - t/T`），`m` 为 mask token。

**Reverse（推理时）**：从全 mask 序列 `x_T = (M,...,M)` 出发，迭代 T 步：

```
Step t:
  1. Forward: p_θ(x_{t-1} | x̄_t) = g_θ(x̄_t)     # 双向 Transformer，预测所有 masked 位置
  2. Sample:  x_{t-1} ~ p_θ(· | x̄_t)             # 对每个 masked 位置采样/argmax
  3. Remask:  x̄_{t-1} = γ(x_{t-1}, t)            # 按策略 remask 一部分 token，保留其余
```

LLaDA 默认使用 **Low-Confidence Remasking (LCR)**：每步 remask `⌊L/T⌋` 个 **置信度最低** 的 token（即保留置信度较高的预测）。

### 1.2 两类核心信息浪费

| 类型 | 描述 | 后果 |
|------|------|------|
| **纵向浪费** | 第 t 步对仍-masked 位置 i 的 logits `ℓ_{t,i}` 在 t+1 步被丢弃 | 模型每步"从零开始"预测同一位置，无法利用历史预测 |
| **横向短视** | unmask 决策仅看当前步 `conf(ℓ_{t,i})`，不看历史轨迹 | early 高 conf 但 context-brittle 的 token 被过早 commit |
| **Context Rigidity** | 一旦 unmask，token 被 freeze，无法修正 | early wrong commit 传播为 cascade error（CORE 的核心观察） |

### 1.3 与 Autoregressive 的本质差异

| | AR (GPT) | MDLM (LLaDA) |
|---|----------|--------------|
| 生成方向 | 严格左→右 | 任意 order，可并行 |
| 每步输出 | 1 token | 多个 token（block） |
| 中间状态 | KV cache（隐式历史） | 每步 logits 通常丢弃 |
| 修正能力 | 无法 revise 已生成 token | remasking 可恢复 iterative refinement |
| Test-time scaling | CoT / Best-of-N / Tree Search | 搜索 unmask trajectory / ensemble schedule |

Svete & Sabharwal (ICLR 2026 Oral) 从理论上证明：MDM 在**可并行问题**（如 regular language）上比 CoT-augmented Transformer 更高效，unmask order 是计算能力的一部分而非实现细节。

---

## 2. 问题形式化与研究维度

### 2.1 符号

- 序列长度 `L`，denoising 步数 `T`
- 位置 `i ∈ {1,...,L}`，step `t ∈ {T,...,0}`
- `M_{t,i} ∈ {0,1}`：1 = 仍 masked
- `ℓ_{t,i} ∈ ℝ^{|V|}`：第 t 步位置 i 的 logits
- `S_{t,i}`：跨步状态（embedding / confidence / history 等）

### 2.2 目标 Inference

```
Step t:
  ℓ_t ← Model(x̄_t)
  S_{t,i} ← Update(S_{t-1,i}, ℓ_{t,i})     # 纵向：保留跨步信息
  U_t ← Schedule({S_{t,i} : M_{t,i}=1})    # 横向：决定 unmask 集合
  x̄_{t-1} ← ApplyUnmask(x̄_t, U_t)
  Optionally: RemaskSet ← Revision({S_{t,j} : M_{t,j}=0})  # 修正已生成 token
```

### 2.3 两个正交维度

| 维度 | 核心问题 | 代表方法 |
|------|----------|----------|
| **纵向（Temporal）** | 第 n 步信息如何传递到第 n+1 步？ | Soft-Mask, MRP, RCR, ReMDM |
| **横向（Spatial）** | 哪些 token 优先生成/暂缓？ | LCR, SAS, SOAR, Unmask Policy |
| **修正（Revision）** | 已生成 token 何时 remask？ | ReMDM, CORE, RCR, RemeDi |
| **搜索（Search）** | 如何探索多条 trajectory？ | HEX, S³, UMF, MEDAL |

---

## 3. Survey 综述论文

### [S1] A Survey on Diffusion Language Models (arXiv:2508.10875, 2025)

- **覆盖**：continuous + discrete DLM 的训练、推理、加速
- **Inference 三目标**：(i) 生成质量（unmask/remask schedule）；(ii) 内容控制；(iii) 效率（KV cache、step distillation）
- **关键分类**：parallel decoding strategies、unmasking/remasking、caching
- **对本调研价值**：提供全局 taxonomy，确认 inference 是 DLM 研究的核心开放问题

### [S2] Discrete Diffusion in Large Language and Multimodal Models (arXiv:2506.13759, 2025)

- **核心观点**：dLLM inference 的根本问题是 **"每步 unmask 哪些、remask 哪些"**
- **效率瓶颈**：full attention mask 导致每步计算量大，需 AR 加速技术适配
- **对本调研价值**：明确 unmask/remask 策略是 discrete diffusion 区别于 AR 的核心设计空间

### [S3] Accelerating Inference in Diffusion LLMs: A Survey (TechRxiv, 2025)

- **分类**：intra-step 成本降低 vs inter-step 步数减少
- **Dynamic parallel decoding**：confidence threshold 决定 unmask 数量（Fast-dLLM 路线）
- **对本调研价值**：将 confidence-based unmask 定位为主流 baseline 类别

---

## 4. 基座模型与 Decoding 基础设施

### [B1] LLaDA: Large Language Diffusion Models (Nie et al., arXiv:2502.09992, 2025)

| 项目 | 详情 |
|------|------|
| **规模** | 8B（Base + Instruct），首个大规模 from-scratch MDLM |
| **架构** | 标准 Transformer + bidirectional attention |
| **训练** | 随机 mask ratio τ~U(0,1)，CE loss 仅在 masked 位置 |
| **Inference** | Block-wise semi-AR：block 内 parallel denoising，block 间 causal |
| **默认 remask** | LCR：每步 remask 低置信度 token |
| **ModelScope** | `GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct` |
| **意义** | 本项目的实验基座；大多数 2025–2026 test-time 方法在其上验证 |

**LLaDA Inference 伪代码**：
```
x̄_T ← [prompt; M, M, ..., M]           # 全 mask 生成区
for t = T downto 1:
    p ← g_θ(x̄_t)                        # 预测所有 masked 位置
    x_{t-1} ← sample/argmax(p)          # 临时 unmask 全部
    n_t ← ⌊(T-t)/T · L_gen⌋            # 本步保留的 token 数
    x̄_{t-1} ← remask_low_conf(x_{t-1}, n_t)  # LCR
return x_0
```

### [B2] Dream 7B (Ye et al., arXiv:2508.15487, 2025)

| 项目 | 详情 |
|------|------|
| **初始化** | 从 AR 预训练权重（Qwen 系）出发，保留 shift operation |
| **CART** | Context-Adaptive noise Rescheduling at Token-level：token 级动态 noise |
| **Inference** | 低 entropy token 优先 unmask（与 LLaDA LCR 类似但 metric 不同） |
| **意义** | 多数 heuristic / Soft-Mask / Prophet 等的第二验证模型 |

### [B3] Fast-dLLM v1 (Wu et al., arXiv:2025) & v2 (arXiv:2509.26328, 2025)

**v1 贡献**：
- 证明高 confidence 下 greedy parallel decoding 等价于 sequential decoding
- 引入 **confidence threshold τ**：unmask 所有 conf > τ 的 token（可变并行度）
- Prefix / Dual KV-Cache 设计

**v2 贡献**：
- 将 AR 模型（Qwen2.5）适配为 block diffusion dLLM（~1B token finetune）
- **Hierarchical caching**：block-level + sub-block cache
- 最高 **2.5×** 加速，精度不降

**关键参数**：`small_block_size=8`, `threshold=0.9`

**与跨步信息关系**：Fast-dLLM 解决的是**并行度/效率**，不是跨步信息保留；但其 confidence threshold 是 SOAR、Prophet 等方法的共同 baseline。

### [B4] ParallelBench (OpenReview 2026)

- **发现**：增加并行度（一次 unmask 更多 token）导致质量下降——"parallel generation curse"
- **Remasking 效果因任务而异**：ReMDM/RCR 在 Waiting Line 等任务上无效
- **CoT 缓解但不根本解决**：8× token 开销
- **对本调研价值**：警示 test-time 方法需在多种 parallel factor 下验证

---

## 5. 纵向机制：跨步信息保留与传递（详细）

---

### [1] Soft-Masked Diffusion Language Models ⭐ 逻辑起点

| 字段 | 内容 |
|------|------|
| **作者** | Hersche, Moor-Smith, Hofmann, Rahimi (IBM + ETH) |
| **Venue** | ICLR 2026 |
| **arXiv** | 2510.17206 |
| **代码** | https://github.com/IBM/soft-masked-diffusion-language-models |
| **类型** | 训练方法（continued pretrain / finetune），非纯 test-time |

#### 动机

标准 MDLM 对每个 mask 位置做 **binary 决策**：unmask 或保留 `[MASK]`。当保留 mask 时，模型对该位置的 top-k 预测信息被完全丢弃——这是跨步信息浪费的最直接体现。

#### 方法：Soft-Masking (SM)

对仍 masked 的位置 l，将 feedback token 从 one-hot 松弛为 simplex 上的分布：

```
x^l_{t-1} = sm(x̂^l_{t-1}, p^l_{t-1}) = 
    (1 - λ(p)) · m + λ(p) · Σ_{i∈top-k(p)} π_i · v_i    if x̂^l = m
    x̂^l_{t-1}                                              otherwise
```

其中：
- `p^l_{t-1}`：上一步模型输出的概率向量
- `π_i = p_i / Σ_{j∈top-k} p_j`：top-k 上的 renormalized 权重
- `λ(p) = ω_s · σ(ω_a · (-H(p)) - ω_b)`：基于 **负熵** 的动态 confidence 权重
- `ω_s, ω_a, ω_b`：3 个可学习参数；`λ ∈ [0,1)`

**关键设计**：
- 高 confidence → 高 λ → 更多 top-k 信息混入
- 低 confidence → 低 λ → 保留更多 `[MASK]` embedding
- 仅 3 个额外参数，可 parallelizable two-pass training

#### 训练（Algorithm 1 概要）

```
1. 采样 x_0,  corrupt → x_t
2. Pass 1 (no grad): p̃ ← g_θ̃(x_t)           # 近似上一步分布
3. Pass 2 (grad):    p ← g_θ(sm(x_t, p̃))    # soft-masked input
4. Loss: CE on masked positions
5. 同时更新 θ (backbone) 和 ω (SM params)
```

- 训练时以概率 `p_sm` 激活 SM，使模型适应 soft/hard 混合输入
- 时间步采样区间 `[b_l, b_h]` 缩窄以降低 gradient variance

#### 实验结果

**169M MDLM + OpenWebText（100k continued pretrain, k=3）**：
- Validation PPL：23.14 → 21.63（SM）vs 22.88（binary baseline）
- MAUVE（L=1024, 无约束生成）：SM + ReMDM 组合最优

**Dream-7B / Dream-Coder-7B（DoRA finetune, 33.5k steps）**：
- HumanEval / MBPP（含 + 版本）：high-throughput（少 NFE）设定下提升最显著
- SM 与 Fast-dLLM caching 兼容
- SM 与 ReMDM remasking **互补**（Table 1）

#### 消融发现

- `k=1` 最优于 coding；`k=3` 综合最优
- SM 在 decoding **前 20%** 步骤最有益（early step 信息传递最关键）
- `λ=0` 退化为 vanilla MDLM；`λ=1, k=1` 等价于 uniform DLM feedback

#### 局限

- **必须 finetune**：不能 plug-in 到已有 checkpoint 做纯 test-time
- 改变 embedding 输入，需要模型适应
- 未显式建模 unmask 优先级或 remask 修正

#### 对跨步信息问题的意义

SM 是第一个**系统性**解决 "retained mask 丢弃预测信息" 的工作。它确立了核心原则：**未 commit 的 token 应在 embedding 层保留 soft 信息**。后续 MRP（logit 残差）、RCR（running conf）从不同层面对同一原则做扩展。

---

### [2] Multi-Token Residual Prediction (MRP)

| 字段 | 内容 |
|------|------|
| **arXiv** | 2605.18817, 2026 |
| **类型** | 轻量 head 训练 + inference 两种模式 |

#### 核心观察

**相邻 denoising step 的 logit 分布高度相似**：
```
log p(x_0 | x_{t-1}) ≈ log p(x_0 | x_t) + Δlogits
```
其中 Δlogits 是 small residual，可用 cheap head 从 hidden states 预测。

#### 方法

- 冻结 backbone + LM head
- 训练 3-layer MRP module：`Δℓ ← MRP(hidden_t)`
- 修正 logits：`ℓ̂_{t+1} = ℓ_t + Δℓ`

#### 两种 Inference 模式

| 模式 | 机制 | 性质 |
|------|------|------|
| **Speculative** | MRP 提议 → backbone verify | 无损加速，SGLang 上最高 1.42× |
| **Direct + Remask** | 直接用修正 logits unmask → 撤销 over-eager reveal | 恢复质量：HumanEval +22.6, GSM8K +17.7 |

#### 实验（SDAR 1.7B/4B/8B）

- Static regime（高质量低吞吐）：speculative 1.42× 无损加速
- Dynamic regime（低质量高吞吐, τ=0.5）：direct 模式平均 +16 accuracy points

#### 对跨步信息的意义

MRP 从 **logit 层** 利用 step 间结构（残差可预测性），与 SM 的 embedding 层互补。Direct+remask 模式直接回应 "是否优先生成该 token"——低质量 reveal 可被撤销。

---

### [3] Running Confidence Remasking (RCR) — MDPO 论文

| 字段 | 内容 |
|------|------|
| **论文** | MDPO: Overcoming the Training-Inference Divide (He et al., arXiv:2508.13148) |
| **代码** | https://github.com/autonomousvision/mdpo |
| **RCR 类型** | **Training-free** decoding 策略 |

#### 动机：LCR 的 Freeze 问题

LLaDA 的 LCR 一旦 unmask 某 token，**永久 freeze**：
```
m^i_{t-1} = 1 - p_θ(x^i_{t-1} | x̄_t)    # 当前步低 conf → remask
```
若某 token 在 step t 碰巧落入 top-n_t（不被 remask），即使 conf 很低也被 freeze 至结束。

**Over-denoising 现象**：LLaDA 在 MATH-500 上 accuracy 18.2%，但中间 step 至少一次正确的高达 +9.8%——说明 early correct answer 被 later step "refine" 成 wrong。

#### RCR 方法

跟踪每个位置的 **running maximum confidence**：
```
m^i_{t-1} = 1 - max_{t'≥t} p_θ(x^{i}_{t'-1} | x̄_{t'})    # 历史最高 conf 的补
```
- running max conf **最低** 的位置 → 最可能被 remask
- 允许 early low-conf token 在 later step 被重新评估

#### 与 LCR 对比（Figure 2 案例）

LCR 在 step 1 freeze 了错误 token "problem" → 无法修正 → 最终错误  
RCR 跟踪 running conf → "problem" 后续 conf 始终低 → 被 remask → 修正

#### MDPO（同论文，训练侧）

- 将 denoising 建模为 sequential decision process
- GRPO + intermediate step rewards 对齐 training/inference schedule
- MATH500 +9.6%, Countdown +54.2%（同等 update 数）
- **RCR + MDPO 可叠加**

#### 实验平台

- LLaDA-8B-Base 为主要验证模型
- 数学推理：MATH-500, Countdown, GSM8K

#### 对跨步信息的意义

RCR 用 **标量 running confidence** 作为跨步信号，解决 "是否应 remask 已生成 token"。这是 training-free 的最直接可用的纵向机制之一。

---

### [4] ReMDM: Remasking Discrete Diffusion Models

| 字段 | 内容 |
|------|------|
| **作者** | Wang, Schiff, Sahoo, Kuleshov (Cornell) |
| **Venue** | NeurIPS 2025 |
| **arXiv** | 2503.00307 |
| **代码** | https://guanghanwang.com/remdm |
| **类型** | Training-free sampler |

#### 动机

Masked DLM 丢失 diffusion 的核心优势——**iterative refinement**。Token 一旦生成不可更新。

#### 理论基础

对已有 token `z_t ≠ m`，定义 remasking posterior：
```
q(z_s | z_t = x, x_0) = (1 - σ_t) · x + σ_t · m
```
- `σ_t = 0` → 标准 MDLM
- `σ_t > 0` → 以概率 σ_t remask 已解码 token
- **Theorem 3.1**：保持与 MDLM 相同的 marginal `q(z_t | x_0)`，可 plug-in 预训练权重
- 非 Markovian process（类似 DDIM 与 DDPM 的关系）

#### Remasking 策略变体

| 策略 | σ_t 定义 | 用途 |
|------|----------|------|
| **ReMDM-cap** | `min(η_cap, 1 - α_s/α_t)` | 限制 remask 概率上限 |
| **ReMDM-rescale** | `η_rescale × σ_max` | 缩放默认 schedule |
| **ReMDM-conf** | 基于 token confidence 调整 σ_t | 低 conf token 更易 remask |
| **ReMDM-loop** | 三阶段：draft → remask loop → fill | 最强策略 |

**ReMDM-loop 三阶段**：
1. 标准 MDLM 生成初稿（t > t_on）
2. 中间段固定 α 循环 remask+repredict（t_on ≥ t ≥ t_off）
3. 标准 MDLM 填充剩余（t < t_off）

#### 关键超参数

- `sampling_steps T`：可超过序列长度 L（inference-time scaling 的关键）
- `η_cap`：0.02（T≥1024）/ 0.04（T<1024）
- Loop: `t_on=0.55, t_off=0.05, α(t_on)=0.9`
- `top-p=0.9`

#### 实验结果

**OpenWebText, 169M MDLM**：
- T 超过 L 后，ReMDM MAUVE 持续上升；MDLM 饱和
- 15.62× MAUVE vs masked diffusion baseline

**Large pretrained dLLM downstream tasks**：
- 事实知识 + 推理任务一致提升

#### 与 Soft-Mask 关系

Soft-Mask [1] 实验明确展示：**SM + ReMDM 互补**，SM 在 embedding 层传递信息，ReMDM 在 sampling 层允许修正。

#### 局限

- ReMDM-conf 在 code 任务上可能**降低**性能（CORE 论文指出 stale confidence 问题）
- 需要仔细 tuning σ schedule
- 增加 step 数 = 增加 compute

---

### [5] CORE: Context-Robust Remasking

| 字段 | 内容 |
|------|------|
| **作者** | Zhai, Mollah, Wang, Shah (UCF) |
| **Venue** | ICML 2026 |
| **arXiv** | 2602.04096 |
| **代码** | https://github.com/UCF-CRCV/core |
| **类型** | Training-free |

#### 动机：Context Rigidity + Stale Confidence

1. **Context Rigidity**：early unmask 的 token 成为后续生成的 anchor，即使 suboptimal 也无法修正
2. **Stale Confidence**：ReMDM-conf 等用 **生成时** 的 confidence，不反映 **当前 context 更新后** 的可靠性
3. **Brittle ≠ Uncertain**：token 可在 early ambiguous context 下高 conf，但在 context 稳定后 incompatible

#### 方法

**Instability Score**：对 unmasked token i，mask 周围 context 后测 likelihood drop：
```
ỹ^i = MASK  if i ∈ S_t  else  y^i          # perturb context
ℓ_i = -log p_θ(y^i | ỹ)                     # instability score
```
- 高 ℓ_i → context-brittle → 优先 remask

**算法流程（每 REVISE_EVERY 步）**：
1. 从 unmasked 位置中选 CANDIDATE_M=32 个候选（margin-based screening）
2. 同时 mask 候选集 → 一次 forward pass 计算 instability
3. Remask instability 最高的 token（每 pass 最多 1 个）
4. 并行执行 base unmasking（正常 LCR）

#### 默认超参数（LLaDA sampler）

| 参数 | 值 | 含义 |
|------|-----|------|
| `REVISE_EVERY` | 8 | 每 8 步做一次 revision pass |
| `CANDIDATE_M` | 32 | 候选集大小 |
| `REVISION_WINDOW` | [0.25, 0.75) | 仅在中间 50% 步骤做 revision |
| `MAX_REVISE_PER_PASS` | 1 | 每次最多 remask 1 token |
| `BASE_MASKING` | low_confidence / topk_margin | 底层 unmask 策略 |

#### 实验结果（LLaDA-8B-Base）

| Benchmark | 提升 | 备注 |
|-----------|------|------|
| MBPP | **+9.2%** | 结构敏感任务增益最大 |
| HumanEval | 一致提升 | |
| GSM8K | 竞争性或提升 | 不牺牲推理换 code |
| ReMDM-conf + Top-k Margin | **-6.4% MBPP** | confidence-based remask 可能有害 |

#### 对跨步信息的意义

CORE 将跨步信息问题从 "保留预测" 扩展到 "评估 prediction 在当前 evolving context 下的 robustness"。这是纵向（history）+ 横向（context perturbation）的交叉。

---

### [6] RemeDi: Don't Settle Too Early

| 字段 | 内容 |
|------|------|
| **Venue** | ICLR 2026 Poster |
| **类型** | SFT + RL 训练 |

#### 方法

- 联合预测 token distribution + **per-token confidence score**
- Confidence 决定 unmask 后是否 remask
- Remask-aware training pipeline：SFT 教 detect+remask → RL 优化 trajectory

#### 与 RCR/ReMDM 关系

RemeDi 是 **学习版** 的 confidence-based remasking；RCR/ReMDM 是 heuristic 版。RemeDi 在 open-source DLM 上 SOTA。

---

### [7] Prophet: DLM Know the Answer Before Decoding

| 字段 | 内容 |
|------|------|
| **arXiv** | 2508.19982, 2025 |
| **类型** | Training-free early-exit |

#### 核心观察

DLM 在 decoding **早期** 即对最终答案有强倾向，但预测随 step 逐步稳定。中间 step logits 含有效信息。

#### 方法：Progress-Aware Early Commit

```
ḡ_t = aggregate margin over answer region
Early exit if: ḡ_t ≥ τ(p)

τ(p) = τ_high  if p < 0.33    # 早期：risk-averse，要求极高 conf
       τ_mid   if 0.33 ≤ p < 0.67
       τ_low   if p ≥ 0.67    # 后期：risk-tolerant
```

默认：`τ_high=8.0, τ_mid=5.0, τ_low=3.0`（logit margin 单位）

#### 实验（LLaDA-8B + Dream-7B）

- Full (T=50) vs Half (T=25) vs Prophet
- Prophet 在 Half 的 compute 下接近 Full 的质量

#### 对跨步信息的意义

Prophet 反向利用跨步信息——用 **confidence 演化趋势** 决定何时停止，而非如何传递。与 SchED 同类。

---

## 6. 横向机制：Unmask 顺序与优先级调度（详细）

---

### [8] On the Reasoning Abilities of MDMs (ICLR 2026 Oral)

| 字段 | 内容 |
|------|------|
| **作者** | Svete & Sabharwal |
| **arXiv** | 2510.13117 |

#### 理论贡献

1. MDM ≡ polynomially-padded looped transformer (PLT)
2. MDM 可解所有 CoT-augmented Transformer 可解问题
3. **Takeaway 2**：在 regular language 等可并行问题上，MDM 比 CoT **固有更高效**
4. CoT 的 "sequentiality bottleneck"：无法利用并行性

#### 对 unmask order 的含义

Unmask order = "order of thought"。不同 order 激活不同的 conditional expert `p_θ(x_i | x_U)`，order 选择是推理能力的一部分。

---

### [9] Self-Aware Scheduling (SAS)

| 字段 | 内容 |
|------|------|
| **arXiv** | 2606.23567, 2026 |
| **项目页** | https://jimmyxu123.github.io/SAS |

#### 方法

1. 推导 sequential decoding mismatch 的 KL 上界 → dense self-aware reward
2. 将 order selection 建模为 policy optimization（frozen denoiser）
3. GRPO 训练 lightweight order policy

#### 结果

| 设置 | 结果 |
|------|------|
| Sudoku 1B MDM | 82.0% → 91.8%（best heuristic → SAS） |
| + second-stage finetune on learned trajectories | 97.5% |
| LLaDA-8B GSM8K | 64% → 76% pass@1 |
| LLaDA-8B MBPP | 39.5% → 41% |

#### 对跨步信息的意义

SAS 学习 **横向 order**，不改变纵向信息传递，但与任何纵向机制正交可组合。

---

### [10] Learning Unmasking Policies (Jazbec et al., ICML 2026 Oral)

| 字段 | 内容 |
|------|------|
| **Venue** | **ICML 2026 Oral**（Apple ML Research） |
| **arXiv** | 2512.09106 |
| **ICML** | https://icml.cc/virtual/2026/oral/71028 |

#### 方法

- 将 masked diffusion sampling 形式化为 **MDP**
- dLLM = environment；action = 选择 unmask 哪些位置
- **Single-layer transformer policy** 读 token confidences → unmask 决策
- RL 训练（类似 GRPO）

#### 关键发现

- 训练 policy 在 **full-diffusion** 设定下优于 heuristic
- 对 **更大 block size** 更鲁棒（heuristic 在大 block 下退化）
- Policy 可 transfer 到 out-of-domain

#### 与 test-time 关系

训练侧方法，但证明 unmask 决策可以用 confidences 序列做 **learned scheduling**；test-time 可用 heuristic 近似。

---

### [11] Confidence-based Heuristics 族（详细）

#### LLaDA Low-Confidence Remasking (LCR)

```
每步 remask n_t = ⌊(T-t)/T · L⌋ 个 token
score_i = 1 - p_θ(x^i | x̄_t)    # 低 conf → 高 score → remask
```

#### MaskGIT High-Confidence

```
每步 unmask top-K 最高 conf token（与 LCR 相反）
```

#### Fast-dLLM Threshold

```
Unmask all i where conf_i > τ
若 none exceed τ → unmask argmax conf（防 stuck）
```

#### Top-k Margin (Kim et al., 2025)

```
margin_i = p_(1) - p_(2)    # top-1 vs top-2 概率差
```

#### SchED (Mohamed et al., arXiv:2512.02892, ACL 2026 Findings)

**Progress-aware early exit**：
```
ḡ_t = aggregate top-2 logit margin over answer region
Exit if: ḡ_t ≥ τ(p)

τ(p): linear / cosine / exponential schedule from τ_high to τ_low
```

- Instruct models: **~4× speedup**, 99.8–100% quality retention
- Base models: up to 2.34× speedup, 99.1–100% retention
- 代码：https://github.com/amr-mohamedd/SchED

#### OSDT: One-Shot Dynamic Thresholding (OpenReview 2026)

- Phase 1：在单个序列上校准 threshold
- Phase 2：reuse 于后续输入（block 或 step-block 粒度）
- 针对 task-level confidence pattern 适配

**Heuristics 共同局限**：决策基于 **当前步** conf，不保留 `conf_{t-1}` 历史。

---

### [12] SOAR: Confidence-Switched Position Beam Search

| 字段 | 内容 |
|------|------|
| **arXiv** | 2602.10953, 2026 |
| **类型** | Training-free |

#### 方法

```
if any conf_i > τ:
    Parallel Mode: unmask all conf > τ 的 token（Fast-dLLM 式加速）
else:
    Beam Search Mode: 对 top-K 最 conf 的 **位置** 分别探索（k=1 token/step）
```

#### 默认参数

| 模型 | τ | max beam K | max length |
|------|---|-----------|------------|
| LLaDA-8B | 0.95 | 2 | 256 |
| Dream-7B | 0.90 | 2 | 512 |

#### 结果

- τ > 0.8 时 SOAR 一致优于 standard decoding（accuracy + speed）
- GSM8K (coding) + MBPP (math) 验证

#### 对跨步信息的意义

SOAR 在 **横向** 切换并行/搜索模式；高 conf 时激进 unmask（纵向信息传递少），低 conf 时保守搜索（更多 step 积累信息）。

---

## 7. Test-Time Scaling：搜索、集成与 Step-Level 组合（详细）

---

### [13] HEX: Hidden Semi-Autoregressive Experts

| 字段 | 内容 |
|------|------|
| **arXiv** | 2510.05040, ICLR 2026 |
| **类型** | Training-free |

#### 核心洞察

DLM 训练产生一族 conditional：`p_θ(x_i | x_U)`，不同 visible set U = 不同 "expert"。

Bayes-optimal mixture：
```
p_mix(x_i | prompt) = E_{U}[p_θ(x_i | x_{prompt}, x_U)]
```

不可 tractable → 用不同 **block schedule** 的 semi-AR 解码近似：
```
p_mix ≈ E_{b~B}[p_θ(x_i | x_{prompt}, x_{U_b})]
```

#### 实现

- 采样多种 block schedule b
- 每种 schedule 生成一个 answer
- **Majority vote** 或 mixture averaging

#### 对跨步信息的意义

HEX 在 **trajectory 级** 集成不同 order 的推理路径；每条路径内部的跨步信息仍可能被浪费。

---

### [14] S³: Stratified Scaling Search

| 字段 | 内容 |
|------|------|
| **arXiv** | 2604.06260, 2026 |
| **类型** | Training-free |

#### 方法

- 每 denoising step 维护 **多条 partial trajectory**
- Lightweight reference-free verifier 评估
- Selective resample promising candidates
- 近似 reward-tilted Gibbs distribution

#### 结果（LLaDA-8B-Instruct）

- MATH-500, GSM8K, ARC-Challenge, TruthfulQA 一致提升
- 数学推理任务增益最大

---

### [15] MEDAL: MCTS for DLM Inference

| 字段 | 内容 |
|------|------|
| **Venue** | EACL 2026 Findings |
| **类型** | Training-free |

#### 动机

Greedy confidence heuristic 是 **myopic** 的——early commit 高 conf token 后 forced adapt。

#### 方法

- 将 unmask trajectory 建模为 search tree
- MCTS 平衡 exploitation（高 conf）与 exploration（alternative trajectories）
- 两个创新：(1) MCTS 适配 DLM 的 parallel unmask 结构；(2) 用 model 自身 distribution 作 signal

---

### [16] UnMaskFork (UMF)

| 字段 | 内容 |
|------|------|
| **Venue** | ICML 2026 |
| **arXiv** | 2602.04344 |
| **代码** | https://github.com/iamshouvikmitra/mcts-diffusion |

#### 方法

- **Deterministic action branching** 替代 stochastic sampling
- 分支 = 不同 MDLM 或不同 inference 参数的 unmask 决策
- MCTS + **node caching**（复用 partial unmasked state）
- NFE budget 内优化

#### 默认 MCTS 配置

```python
MCTSConfig(
    nfe_budget=12288,
    c_exp=1.0,
    mask_ratio_schedule=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    gen_length=768,
)
```

#### 结果

- 代码 benchmark（HumanEval+, MBPP+, LiveCodeBench）SOTA test-time scaling
- 数学推理 task 也 show scalability

---

### [17] Diffusion Stitching (Reward-Guided)

| 字段 | 内容 |
|------|------|
| **arXiv** | 2602.22871, 2026 |
| **代码** | https://github.com/roymiles/diffusion-stitching |

#### 方法

1. 扩散模型采样 **多条** 低成本 reasoning trajectory
2. **Process Reward Model (PRM)** 对每条 trajectory 的 **每个 intermediate step** 打分
3. **Stitch** 跨 trajectory 的最高质量 step → 组合 rationale
4. AR solver 读 stitched rationale 出最终答案

#### 结果

- 6 个 math/coding task 平均 **+23.8%** accuracy
- **1.8× latency reduction** vs Dream/LLaDA/TiDAR

#### 对跨步信息的核心启示

> **Step-level 信息** 比 trajectory-level vote 更有价值——恰是 "第 n 步信息应在后续使用" 的最强实证之一。

---

## 8. Latent / 表示层 Test-Time 方法

### [18] LATTS: Latent Space Test-Time Scaling

| 字段 | 内容 |
|------|------|
| **Venue** | OpenReview 2026 |

- 在 **latent embedding space** 扩展 iterative self-refinement steps
- 将 CoT 从 spatial（扩序列长度）转为 temporal（扩 refinement 步数）
- LLaDA-Instruct：GSM8K +4.1%, MATH +4.8%, MBPP +3.2%
- 需 brief post-training

### [19] LaDiR: Latent Diffusion Reasoner (Apple, ICLR 2026)

- VAE 编码 reasoning step → block of latent thought tokens
- Latent diffusion blockwise denoising（intra-block bidirectional, inter-block causal）
- 超越 LLaDA CoT SFT、Coconut 等 latent reasoning 方法
- 需完整训练 pipeline

---

## 9. 训练侧方法（定义 Inference 上下界）

| 论文 | arXiv | 核心贡献 | 与 test-time 关系 |
|------|-------|----------|-------------------|
| **MDPO** | 2508.13148 | GRPO 对齐 inference remasking schedule + RCR | RCR 是 MDPO 的 training-free 插件 |
| **RemeDi** | ICLR 2026 | SFT+RL 学 remask + confidence | 学习版 RCR/CORE |
| **Soft-Mask [1]** | 2510.17206 | Embedding 层 cross-step feedback | 需 finetune；定义纵向 upper bound |
| **MRP** | 2605.18817 | Logit 残差 head | 小模块训练；可 test-time plug-in（direct mode） |
| **SAS** | 2606.23567 | Learned unmask order | Order policy 可与任意 decoder 组合 |
| **Unmask Policy** | 2512.09106 | RL unmask MDP | Full-diffusion 下优于 heuristic |
| **DUS** | ICML 2026 | Dilated unmask schedule | 横向分组，test-time |
| **DyLLM** | ICML 2026 | 跨步 attention 相似度 | 纵向 saliency，test-time |
| **TSPD+CE** | ICML 2026 | token trajectory 特征 | 纵向收敛检测，test-time |
| **Dream 7B** | 2508.15487 | AR init + CART | 影响 step-wise logits 质量 |
| **Fast-dLLM v2** | 2509.26328 | Block diffusion + cache | 工程基础设施 |

---

## 10. 机制对比与文献演化时间线

### 10.1 全方法对比表

| 方法 | 信息层 | 跨步载体 | masked 保留 | unmask 可修正 | 优先级信号 | 额外训练 | Compute |
|------|--------|----------|-------------|---------------|-----------|----------|---------|
| LLaDA LCR | logit | 无 | ✗ | ✗ | 当前 conf | ✗ | 1× |
| Soft-Mask | embed | top-k 混合 | ✓ | ✗ | 当前 conf | ✓ | 1× |
| MRP | logit | Δlogits | ✓ | ✓ | threshold+residual | 小 head | 0.7–1× |
| RCR | logit | running conf | 部分 | ✓ | running conf | ✗ | 1× |
| ReMDM | sampling | σ schedule | — | ✓ | σ_t | ✗ | 1–3× |
| CORE | logit | instability | — | ✓ | context probe | ✗ | 1.1× |
| Prophet/SchED | logit | margin trend | — | ✗ | τ(p) schedule | ✗ | 0.25–0.5× |
| SOAR | logit | conf+beam | 部分 | ✗ | conf+search | ✗ | 0.5–2× |
| SAS | order | learned policy | ✓ | 部分 | learned | ✓ | 1× |
| HEX | trajectory | multi-schedule | — | ✗ | vote | ✗ | N× |
| S³/UMF/MEDAL | trajectory | search tree | ✓ | ✓ | MCTS/verifier | ✗ | N× |
| Stitching | step | PRM scores | — | — | PRM | ✗ | N× |

### 10.2 时间线

```
2025-02  LLaDA 8B 发布 — 定义 block+LCR baseline
2025-03  ReMDM — 首个 principled remasking sampler + inference scaling
2025-08  Dream 7B, MDPO+RCR, Prophet, DLM Survey
2025-09  Fast-dLLM v2
2025-10  Soft-Mask [1], HEX, MDM Reasoning (理论), LaDiR
2025-12  SchED, Unmask Policy (Apple)
2026-02  CORE, UMF, SOAR, Diffusion Stitching
2026-04  S³
2026-05  MRP
2026-06  SAS
2026-07  ICML 2026 集中收录：CORE, UMF, DyLLM, DUS, WeDLM, Eso-LM, LoMDM, Scaling Laws 等
2026-07  (本调研)
```

### 10.3 演化逻辑

```
Binary mask 丢弃信息 [问题]
    ↓
Soft-Mask: embedding 层保留 top-k [1]
MRP: logit 层预测 step 残差
    ↓
Freeze 无法修正 [问题]
    ↓
ReMDM: principled remasking
RCR: running confidence remask
CORE: context perturbation remask
    ↓
Myopic single-step conf [问题]
    ↓
SAS/Unmask Policy: learned order
SOAR: conf-switched beam
Prophet/SchED: progress-aware schedule
    ↓
Single trajectory 不够 [问题]
    ↓
HEX: multi-schedule ensemble
S³/UMF/MEDAL: search
Stitching: step-level PRM combine
```

---

## 11. 统一框架与研究空白

### 11.1 综合状态表示

```text
S_{t,i} = {
  e_{t,i}    : soft embedding（Soft-Mask）
  c*_{t,i}   : running max confidence（RCR）
  Δℓ_{t,i}   : MRP 预测的 logit 残差
  ℓ_{t,i}   : 当前步 logits
  age_{t,i}  : 连续 masked 步数
  ℓ_instab   : CORE instability score（对已 unmask 位置）
}
```

### 11.2 综合优先级（文献启发式组合）

```text
# Unmask 优先级（masked 位置）
priority(i) = w1·conf(ℓ_{t,i}) + w2·c*_{t,i} + w3·margin(ℓ_{t,i}) - w4·age_{t,i}

# Defer（暂缓 unmask）
defer(i) ← conf < τ_defer  OR  conf << c*  OR  age < τ_min

# Remask（已 unmask 位置）
remask(j) ← ℓ_instab(j) > τ_brittle  OR  running_conf_drop(j)
```

### 11.3 研究空白

| # | 空白 | 详情 | 可能方向 |
|---|------|------|----------|
| 1 | **纵向信号未统一** | SM/RCR/MRP 各用不同载体 | Multi-signal fusion |
| 2 | **Test-time Soft-Mask** | [1] 需 finetune | Inference-time top-k embed 近似 |
| 3 | **Defer/Commit 无统一框架** | 各 heuristic 独立 | 联合 Prophet+RCR+CORE 信号 |
| 4 | **纵向×横向交互少** | 多数只改一个方向 | SOAR beam 内用 running conf |
| 5 | **Step-level 组合未探索** | Stitching 用 PRM 但非 DLM 原生 | DLM 原生 step logits 组合 |
| 6 | **Over-denoising 未系统解决** | MDPO 发现但仅 training 侧 | Test-time 检测+rollback |
| 7 | **Compute-quality Pareto** | 方法分散在不同 benchmark | 统一 NFE budget 对比 |

---

## 12. 推荐实验路径（LLaDA-8B）

### Phase 0：环境与 Baseline

```bash
bash scripts/download_llada.sh instruct
```

- 模型：`/root/autodl-tmp/model/LLaDA/instruct`
- Benchmark：GSM8K, MATH-500, HumanEval, MBPP, ARC-Challenge
- Baseline：LLaDA LCR + block diffusion, T=128, block=64

### Phase 1：Training-Free 纵向（优先）

1. **RCR** — 最低成本，直接 plug-in
2. **ReMDM-loop** — 需 tuning σ schedule
3. **CORE** — 需额外 forward，评估 cost/benefit
4. **Step-wise 分析** — confidence trajectory, jitter, over-denoising 率

### Phase 2：Training-Free 横向

1. Heuristic 对比：LCR vs High-Conf vs Margin vs Threshold
2. **SOAR** — conf-switch beam
3. **Prophet/SchED** — early exit（效率方向）

### Phase 3：Test-Time Scaling

1. **HEX** — 2-4 种 block schedule + vote
2. **S³ 或 UMF** — 给定 NFE budget=12288
3. 分析 step-level vs trajectory-level

### Phase 4：Upper Bound（可选）

1. Soft-Mask continued pretrain（若有 compute）
2. MRP head 训练
3. SAS order policy

---

## 13. 完整论文索引与参考文献

### 13.1 完整索引（28 篇）

| # | 论文 | arXiv | Venue | 类型 |
|---|------|-------|-------|------|
| 1 | Soft-Masked DLM | 2510.17206 | ICLR 2026 | 训练 |
| 2 | MRP | 2605.18817 | 2026 | 训练+decode |
| 3 | MDPO + RCR | 2508.13148 | 2025 | 训练+decode |
| 4 | ReMDM | 2503.00307 | NeurIPS 2025 | test-time |
| 5 | CORE | 2602.04096 | ICML 2026 | test-time |
| 6 | RemeDi | ICLR 2026 | ICLR 2026 | 训练 |
| 7 | Prophet | 2508.19982 | 2025 | test-time |
| 8 | On Reasoning Abilities of MDMs | 2510.13117 | ICLR 2026 Oral | 理论 |
| 9 | SAS | 2606.23567 | 2026 | 训练 |
| 10 | Learning Unmasking Policies | 2512.09106 | **ICML 2026 Oral** | 训练 |
| 11 | SchED | 2512.02892 | ACL 2026 | test-time |
| 12 | SOAR | 2602.10953 | 2026 | test-time |
| 13 | HEX | 2510.05040 | ICLR 2026 | test-time |
| 14 | S³ | 2604.06260 | 2026 | test-time |
| 15 | MEDAL | — | EACL 2026 | test-time |
| 16 | UnMaskFork | 2602.04344 | ICML 2026 | test-time |
| 17 | Diffusion Stitching | 2602.22871 | 2026 | test-time |
| 18 | LATTS | — | OpenReview 2026 | test-time |
| 19 | LaDiR | 2510.04573 | ICLR 2026 | 训练 |
| 20 | LLaDA | 2502.09992 | 2025 | 基座 |
| 21 | Dream 7B | 2508.15487 | 2025 | 基座 |
| 22 | Fast-dLLM v2 | 2509.26328 | 2025 | 模型+decode |
| 23 | DLM Survey | 2508.10875 | 2025 | 综述 |
| 24 | Discrete Diffusion Survey | 2506.13759 | 2025 | 综述 |
| 25 | dLLM Acceleration Survey | TechRxiv 2025 | 2025 | 综述 |
| 26 | ParallelBench | — | OpenReview 2026 | Benchmark |
| 27 | OSDT | — | OpenReview 2026 | test-time |
| 28 | Simple MDLM (Block) | 2024 | NeurIPS 2024 | 基线 |

### 13.1b ICML 2026 新增索引（见 §14 详述）

| # | 论文 | arXiv | 类型 | 与跨步信息相关度 |
|---|------|-------|------|-----------------|
| IC-1 | DyLLM | — | test-time 加速 | ⭐⭐⭐⭐⭐ 跨步 attention 相似度 |
| IC-2 | TSPD + Confidence Extrapolation | — | test-time 加速 | ⭐⭐⭐⭐⭐ token trajectory |
| IC-3 | Plan for Speed (DUS) | — | test-time 调度 | ⭐⭐⭐⭐ 横向 dilated group |
| IC-4 | Explore-Then-Exploit (ETE) | — | test-time 调度 | ⭐⭐⭐⭐ 高 uncertainty 探索 |
| IC-5 | WeDLM | — | 架构+decode | ⭐⭐⭐ causal KV cache |
| IC-6 | Eso-LM | 2506.01928 | 架构+训练 | ⭐⭐⭐ hybrid AR-MDM |
| IC-7 | LoMDM / OeMDM | — | 训练 | ⭐⭐⭐⭐ 可学习 generation order |
| IC-8 | Any-Order GPT as MDM | — | 架构 | ⭐⭐⭐ decoder-only MDM |
| IC-9 | Scaling Beyond Masked DLM | 2602.15014 |  scaling | ⭐⭐⭐ Pareto 分析 |
| IC-10 | CDLM | — | 训练/蒸馏 | ⭐⭐ few-step consistency |
| IC-11 | IMDM | — | 训练/蒸馏 | ⭐⭐⭐ 随机 infinite mask |
| IC-12 | XDLM | — | 训练 | ⭐⭐ MDM+UDLM 统一 |
| IC-13 | CANDI | — | 训练 | ⭐⭐ hybrid discrete-continuous |
| IC-14 | CoDiLA | — | test-time | ⭐⭐⭐ 块内 AR 辅助 |
| IC-15 | PoE-Bridge | — | test-time | ⭐⭐⭐ DLM-AR 桥接 |
| IC-16 | Soft-Masked DLM | 2510.17206 | ICLR 2026 | ⭐⭐⭐⭐⭐（Hersche 亦参与 CoDiLA） |

### 13.2 BibTeX

```bibtex
@inproceedings{hersche2026softmasked,
  title={Soft-Masked Diffusion Language Models},
  author={Hersche, Michael and Moor-Smith, Samuel and Hofmann, Thomas and Rahimi, Abbas},
  booktitle={ICLR}, year={2026}, eprint={2510.17206},
}

@article{nie2025llada,
  title={Large Language Diffusion Models},
  author={Nie, Shen and others},
  journal={arXiv:2502.09992}, year={2025},
}

@article{wang2025remdm,
  title={Remasking Discrete Diffusion Models with Inference-Time Scaling},
  author={Wang, Guanghan and Schiff, Yair and Sahoo, Subham and Kuleshov, Volodymyr},
  journal={NeurIPS}, year={2025}, eprint={2503.00307},
}

@article{he2025mdpo,
  title={MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models},
  author={He, Haoyu and Renz, Katrin and Cao, Yong and Geiger, Andreas},
  journal={arXiv:2508.13148}, year={2025},
}

@article{zhai2026core,
  title={CORE: Context-Robust Remasking for Diffusion Language Models},
  author={Zhai, Kevin and Mollah, Sabbir and Wang, Zhenyi and Shah, Mubarak},
  journal={arXiv:2602.04096}, year={2026},
}

@article{mrp2026,
  title={Multi-Token Residual Prediction},
  journal={arXiv:2605.18817}, year={2026},
}

@inproceedings{svete2026reasoning,
  title={On the Reasoning Abilities of Masked Diffusion Language Models},
  author={Svete, Anej and Sabharwal, Ashish},
  booktitle={ICLR}, year={2026}, eprint={2510.13117},
}

@article{xu2026sas,
  title={Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models},
  journal={arXiv:2606.23567}, year={2026},
}

@article{mohamed2025sched,
  title={Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules},
  author={Mohamed, Amr and Zhang, Yang and Vazirgiannis, Michalis and Shang, Guokan},
  journal={arXiv:2512.02892}, year={2025},
}

@article{soar2026,
  title={Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models},
  journal={arXiv:2602.10953}, year={2026},
}

@article{hex2026,
  title={Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts},
  journal={arXiv:2510.05040}, year={2026},
}

@article{umf2026,
  title={UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching},
  journal={arXiv:2602.04344}, year={2026},
}

@article{stitching2026,
  title={Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching},
  journal={arXiv:2602.22871}, year={2026},
}

@inproceedings{jazbec2026unmaskpolicy,
  title={Learning Unmasking Policies for Diffusion Language Models},
  author={Jazbec, Metod and others},
  booktitle={ICML}, year={2026}, eprint={2512.09106},
}

@article{sahoo2026scaling,
  title={Scaling Beyond Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and others},
  journal={arXiv:2602.15014}, year={2026},
}

@article{sahoo2025esolm,
  title={Esoteric Language Models},
  author={Sahoo, Subham Sekhar and others},
  journal={arXiv:2506.01928}, year={2025},
}
```

---

## 14. ICML 2026 录取 DLM 论文专节（扩展）

> ICML 2026 是 DLM inference / decoding 研究的**集中爆发点**。本节按主题分类，补充正文未详述的 ICML 论文；已收录的 CORE [5]、UnMaskFork [16]、Learning Unmasking Policies [10] 见对应章节。

### 14.0 ICML 2026 DLM 论文总览

| 分类 | 论文 | Presentation | Test-time? |
|------|------|--------------|------------|
| **跨步轨迹 / Remask** | CORE | Poster #2000 | ✓ |
| **跨步轨迹 / Remask** | DyLLM | Poster #1914 | ✓ |
| **跨步轨迹 / Remask** | TSPD + CE | Poster #2706 | ✓ |
| **Unmask 调度** | DUS (Plan for Speed) | Poster #1914 | ✓ |
| **Unmask 调度** | ETE (Bits to Rounds) | Poster | ✓ |
| **Unmask 调度** | Learning Unmasking Policies | **Oral** | 训练 policy |
| **Unmask 调度** | LoMDM | Poster #911 | 训练 |
| **Test-time Search** | UnMaskFork | Poster #1701 | ✓ |
| **并行解码** | CoDiLA | Poster #2807 | ✓ |
| **并行解码** | PoE-Bridge | Poster #2500 | ✓ |
| **架构 / Cache** | WeDLM | **Oral** | decode 框架 |
| **架构 / Cache** | Eso-LM | Poster #2510 | 训练+采样 |
| **架构 /  formulation** | Any-Order GPT as MDM | Poster #2611 (+ Oral) | — |
| **Scaling / 范式** | Scaling Beyond Masked DLM | Poster | 训练 |
| **Few-step / 训练** | CDLM | Poster | 训练 |
| **Few-step / 训练** | IMDM | Poster #2602 | 训练 |
| **范式统一** | XDLM | Poster | 训练 |
| **范式统一** | CANDI | Poster #3904 | 训练 |

---

### 14.1 Test-Time：跨步轨迹与 Token 收敛（⭐ 与本项目最相关）

#### [IC-1] DyLLM: Saliency-based Token Selection and Partial Attention

| 字段 | 内容 |
|------|------|
| **作者** | Younjoo Lee, Junghoo Lee, Seungkyun Dan, Jaiyoung Park, Jung Ho Ahn |
| **Venue** | ICML 2026 Poster #1914 |
| **类型** | Training-free inference 加速 |

**核心观察**：跨 denoising step，**大多数 token 的 attention context 表示稳定**；仅少数 **salient tokens** 对下一步更新有实质贡献。

**方法**：
```
saliency(i) = 1 - cos_sim(attn_ctx_{t,i}, attn_ctx_{t-1,i})
```
- 仅对 salient token 重算 FFN + Attention
- 其余 token **复用 cached activations**

**结果**：LLaDA / Dream 上最高 **9.6× throughput**，精度基本保持。

**与跨步信息的关系**：直接利用 **step t 与 t-1 的 hidden/attention 差异** 决定哪些位置需要继续 denoising——是"第 n 步信息用于第 n+1 步决策"的 efficiency 向实例。

---

#### [IC-2] TSPD + Confidence Extrapolation (CE)

| 字段 | 内容 |
|------|------|
| **作者** | Zekai Li, Ji Liu, Yiqing Huang, Ziqiong Liu, Dong Li, Emad Barsoum |
| **Venue** | ICML 2026 Poster #2706 |
| **类型** | Training-free |

**动机**：大量 denoising step 浪费在已收敛 token 的重复 refine 上。

**TSPD（Temporal-Spatial Parallel Decoding）**：
- 轻量 **correctness sensor** 读 per-token trajectory 特征：
  - confidence、entropy、**momentum**（跨步变化率）
  - token position（spatial）
- 判定 token 已收敛 → **safe fix**（不再 remask）

**Confidence Extrapolation (CE)**：
- Training-free 状态空间模块，**forecast 未来 logit 趋势 + uncertainty**
- 支持 proactive look-ahead、oscillation 时 stabilization

**与跨步信息的关系**：明确将 diffusion decoding 建模为 **dynamic control problem**，token-wise trajectory 是核心信号——与 Prophet/SchED 的 progress-aware 思路一致，但更细粒度（per-token）。

---

#### [IC-3] CORE（已在 §5 详述，ICML 2026 Poster #2000）

- 链接：https://icml.cc/virtual/2026/poster/62928
- ICML 官方标题为 **CoRe**（Context-Robust Remasking）

---

### 14.2 Test-Time：Unmask 调度与并行策略

#### [IC-4] Plan for Speed: Dilated Unmasking Scheduler (DUS)

| 字段 | 内容 |
|------|------|
| **作者** | Omer Luxembourg, Haim Permuter, Eliya Nachmani |
| **Venue** | ICML 2026 Poster #1914 |
| **ICML** | https://icml.cc/virtual/2026/poster/65445 |
| **类型** | Inference-only, planner-free |

**问题**：confidence-based parallel unmask 忽略 **相邻 token 交互**，并行 unmask 相邻位置时 joint entropy 激增，质量退化至近似 AR 速度。

**DUS 方法**：
1. 将序列位置划分为 **non-adjacent dilated groups**（扩张分组）
2. 同组内 parallel unmask，组间顺序推进
3. 目标：最小化每步 **joint entropy gain 上界**

**Benchmark**：GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro, IFEval

**与跨步信息的关系**：**横向**调度——决定"哪些位置同时 unmask"而非保留 masked 位置的 step 历史；与 DUS 正交可组合 RCR/CORE。

---

#### [IC-5] From Bits to Rounds: Explore-Then-Exploit (ETE)

| 字段 | 内容 |
|------|------|
| **作者** | Hengyu Fu, Baihe Huang, Virginia Adams, Charles Wang, Junkeun Yi, Mohammad Mahdi Kamani, Venkat Krishna Srinivasan, Jiantao Jiao |
| **Venue** | ICML 2026 Poster |
| **类型** | Training-free |

**理论**：信息论下界——decoding rounds ∝ 总信息量 / 每轮信息预算（**bits-to-rounds principle**）。

**ETE 策略**：
- **Cross-block decoding** + 对 **high-uncertainty token 主动探索**
- 不只 unmask 高 conf token，而是 reshape conditional distribution 触发 confident prediction cascade
- 兼容 KV caching → 提升 tokens/sec

**与跨步信息的关系**：低 conf / high uncertainty 位置的 **defer** 与 **explore** 显式分离——"不优先生成"的 theory-grounded 版本。

---

#### [IC-6] Learning Unmasking Policies（ICML 2026 Oral，§6 已述）

- **ICML Oral**：https://icml.cc/virtual/2026/oral/71028
- Single-layer transformer policy：confidences → unmask 决策
- Full-diffusion 设定优于 heuristic；block 设定匹配 SOTA heuristic

---

#### [IC-7] LoMDM: Learnable-Order Masked Diffusion Model

| 字段 | 内容 |
|------|------|
| **作者** | Chunsan Hong, Sanghyun Lee, Jong Chul Ye |
| **Venue** | ICML 2026 Poster #911 |
| **ICML** | https://icml.cc/virtual/2026/poster/65757 |
| **类型** | 训练（from scratch） |

**方法链**：
1. **OeMDM**（order-expressive MDM）：统一 MDM / ARM / block diffusion 的 broad class
2. **LoMDM**：**联合学习** generation ordering + diffusion backbone（单目标，非两阶段）

**对比 SAS [9]**：SAS 在 frozen denoiser 上学 order；LoMDM 从头联合训练。

**意义**：generation order 是模型内生能力，非 inference plug-in。

---

### 14.3 Test-Time：Search 与 DLM-AR 桥接

#### [IC-8] UnMaskFork（§7 已述，ICML 2026 Poster #1701）

#### [IC-9] PoE-Bridge: Product-of-Experts Bridge

| 字段 | 内容 |
|------|------|
| **作者** | Juntong Shi, Brian Trippe, Jure Leskovec, Stefano Ermon, Minkai Xu |
| **Venue** | ICML 2026 Poster #2500 |
| **类型** | Test-time decoding framework |

**问题**：DLM parallel sampling 与 AR target 概率空间 gap 大 → importance sampling 需大量 particles。

**方法**：
```
P_bridge ∝ P_DLM · P_AR   (Product-of-Experts)
```
1. DLM multi-token sampling（proposal）
2. PoE rejection sampling 验证
3. AR target importance sampling 最终生成

**结果**：**5× speedup** vs 标准 DLM decode；恢复 ≥95% AR 性能。

---

#### [IC-10] CoDiLA: Coherent Diffusion with Local Autoregression

| 字段 | 内容 |
|------|------|
| **作者** | Michael Hersche, Nicolas Menet, Ronan Tanios, Abbas Rahimi（IBM） |
| **Venue** | ICML 2026 Poster #2807 |
| **ICML** | https://icml.cc/virtual/2026/poster/62309 |
| **类型** | Test-time（小 AR 辅助模型） |

**问题**：DLM parallel sampling 假设 token 独立 → 破坏 multi-token 结构（语法、代码）。

**方法**：
- Block 级 parallel generation（DLM）
- Block **内部** 由 compact AR model（如 0.6B）在 diffusion latents 上 sequential decode
- 跨 block 保持 bidirectional

**与 Soft-Mask [1] 关系**：同一第一作者 Hersche；SM 解决 embedding 层 cross-step feedback，CoDiLA 解决 parallel 时的 **local joint dependency**。

---

### 14.4 架构与高效 Decode 框架

#### [IC-11] WeDLM: Reconciling DLMs with Standard Causal Attention（ICML 2026 Oral）

| 字段 | 内容 |
|------|------|
| **作者** | Aiwei Liu, Minghua He, Shaoxun Zeng, et al. |
| **Venue** | **ICML 2026 Oral** |
| **ICML** | https://icml.cc/virtual/2026/oral/71142 |

**问题**：多数 DLLM 用 bidirectional attention → 无法 prefix KV cache → 并行优势无法转化为 wall-clock speed。

**WeDLM 核心**：
1. **Topological Reordering**：将已观测 token 移到物理 prefix，保留逻辑位置；masked 位置仍 causal 条件于全部观测 token
2. **Streaming decoding**：持续 commit conf token 到 growing prefix，避免 block diffusion 的 stop-and-wait

**结果**：vs vLLM-served AR baseline，推理 benchmark **~3×**；低 entropy 生成 **~10×**。

**意义**：基础设施级创新——使 DLM decode 兼容标准 causal cache；后续 test-time 方法（RCR/CORE/DyLLM）可在此框架上叠加。

---

#### [IC-12] Eso-LM: Esoteric Language Models（ICML 2026 Poster）

| 字段 | 内容 |
|------|------|
| **作者** | Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, et al. |
| **Venue** | ICML 2026 Poster #2510 |
| **arXiv** | 2506.01928 |
| **代码** | https://github.com/s-sahoo/Eso-LMs |

**方法**：
- **Causal attention** 替代 bidirectional MDM denoiser
- Hybrid loss：AR + MDM 目标，PPL 在两者间平滑插值
- **首个支持 MDM KV caching 且保留 parallel generation** 的方案

**结果**：vs 标准 MDM **14–65×** 加速；vs semi-AR **3–4×**。

**与 Scaling Beyond Masked DLM [IC-13] 关系**：Eso-LM 作为 interpolating diffusion 代表参与 1.7B scaling 对比。

---

#### [IC-13] Any-Order GPT as Masked Diffusion Model

| 字段 | 内容 |
|------|------|
| **作者** | Shuchen Xue, Tianyu Xie, Tianyang Hu, et al.（含 Fast-dLLM 作者 Shuchen Xue） |
| **Venue** | ICML 2026 Poster #2611 + **Oral 3A Diffusion Models** |
| **ICML** | https://icml.cc/virtual/2026/poster/61245 |

**贡献**：在 **decoder-only** 架构下公平对比 MDM（Any-Order AR）vs 标准 AR，剥离 encoder-only vs decoder-only 的 confound。

**发现**：decoder-only MDM 用 temperature annealing 等技巧可达 **~25× inference speedup**，PPL 与 AR comparable。

---

### 14.5 训练 / Scaling / 范式统一（定义 Inference 上下界）

#### [IC-14] Scaling Beyond Masked Diffusion Language Models

| 字段 | 内容 |
|------|------|
| **作者** | Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, et al. |
| **Venue** | ICML 2026 Poster |
| **arXiv** | 2602.15014 |
| **项目页** | https://s-sahoo.com/scaling-dllms/ |

**首个 IsoFLOP scaling 对比**：MDLM vs Duo (uniform-state) vs Eso-LM vs AR @ 1.7B

**关键结论**：
1. MDLM + simple CE loss → **12% FLOPs 效率提升**
2. **Perplexity 不能跨 diffusion family 比较**；应看 speed-quality Pareto
3. Duo @ 1.7B：**GSM8K 65.8** > AR 62.9 > MDLM 58.8（尽管 PPL 更差）

---

#### [IC-15] CDLM: Consistent Diffusion Language Models

| 字段 | 内容 |
|------|------|
| **Venue** | ICML 2026 Poster #66178 |
| **ICML** | https://icml.cc/virtual/2026/poster/66178 |

**MPDC 原则**：用 stochastic posterior bridges 集合替代连续 ODE 的唯一轨迹；学习 **path-independent denoiser**。

**意义**：few-step regime SOTA；统一 masked + uniform diffusion。

---

#### [IC-16] IMDM: Infinite Mask Diffusion for Few-Step Distillation

| 字段 | 内容 |
|------|------|
| **Venue** | ICML 2026 Poster #2602 |

**问题**：标准 MDM 的 deterministic single-state mask 存在 **factorization error 下界**，阻碍 few-step generation。

**方法**：**stochastic infinite-state mask** 降低理论下界；兼容 pretrained weights + distillation。

---

#### [IC-17] XDLM: Balancing Understanding and Generation

| 字段 | 内容 |
|------|------|
| **Venue** | ICML 2026 Poster #61499 |

**统一 MDLM（理解强）与 UDLM（生成强）** via stationary noise kernel。

**结果**：zero-shot text +5.4 vs UDLM；few-step image FID 54.1 vs MDLM 80.8；8B LLM **32 step MBPP 15.0**（baseline 翻倍）。

---

#### [IC-18] CANDI: Hybrid Discrete-Continuous Diffusion

| 字段 | 内容 |
|------|------|
| **Venue** | ICML 2026 Poster #3904 |

**Token identifiability 框架**：分析 Gaussian noise 对 discrete token 的两种 corruption 机制。

**CANDI**：解耦 discrete / continuous corruption；low NFE text generation 优于 masked diffusion。

---

### 14.6 ICML 2026 对本项目的启示

| 趋势 | 代表论文 | 建议 |
|------|----------|------|
| **Token trajectory 是 first-class signal** | DyLLM, TSPD+CE, Prophet | Phase 1 分析 conf/entropy/momentum 轨迹 |
| **Remask > 单纯 unmask** | CORE, RCR, ReMDM | 优先 plug-in CORE+RCR |
| **Order/Schedule 可学习或结构化** | DUS, ETE, LoMDM, Unmask Policy | 横向 ablation：DUS vs LCR |
| **Cache 是 speed 前提** | WeDLM, Eso-LM, DyLLM | 长期考虑 WeDLM/Eso-LM 基座 |
| **PPL ≠ 下游** | Scaling Beyond Masked DLM | 评估用 GSM8K/MBPP 而非仅 PPL |
| **DLM-AR 混合是主流** | PoE-Bridge, CoDiLA | 块内依赖 vs 跨步 logits 融合 |

---

*本文档为 dlm-seq-flow 项目的详细文献调研。模型下载见 `scripts/download_llada.sh`。*
