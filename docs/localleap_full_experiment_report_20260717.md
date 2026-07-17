# LocalLeap / LLaDA training-free decoding：完整实验与方法报告

> 截止时间：2026-07-17  
> 模型：LLaDA-8B-Instruct（`/root/autodl-tmp/model/LLaDA/instruct`）  
> 正式基线：原始 LLaDA low-confidence decoding  
> 当前机器：2 × NVIDIA RTX 4080 SUPER，单卡显存 32,760 MiB  
> 说明：本文严格区分“本地正式审计结果”“小样本/旧协议探索”“只实现未正式评测”和“论文报告值”。

## 1. 结论摘要

当前最可靠的单轨迹架构仍是：**原始 LLaDA 作为唯一正式 baseline，`symmetric attention + tau=0.004` 作为精度优先方法，`symmetric_fast + tau=0.004` 作为固定预算速度方法**。

最好的本地结果不是同一个方法在所有数据集上普遍获益：

- HumanEval，256 步：`symmetric` 为 **71/164（43.29%）**，原始 LLaDA 为 **67/164（40.85%）**，提升 **+2.44 个百分点**；配对 McNemar `p=0.34375`，尚不显著。
- HumanEval，128 步、固定 NFE：`symmetric_fast` 为 **52/164（31.71%）**，匹配步数的原始 LLaDA 为 **42/164（25.61%）**，提升 **+6.10 个百分点**；重新正确配对后 `p=0.12145`，仍未显著。
- MBPP，128 步、固定 NFE：`symmetric_fast` 为 **99/500（19.8%）**，原始 LLaDA 为 **89/500（17.8%）**，提升 **+2.0 个百分点**，`p=0.34814`，墙钟时间反而慢约 1.1%。
- MBPP，精度优先：directed attention 为 **123/500（24.6%）**，原始 LLaDA 为 **89/500（17.8%）**，提升 **+6.8 个百分点**，`p=0.00169`；但平均 NFE 约 150、墙钟速度只有 baseline 的 0.847 倍，不能当作等算力结果。
- MATH-500，128 步：baseline **152/500（30.4%）**；`symmetric_fast` **150/500（30.0%）**；精度优先 `symmetric` 仍为 **152/500（30.4%）**，却使用 1.203 倍 NFE。
- GSM8K，128 步、正确的 `flexible-extract`：baseline **905/1319（68.61%）**；`symmetric_fast` **901/1319（68.31%）**；精度优先 `symmetric` **865/1319（65.58%）**。这里没有泛化增益。

因此，当前证据支持的是：**跨步条件化稳定性在代码任务上有信息，但并非普适纠错信号；attention 排序与强依赖等待在数学任务上可能过度延迟或扰乱原始置信度顺序。** 目前不能宣称该方法整体优于原始 LLaDA。

## 2. 公平比较与审计协议

本地正式比较的共同设置为：

| 项目 | 设置 |
| --- | --- |
| Backbone | LLaDA-8B-Instruct 本地权重 |
| Baseline | 原始 low-confidence remasking，无 attention selector |
| Block length | 32 |
| Temperature | 0，确定性解码 |
| Seeds | lm-eval 0，NumPy/Torch/few-shot 1234 |
| Generation length | 正式新 benchmark 主要为 256 |
| Fixed-budget 条件 | baseline 与 `symmetric_fast` 使用相同步数和总 NFE |
| Accuracy-first 条件 | `symmetric` 可以 underfill，实际 NFE 必须单独报告 |
| 配对 | stable task id、prompt hash、target hash 对齐 |
| 健康检查 | duplicate/missing id、残留 mask、NFE 范围、生成记录与聚合一致性 |

HumanEval 使用修正后的代码抽取、清洗和执行路径；MBPP 使用 `pass_at_1`；MATH-500 使用 Prism 官方 system prompt 和答案归一化的本地 task；GSM8K 必须使用 `exact_match,flexible-extract`。

早期 lm-eval 的 HumanEval `create_test` 原始聚合曾把 baseline 报成 14/164，而独立 postprocess 审计得到 67/164。后续所有 HumanEval 正式结果均以 `localleap_postprocess_audit_v1` 的逐样本记录为准。

### 2.1 本轮新修正的配对问题

早期 `recovery_he_*_fast_{128,64}/paired_vs_baseline` 误用了 256 步的 67/164 baseline。方法和 baseline 各自的单独准确率审计仍有效，但旧配对差值和旧 p 值无效。

本轮已在新目录 `paired_vs_matched_baseline_v3` 中按相同步数重新配对，未覆盖旧文件：

| 步数 | 方法 | 匹配 baseline | 方法 | method-only / baseline-only | McNemar p |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | symmetric_fast | 42 | 52 | 22 / 12 | 0.12145 |
| 128 | directed_fast | 42 | 51 | 23 / 14 | 0.18774 |
| 64 | symmetric_fast | 28 | 14 | 7 / 21 | 0.01254 |
| 64 | directed_fast | 28 | 18 | 8 / 18 | 0.07552 |

所有四组重新配对的 prompt/target hash mismatch 均为 0。

## 3. HumanEval：所有完整正式结果

### 3.1 256 步精度优先与纵向方法

baseline 为 67/164（40.85%），总 NFE 为 `164 × 256 = 41,984`。

| 方法 | 关键设置 | 正确数 | Pass@1 | 相对 baseline | 配对结论 |
| --- | --- | ---: | ---: | ---: | --- |
| Original LLaDA | low-confidence | 67 | 40.85% | — | baseline |
| Symmetric | tau=0.05 | 67 | 40.85% | 0.00 pp | 输出与 baseline 相同 |
| Symmetric | tau=0.02 | 66 | 40.24% | -0.61 pp | p=1.0 |
| Symmetric | tau=0.01 | 66 | 40.24% | -0.61 pp | p=1.0 |
| Symmetric | tau=0.005 | 69 | 42.07% | +1.22 pp | p=0.75391 |
| Symmetric | tau=0.004 | 71 | 43.29% | +2.44 pp | 7 only / 3 only，p=0.34375 |
| Symmetric | tau=0.0025 | 71 | 43.29% | +2.44 pp | 与 tau=.004 相同得分 |
| Symmetric | tau=0.001 | 71 | 43.29% | +2.44 pp | 与 tau=.004 相同得分 |
| Symmetric | tau=0.0005 | 71 | 43.29% | +2.44 pp | 与 tau=.004 相同得分 |
| Directed read | tau=0.004 | 71 | 43.29% | +2.44 pp | 7 only / 3 only |
| Candidate stability | Top-8 + OTHER, delta=0 | 71 | 43.29% | +2.44 pp | 没有超过简单 top-1 稳定性 |
| Candidate frontier | Top-8 frontier | 67 | 40.85% | 0.00 pp | 2 only / 2 only |
| STCC vertical | JSD eps=.005 | 47 | 28.66% | -12.19 pp | 显著变差，p=.00119 |
| STCC vertical | JSD eps=.01 | 46 | 28.05% | -12.80 pp | 显著变差，p=.00075 |
| Global Top-K retention v2 | K=4 | 64 | 39.02% | -1.83 pp | 破坏成熟候选排序 |
| Parent-preserving retention v2.1 | K=4 only unstable tail | 71 | 43.29% | +2.44 pp | 恢复到 parent，不是新提升 |

这些 256 步 HumanEval 实验都没有残留 mask、重复 ID 或 prompt hash mismatch。

### 3.2 128/64 步并行速度档

| 步数 | 方法 | 正确数 | Pass@1 | 同步数 baseline | 变化 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Original LLaDA | 42 | 25.61% | — | — |
| 128 | symmetric_fast | 52 | 31.71% | 42 | +6.10 pp |
| 128 | directed_fast | 51 | 31.10% | 42 | +5.49 pp |
| 128 | retention v2 fast | 38 | 23.17% | 42 | -2.44 pp |
| 128 | retention v2.1 fast | 47 | 28.66% | 42 | +3.05 pp，但低于 parent 的 52 |
| 64 | Original LLaDA | 28 | 17.07% | — | — |
| 64 | symmetric_fast | 14 | 8.54% | 28 | -8.54 pp |
| 64 | directed_fast | 18 | 10.98% | 28 | -6.10 pp |

128 步是目前可接受的并行档；64 步时强制填满预算会大量提交尚未稳定的冲突位置，质量明显崩塌。

## 4. MBPP：所有完整正式结果

任务为 500 题、3-shot、generation length 256。

| 方法 | NFE | 正确数 | Pass@1 | 增益 | 配对/速度 |
| --- | ---: | ---: | ---: | ---: | --- |
| Original LLaDA | 128/题 | 89 | 17.8% | — | wall 14,522 s |
| symmetric_fast tau=.004 | 128/题 | 99 | 19.8% | +2.0 pp | p=.34814；wall 14,680 s；0.989× baseline speed |
| symmetric tau=.004 | 平均 147.89/题 | 121 | 24.2% | +6.4 pp | 69 only / 37 only；p=.00244 |
| directed tau=.004 | 平均 149.99/题 | 123 | 24.6% | +6.8 pp | 73 only / 39 only；p=.00169；wall 17,149 s |
| 64-step Original LLaDA | 64/题 | 46 | 9.2% | — | 单独速度档，不与 128-step 混比 |

MBPP 是横向 attention 最有利的正式证据，因为 128 步对应每步预算约 `b=2`，实际出现了同批冲突。directed 比 symmetric 多 2 题，说明 attention 的方向性可能有用，但两种方法之间还没有独立、预注册的直接显著性检验。

## 5. MATH-500 与 GSM8K：正式泛化结果

### 5.1 MATH-500（Prism prompt/normalization 对齐）

| 方法 | 正确数 | Accuracy | 总 NFE | NFE 比 | Wall time | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original LLaDA | 152/500 | 30.4% | 64,000 | 1.000× | 6,084 s | baseline |
| symmetric_fast | 150/500 | 30.0% | 64,000 | 1.000× | 6,311 s | -0.4 pp，p=.91613 |
| symmetric accuracy-first | 152/500 | 30.4% | 77,004 | 1.203× | 7,617 s | 无精度增益且更慢 |

### 5.2 GSM8K（0-shot，flexible-extract）

| 方法 | 正确数 | Accuracy | 总 NFE | NFE 比 | Wall time | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original LLaDA | 905/1319 | 68.61% | 168,832 | 1.000× | 13,100 s | baseline |
| symmetric_fast | 901/1319 | 68.31% | 168,832 | 1.000× | 13,729 s | -0.303 pp，p=.85924 |
| symmetric accuracy-first | 865/1319 | 65.58% | 198,984 | 1.179× | 16,172 s | -3.033 pp，p=.02466，显著负向 |

早期 `strict-match` 得到 baseline 1/1319、method 0/1319，是 evaluator 配置错误，不是模型结果。正式 GSM8K 结论只能使用上述 `flexible-extract` v3 记录。

## 6. 中间变量对横向/纵向假设的检验

### 6.1 tau 变化实际代表什么

HumanEval 256 步中：

| tau | 强依赖候选数 | 不稳定候选数 | 得分 |
| ---: | ---: | ---: | ---: |
| .005 | 381,662 | 43,491 | 69/164 |
| .004 | 453,215 | 52,135 | 71/164 |
| .0005 | 650,752 | 70,714 | 71/164 |

当 tau 从 .005 降到 .004，更多跨步变化被视为“真实新条件到达”，方法从 69 提升到 71。继续降低到 .0005 后得分平台化；此时几乎所有有历史的候选都被视为强依赖，而且“不稳定数=top-1 变化数”。这说明有效核心更接近**相邻步骤 top-1 稳定性排序**，而不是 tau 精确刻画了一个可迁移的 attention 物理边界。

### 6.2 横向证据的边界

HumanEval 256 步的预算为 `b=1`，所有 tau 实验的 `rejected_pairs_total=0`。所以 71/164 的提升不能作为“避免强依赖 token 同批提交”的证据，只能支持纵向排序。

HumanEval 128 步和 MBPP 128 步才真正产生横向冲突：

- retention v2.1 HumanEval-128：rejected pairs 91,129，forced fills 4,343；
- symmetric MBPP：rejected pairs 255,310，underfilled steps 16,345；
- MBPP directed 结果 123/500，略高于 symmetric 121/500；
- 平均 attention asymmetry 在 HumanEval 约 .0051、MBPP 约 .0064，相比平均对称 dependency 约 .009–.010，并非可以忽略。

横向结论因此是：**attention 确实不完全对称，方向版本值得保留；但固定 NFE 下的强制填充会重新引入冲突，质量与速度之间仍有直接张力。**

### 6.3 Top-K/JSD 为什么没有超过 top-1

Top-8 candidate-memory 的平均 overlap 为 6.90/8，稀疏 JSD 均值约 .0163，说明多数相邻步骤只是候选集合内部的小范围重排。完整 candidate stability 得分仍是 71/164；frontier 只有两个 tie-boundary frontier 真正被接受，最终 67/164，与 baseline 相同。这解释了此前“candidate-frontier 为什么和 baseline 一样”：不是信息泄漏，而是实际改变决策的边界事件极少。

全局 top-K retention v2 把 Top-K 连续性放在成熟候选的置信度之前，导致 64/164。v2.1 只在 unstable tail 内使用 Top-K，恢复到 71/164；这验证了一个普适规律：**历史候选更适合做局部否决/尾部排序，不应覆盖当前步已经成熟的置信度顺序。**

### 6.4 为什么 top-1 仍有动机，但不能当答案

上一轮 top-1 不是正确答案假设，而是一个“在旧条件下的响应探针”。当新 token 写入后，当前 top-1 是否变化，测量的是条件是否真正改变了局部决策。问题在于二值变化丢失了改变幅度，因此新增但尚未正式评测的 `revision_margin_fast` 使用：

```text
log q_current(current_top1) - log q_current(previous_top1)
```

它只重排 parent 的 unstable tail，不新增阈值，也不把旧 top-1 当 gold。

## 7. 已尝试、已实现和被淘汰的方法

| 方法族 | 实现状态 | 正式结果/决定 |
| --- | --- | --- |
| Attention stability | 对称 attention + 相邻 top-1 maturity | 当前精度 parent；HE 71/164 |
| Directed attention | 保留 `A[target,source]` 方向 | HE 71/164；MBPP 123/500，精度优先最好 |
| symmetric_fast / directed_fast | stable-conflict pruning + 固定预算填充 | HE128 有增益；64 步崩塌；MATH/GSM 不泛化 |
| Candidate memory | Top-8 + OTHER、稀疏 JSD、overlap、entropy | 71/164，无额外收益，状态/日志明显更重 |
| Candidate frontier | 只存高置信前沿 | 67/164；真正触发边界事件仅 2 次，淘汰 |
| STCC distribution response | JSD class、streak、横向边、额外提交 | HE 47/164、46/164，明显负向，取消整队列 |
| Retention v2 | 全局 Top-K overlap 排序 | 64/164，负向 |
| Retention v2.1 | 只在 unstable tail 用 Top-K | 71/164，恢复 parent；fast 47/164 < 52/164 |
| Response credit | 只在强条件到达且 top-1 保持时累积 credit | 已实现并测试，尚未正式跑 |
| Revision margin | 新 top-1 对旧 top-1 的当前步 log-prob margin | 已实现并测试，尚未正式跑 |
| Draft exchange | anchor/explorer 两轨迹；保留全稿一致骨架，分歧位重新 mask/refine | 已实现并测试，尚未正式跑 |
| Differential execution selector | 只使用 prompt 可见示例、确定性 probes 和行为簇，不接触 hidden tests | 已实现并测试，尚未正式跑 |
| Flip-gated BO2 | 高 flip 时再生成 | temperature=0 时 26/26 完全相同，增益 0 |
| 轨迹/RCR/lateral 早期版本 | 历史最高置信、轨迹几何、横向响应 | GSM8K 50 题从 62% 降到 58%，旧小样本协议 |

### 7.1 尚未正式运行的新方法，不得提前宣称提升

`response_credit_fast`、`revision_margin_fast` 和 `draft_exchange(_exec)` 已有实现、selector tests、evaluator tests 和 sample-gate queue，但旧监控任务已按要求取消，且新双卡节点上没有启动正式生成。因此它们只能列为“实现完成，效果未知”。

## 8. 旧协议探索结果（非当前正式 benchmark）

这些结果解释研究路线，但不能与上面的 full audited LocalLeap 表混合：

| 实验 | 数据/规模 | 结果 | 结论 |
| --- | --- | --- | --- |
| Round-0 LCR | GSM8K 50 | 31/50=62% | 旧 baseline |
| RCR / Traj / Traj+Lateral | GSM8K 50 | 均 58% | 轨迹硬排序伤害 baseline |
| Gold-ever-argmax ceiling | GSM8K 19 个错题 | 0% | 错误通常是稳定错，不是中途曾对 |
| Gold-ever-argmax ceiling | MATH 26 个错题 | 0% | 数学任务同样缺少可直接恢复候选 |
| Flip diagnostic | GSM8K 50 | 错题答案区 4.27 vs 对题 2.51 | flip 有检测性，不等于纠错性 |
| Flip-gated BO2 | GSM8K 50 | 31/50，gain 0 | T=0 无轨迹多样性 |

## 9. 与近期 training-free / test-time 方法的论文报告值对比

以下全部是论文或官方仓库报告值，不是本地复现；不同模型、prompt、temperature、few-shot、NFE 和 evaluator 的数字不能直接排序。

### 9.1 与 Order-Token Search（OTS）的近配置对齐

OTS 使用 LLaDA、block 32，并在 generation length 256 / 128 diffusion steps 下报告：

| 方法 | GSM8K | MATH-500 | HumanEval |
| --- | ---: | ---: | ---: |
| OTS 论文 LLaDA low-confidence | 76.7 | 32.4 | 26.2 |
| OTS | 79.8 | 36.0 | 34.2 |
| 本地 Original LLaDA 128-step | 68.61 | 30.4 | 25.61 |
| 本地 symmetric_fast 128-step | 68.31 | 30.0 | 31.71 |

本地 HumanEval baseline 与 OTS baseline 接近，但 GSM8K 差距很大，表明 prompt/evaluator/temperature 等仍未完全对齐。OTS 用 beam search 同时探索 order 与 token；本地方法只改提交顺序，不扩展 token 轨迹，算力更接近单轨迹。

来源：[Order-Token Search, arXiv:2601.20339](https://arxiv.org/abs/2601.20339)。

### 9.2 与 Prism 的 accuracy–NFE 曲线

Prism 在 LLaDA-8B-Instruct 上报告：

| 方法 | GSM8K (NFE) | MATH-500 (NFE) | HumanEval (NFE) | MBPP (NFE) |
| --- | ---: | ---: | ---: | ---: |
| N=1 | 67.58 (256) | 26.40 (256) | 54.88 (512) | 21.80 (512) |
| Prism K=2 | 74.24 (283) | 30.16 (334) | 71.34 (549) | 29.40 (561) |
| Prism K=4 | 75.30 (509) | 37.70 (622) | 76.19 (1133) | 32.40 (1196) |
| Prism K=8 | 85.30 (1048) | 42.80 (1304) | 79.27 (2480) | 38.20 (2576) |

Prism 是多轨迹、分层剪枝、partial remasking 和 self-verification 的 test-time scaling；K>1 不能与单轨迹 128-step 方法只比准确率。它最值得借鉴的是“只在中前期分配额外轨迹，并用验证信号淘汰”，而不是简单增加并行候选数。

来源：[Prism, arXiv:2602.01842](https://arxiv.org/abs/2602.01842)，[official repository](https://github.com/viiika/Prism)。

### 9.3 与 SOAR 的质量–速度协议

SOAR 的 LLaDA-8B-Base（不是 Instruct）论文行：

| 方法 | HE 256/512 | MBPP 256/512 | GSM8K 256/512 | 平均速度 |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 70.9 | 1.00× reference |
| Adaptive parallel | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 71.0 | 2.19× |
| SOAR | 32.9 / 39.0 | 40.8 / 39.4 | 71.3 / 71.5 | 1.62× |

SOAR 在低置信时短暂扩 beam，高置信时收缩并并行提交。其绝对准确率不能与本地 Instruct 直接比，但“每个结果同时报告质量与真实速度”的协议应保留。

来源：[SOAR, arXiv:2602.10953](https://arxiv.org/abs/2602.10953)，[official repository](https://github.com/duterscmy/SOAR)。

### 9.4 FiRe / full-draft refinement

FiRe 官方仓库的附加实验报告，HumanEval、64 fill/refine budget 下：

| 生成方式 | baseline | blockwise refine | full-draft refine |
| --- | ---: | ---: | ---: |
| block length 16 | 11.0 | 9.8 | 21.3 |
| block length 32 | 13.4 | 15.2 | 20.1 |

仓库还报告 Fast-dLLM + FiRe 在 GSM8K 64 题、5-shot 上：block16 从 73.44 提升到 81.25；block32 从 68.75 提升到 71.88。这些是仓库附加实验、子集设置，不是本地复现。

它支持“先完成全稿，再利用全局双向条件重修”的方向；本地 draft exchange 只 remask 两条本地策略不一致的位置，不复制 FiRe 的训练、rho 分配或统一更新规则。

来源：[FiRe official code repository](https://github.com/runchu-tian/tolerator)。

### 9.5 Diffusion in Diffusion 与 DiffCodeGen

Diffusion in Diffusion 是 draft-then-refine，但主要在 OpenWebText 语言建模上报告，不与本地四个 benchmark 直接重叠；其方法包含 mix-scale training，因此也不属于当前纯 training-free 约束。

DiffCodeGen 在 LiveCodeBench 上使用 18 个候选的 coverage-guided differential analysis。论文报告 Qwen2.5-Coder-Instruct-7B 从 29.4 提升到 38.9，GPT-4o-mini 从 40.9 提升到 54.4；其 selection 不调用额外 LLM，但需要 fuzzing 和多候选执行。本地 differential selector 只作为 draft-exchange 的无 hidden-test selector，候选数和 benchmark 均不同。

来源：[Diffusion in Diffusion, arXiv:2601.13599](https://arxiv.org/abs/2601.13599)，[DiffCodeGen, arXiv:2605.20473](https://arxiv.org/abs/2605.20473)。

## 10. 双卡调整建议与执行策略

单个 LLaDA 8B attention-trace 进程的已观测峰值约为 16.4 GB allocated、17.9 GB reserved。两份模型放在同一张 32.8 GB 卡上有 OOM 风险；每张卡各跑一个独立 evaluator 最稳妥。

当前不建议把单样本推理改成 DDP：模型可完整放入单卡，DDP不会拆分单条生成；未经验证的 tensor parallel 还可能破坏自定义 attention 输出与 trace hooks。双卡的首选利用方式是：

1. GPU0、GPU1 各运行一个独立且配置冻结的 benchmark arm；
2. 同一 paired gate 的 baseline/parent 或两个候选并行，全部完成后再做 CPU audit；
3. 每个 arm 使用独立结果目录与日志，manifest 写入加锁；
4. source hash 在一整个 wave 开始前冻结；wave 内禁止修改生成代码；
5. expensive draft exchange 后续可把 anchor/explorer 分到两卡，但 repair 必须在两者完成后运行；
6. 代码执行评测可能成为 CPU 瓶颈，HumanEval/MBPP 两个 GPU 生成结束后应限制并行 executor 数量。

在取消旧任务后，新节点目前没有运行中的 evaluator。下一队列应先运行 2-example 双卡 smoke，然后用 HumanEval 32/64、MBPP 100、MATH-500 100、GSM8K 128 的 sample gates；只有严格超过 `symmetric_fast` parent 的新方法才进入 full run。

## 11. 正式结果与代码索引

主要结果根目录：

```text
/root/autodl-tmp/LocalLeap/llada/results/attention_stability/
/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/
/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/
/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/best_symmetric_long_20260716_v2/
```

核心实现与审计代码：

```text
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/generate.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/eval_llada.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/differential_selector.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/audit_attention_stability.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/audit_lm_eval_task.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/compare_paired_task_runs.py
/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability/select_queue_profile.py
```

## 12. 最终判断

目前最值得继续的不是重新堆更多 Top-K/JSD 阈值，而是保留已验证的 `symmetric_fast tau=.004` 横向约束，只在 unstable tail 上测试一个纵向变化，并用双卡做严格 sample-gate：

- 单轨迹候选：`revision_margin_fast`，因为它不把旧 top-1 当答案，只测新条件对旧判断的替代强度；
- test-time scaling 候选：两轨迹 disagreement skeleton + 局部 repair，但必须完整报告总 NFE 与 wall time；
- 代码选择：只用公开 prompt 行为，不允许 benchmark hidden tests、gold 或 canonical solution 进入 selector；
- 泛化门槛：HumanEval 上赢 parent 只是开发信号，至少还要在 MBPP 或一个数学集上不退化，才能升级为新 parent。

在现有正式证据下，`symmetric_fast` 是最合适的开发 parent，但原始 LLaDA 始终是论文表中的正式 baseline。
