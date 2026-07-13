# 0712grokproblem — 问题清单（代码静态审计）

> 日期：2026-07-12  
> 审计范围：仓库**最新代码**（`src/`、`scripts/`、`configs/`、相关 `tests/`）  
> 审计方式：静态阅读 + 本地复现提取/ceiling 公式边界（未重跑 GPU 评测）  
> 关联文档：`cursorfeedback/研究现状完整文档.md`

本文汇总本轮检查中搜集到的全部问题。命名统一为 **0712grokproblem**。

---

## 0. 如何读「很多正确率都是 0」

文档与结果里出现的「0」**不是同一类指标**：

| 符号 | 含义 | 是否等于任务失败 |
|------|------|------------------|
| `accuracy` / pass@1 | 任务正确率 | 是任务结果 |
| `ceiling_ratio` | 错题中「中间步曾出现 gold」的比例 | **不是**任务准确率 |
| Round0 `mbpp_lcr=0.0` | MBPP 历史跑数 | 历史配置/抽取问题，见 §2 |

**当前 GSM8K / MATH 任务准确率并非全 0**（约 62% / 48%）。大量「0」主要来自 **ceiling 指标** 与 **早期 MBPP**。其中 MATH ceiling 在最新实现下几乎被公式保证为 0（见 P1）。

---

## 1. 严重 / 会扭曲研究结论

### P1 — MATH ceiling 分母与阈值几乎必然给出 0%

- **位置**：`scripts/run_math_ceiling.py`
- **现象**：`ever_correct_count` 最多约等于 `len(gold_tokens)`，却除以固定窗口长度 30；再要求 `ever_correct_ratio > 0.3`。
- **后果**：短答案（常见 1～8 token）最大 ratio ≤ `8/30 ≈ 0.267 < 0.3`，**ceiling 几乎不可能触发**。`results/math_baseline/math_ceiling.json` 中 `ceiling_ratio=0.0` **不能直接当作「stable wrong」铁证**。
- **建议**：分母改为 `max(len(gold_tokens), 1)`，或改为「任一 gold token 曾在答案区 argmax」的二值 ceiling。

### P2 — HumanEval prompt 重复拼接函数签名

- **位置**：`scripts/run_humaneval_ceiling.py`  
  `prompt = humaneval_prompt(item) + item["prompt"]`  
  而 `src/datasets_v2.humaneval_prompt` **已经包含** `item["prompt"]`。
- **后果**：签名出现两次，污染生成条件；连带破坏 ceiling 位置对齐（见 P3）。
- **建议**：只保留 `humaneval_prompt(item)`，或只喂 `item["prompt"]` 的一种写法。

### P3 — HumanEval ceiling 与 `canonical_solution` 硬对齐错误

- **位置**：`scripts/run_humaneval_ceiling.py`  
  用 `gold_tokens[rel]` 对齐 `prompt_len + rel`。
- **问题**：chat 模板 +（可能重复的）函数签名已在 prompt 内；生成区往往是 continuation/body，而 `canonical_solution` 通常是相对签名的 body 全文。硬对齐会**系统低估** `ever_correct_ratio`。
- **建议**：相对 body 起点对齐，或在解码文本空间做子串/token 序列匹配，再聚合为 ceiling。

### P4 — MATH `\boxed{...}` 嵌套括号正则截断

- **位置**：`src/datasets_v2.py` → `extract_math_answer` / `check_math_answer`  
  模式：`\\boxed\{([^}]+)\}`
- **复现**：`\boxed{\frac{1}{2}}` → 抽成 `\frac{1`。
- **后果**：分数/嵌套公式答案假阴性，压低 MATH accuracy（不仅是 ceiling）。
- **建议**：括号平衡解析，或使用已知 MATH 规范化库。

---

## 2. 中等 / 会造成误杀或历史假 0

### P5 — Round0 MBPP accuracy=0%（历史；config 已改）

- **产物**：`results/round0_baseline/mbpp_lcr.json`（`accuracy=0.0`，`nfe=64`）
- **当时解读**：`steps` 相对 `gen_length` 过短 / 残留 mask 等配置坑。
- **当前代码**：`configs/default.yaml` 已设 `mbpp_steps=mbpp_gen_length=mbpp_block_length=256`。
- **注意**：历史 0% **不能**代表当前 config 下的 MBPP 水平；需用新配置重跑才可引用。

### P6 — MBPP / HumanEval 执行器吞异常、无超时（MBPP）

- **位置**：`src/datasets.py` → `run_mbpp_tests`：`except Exception: return False`，无 timeout。  
  HumanEval 有 subprocess timeout，但仍粗暴按 returncode 判 fail。
- **后果**：语法错、超时挂起、环境错全部记为 incorrect，难区分「模型错」与「评测错」。

### P7 — 代码抽取对「无 def / 未闭合 fence」脆弱

- **位置**：`extract_code_block`（MBPP）、`run_humaneval_ceiling.extract_code`（HumanEval）
- **现象**：依赖 markdown fence 或行首 `def `/`import`；仅缩进 body、或自然语言包裹时，易抽成整段噪声 → 执行失败。
- **后果**：coding pass@1 可能被系统性低估。

### P8 — MATH 无 box 时的数字 fallback 取「全文最后一个数」

- **位置**：`check_math_answer` 数值分支
- **现象**：若模型在最终答案后又写出其他数字，会拿错数比较。
- **后果**：假阴性（或偶然假阳性）。

---

## 3. 已修复（保留备忘，避免回归）

### F1 — top-k 距离把概率值当 token id（已修）

- **位置**：`src/distribution.py` → `_topk_l1` / `kl_divergence`
- **曾导致**：横向耦合≈0 的假象（Round1 稀疏度 ~99.98% 不可信）
- **回归**：`tests/test_distribution.py`

### F2 — `extract_number` 句末句号（已修）

- **曾导致**：`694.` ≠ `694` 假阴性
- **当前**：`-?\d+(?:\.\d+)?` 可正确抽出 `694` / `3.14`
- **回归**：同测试文件

### F3 — 多 block 选择未限制在活跃块（已修）

- **位置**：`src/samplers.py`  
  `selectable_positions = [p for p in masked if block_start <= p < block_end]`，并有 assert
- **说明**：单块（`block_length == gen_length`）历史结果不受影响；分块结果依赖此修复

---

## 4. 研究结论层面的「0」（不一定是代码 bug）

| 项 | 状态 | 说明 |
|----|------|------|
| GSM8K-50 ceiling ≈ 0% | 更像观测结论 | Idea Tree：19 道错题答案区从未出现 gold argmax；ceiling 脚本已不在当前仓库，无法用最新代码复跑 |
| MATH ceiling = 0% | **结论不可靠** | 见 P1；需修公式后重测再写进论文 |
| HumanEval ceiling 低 | **结论不可靠** | 见 P2/P3；早期 6.2% 仅供参考 |
| Flip-gated BO2 增益 0 | 方法结果 | T=0 确定性再生，非评测 bug |
| Round2 traj 伤基线 62→58 | 方法结果 | 诊断量当排序量 |

---

## 5. 建议修复优先级

| 优先级 | ID | 动作 |
|--------|-----|------|
| P0 | P1 | 修 MATH ceiling 分母/阈值，重跑后再解释 H4 |
| P0 | P2+P3 | 修 HumanEval prompt 与对齐，再跑 Node 5 |
| P1 | P4 | 修 MATH boxed 嵌套解析 |
| P2 | P6+P7 | 加强 coding 抽取与执行诊断日志 |
| P2 | P5 | 用当前 mbpp_* 配置重跑 MBPP 基线 |
| — | F1–F3 | 保持回归测试，禁止回退 |

---

## 6. 与完整文档的交叉引用

- 算法细节（含公式）：`cursorfeedback/研究现状完整文档.md` §2  
- 本清单不替代实验结果；修 bug 后应以新 `results/` 为准更新 H4 / HumanEval 表述。

---

*0712grokproblem — Cursor Grok 代码审计问题汇总。*
