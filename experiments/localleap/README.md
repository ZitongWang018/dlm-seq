# LocalLeap 复现资产（受保护）

> 远端完整工程：`/root/autodl-tmp/LocalLeap/`（**清理脚本不得删除**）  
> 本目录是配置/脚本镜像 + 评测协议记录，便于在本仓库继续做对照实验。

## 模型与路径

- 模型：`/root/autodl-tmp/model/LLaDA/instruct`
- 评测入口：`llada/eval_llada.py` + `llada/generate.py`（含 `generate_localleap` 别名）
- HumanEval 官方分数：必须跑 `python postprocess_code.py samples_*.jsonl`（勿信 lm_eval 原始 pass@1）

## 论文配置（LLaDA）

| | baseline | LocalLeap |
|--|--|--|
| block_length | 32 | 32 |
| gen_length | 256（HE/MBPP/MATH）；512（GSM8K/IFEval） | 同左 |
| steps | `= gen_length`（HE: 256） | `= length/32`（HE: 8） |
| remasking | low_confidence | low_confidence + κ/τ/W |
| LocalLeap | — | κ=0.9, τ=0.75, W=4 |
| HumanEval fewshot | 0 | 0 |

## 本机已复现（HumanEval 全量 164）

| 方法 | postprocess pass@1 | TPS（单卡） | 备注 |
|--|--:|--:|--|
| baseline | **40.85%** | ~9.7 | 论文 40.24% / 33.81 TPS（8×H800） |
| LocalLeap | **40.24%** | **41.0** | 与论文准确率一致 |

结果目录（远端，勿删）：

- `LocalLeap/llada/results/baseline/humaneval_len256_blen32_0shot/`
- `LocalLeap/llada/results/localleap/anchor0.9_relax0.75_radius4/humaneval_len256_blen32_0shot/`

## 脚本

| 文件 | 用途 |
|--|--|
| `scripts/baseline.sh` / `localleap.sh` | 官方多任务脚本 |
| `scripts/baseline_paper_local.sh` | 本地模型路径的 baseline |
| `scripts/localleap_paper_local.sh` | 本地模型路径的 LocalLeap |
| `scripts/run_localleap_humaneval_only.sh` | 仅 HE 全量 + postprocess |
| `scripts/smoke_local_model.sh` | 冒烟 |
| `generate.py` | 含 LocalLeap 阈值逻辑的生成代码快照 |
| `postprocess_code.py` | HumanEval sanitize 计分 |

## 必要补丁（已在远端 LocalLeap 落地）

1. `generate_localleap = generate` 别名  
2. `eval_llada.py` 增加 `use_cache=False` 参数  
3. 从模型目录拷入 `configuration_llada.py`  
4. 使用 Fast-dLLM 的 `sanitize.py` 做 HumanEval postprocess

## Attention-stability extension (2026-07-13)

The extension under `attention_stability/` implements the dependency-threshold
decoder on top of the exact LLaDA baseline configuration. It averages attention
over all layers and heads for the active 32-token block, uses adjacent-step top-1
changes for temporal maturity, and greedily excludes strongly dependent tokens
from the same transfer set. The probe is disabled unless
`dependency_threshold` is passed, so the original baseline path is unchanged.

Fair HumanEval configuration: local LLaDA-Instruct weights, `gen_length=256`,
`steps=256`, `block_length=32`, zero-shot, temperature 0, low-confidence
remasking, and the same lm-eval seeds and `postprocess_code.py`. With this
configuration, the baseline budget is one token per block step, so same-batch
dependency exclusion is vacuous; only temporal maturity can change ordering.

Full HumanEval result at `dependency_threshold=0.01`:

| Method | sanitize/code_eval pass@1 | Correct | TPS | NFE |
|--|--:|--:|--:|--:|
| Exact baseline | 40.85% | 67/164 | 9.730 | 41984 |
| Attention-stability | 40.24% | 66/164 | 9.439 | 41984 |

Paired outcome: 62 both correct, 93 both wrong, 4 method-only, 5
baseline-only, and 121 changed generations. Exact McNemar p-value is 1.0. This
run is negative evidence: the decoder changed trajectories but did not improve
HumanEval accuracy at tau 0.01.

The authoritative run remains on the remote server at
`LocalLeap/llada/results/attention_stability/tau0.01/full_tau001_20260713/`.
Use `attention_stability/scripts/run_attention_stability_humaneval.sh` to rerun;
it invokes the same dedicated sanitize/code_eval channel and then produces
record-level audit and paired-analysis artifacts.

### Sequential tau sweep and step diagnostics

`attention_stability/scripts/run_attention_stability_sweep.sh` runs thresholds
sequentially and starts the next run only after generation, dedicated
sanitize/code_eval, record audit, and step-diagnostics validation all finish.
The default sweep is `0.005 0.02 0.05`; pass explicit values to override it.
The sweep stops on the first failed run and records transitions in
`results/attention_stability/sweeps/<sweep_id>/manifest.tsv`.

When `dependency_diagnostics_dir` is set, every HumanEval item gets one atomic
`.pt` file using schema `attention_stability_steps_v1`. Each of its 256 records
contains the directional and symmetric 32x32 attention matrices, block input
tokens, current and previous top-1 candidates, confidences, candidate-change and
maturity flags, maximum dependency on the preceding transfer set, ordering,
selected/rejected positions, budget and mask counts, dependency/asymmetry
statistics, and fallback/underfill flags. The run validates all files and writes
`audit/step_diagnostics_summary.json`; the one-item smoke file is about 2.46 MB.
