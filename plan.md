# Plan: 稳健清理（已执行）

## 已确认
- 方案 **2 稳健**
- **绝不删除** LocalLeap：baseline 设置、实验配置、代码、HE 结果（`/root/autodl-tmp/LocalLeap`）

## 已完成
1. 本地/远端 `dlm-seq` results：删除错误 eval 与失败方向产物
2. 保留：`round0` GSM8K LCR、`round_block32_*`
3. 镜像 LocalLeap 脚本 → `experiments/localleap/`（保护后续实验）
4. 更新 `PROJECT_STATE.md` / `MEMORY.md`
5. commit + push（进行中）
