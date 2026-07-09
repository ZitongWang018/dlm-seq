#!/usr/bin/env bash
# =============================================================================
# LLaDA 模型下载脚本（ModelScope）
#
# 目标目录: /root/autodl-tmp/model/LLaDA
# 所有缓存均限制在 /root/autodl-tmp 内，不会写入 /root 或其他路径
#
# 用法:
#   bash scripts/download_llada.sh                    # 默认下载 Instruct
#   bash scripts/download_llada.sh base               # 下载 Base
#   bash scripts/download_llada.sh instruct           # 下载 Instruct
#   bash scripts/download_llada.sh all                # 下载 Base + Instruct
# =============================================================================

set -euo pipefail

# ---------- 路径与缓存（全部在 autodl-tmp 内） ----------
AUTODL_ROOT="/root/autodl-tmp"
MODEL_ROOT="${AUTODL_ROOT}/model/LLaDA"
CACHE_ROOT="${AUTODL_ROOT}/.cache"

export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export MODELSCOPE_HOME="${CACHE_ROOT}/modelscope"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${CACHE_ROOT}/tmp"
export TORCH_HOME="${CACHE_ROOT}/torch"

mkdir -p \
    "${MODEL_ROOT}" \
    "${MODELSCOPE_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

# ---------- 依赖检查 ----------
if ! python3 -c "import modelscope" 2>/dev/null; then
    echo "[INFO] 正在安装 modelscope ..."
    pip install -U modelscope -q
fi

# ---------- 模型映射 ----------
declare -A MODEL_MAP=(
    ["base"]="GSAI-ML/LLaDA-8B-Base"
    ["instruct"]="GSAI-ML/LLaDA-8B-Instruct"
)

download_one() {
    local variant="$1"
    local repo_id="${MODEL_MAP[$variant]}"
    local local_dir="${MODEL_ROOT}/${variant}"

    echo "============================================================"
    echo "  下载: ${repo_id}"
    echo "  保存: ${local_dir}"
    echo "  缓存: ${MODELSCOPE_CACHE}"
    echo "============================================================"

    python3 <<PYEOF
import os
from modelscope import snapshot_download

repo_id = "${repo_id}"
local_dir = "${local_dir}"
cache_dir = os.environ["MODELSCOPE_CACHE"]

# local_dir: 最终模型权重目录
# cache_dir: 下载过程中的临时/缓存文件，同样限制在 autodl-tmp
path = snapshot_download(
    model_id=repo_id,
    cache_dir=cache_dir,
    local_dir=local_dir,
)
print(f"\n[SUCCESS] 模型已保存至: {path}")
PYEOF
}

# ---------- 主逻辑 ----------
VARIANT="${1:-instruct}"

case "${VARIANT}" in
    base)
        download_one "base"
        ;;
    instruct)
        download_one "instruct"
        ;;
    all)
        download_one "base"
        download_one "instruct"
        ;;
    *)
        echo "未知参数: ${VARIANT}"
        echo "可选: base | instruct | all"
        exit 1
        ;;
esac

echo ""
echo "[DONE] 下载完成。"
echo "  模型目录: ${MODEL_ROOT}"
echo "  缓存目录: ${CACHE_ROOT}"
echo ""
echo "加载示例:"
echo '  from transformers import AutoModel, AutoTokenizer'
echo '  import torch'
echo '  model = AutoModel.from_pretrained("/root/autodl-tmp/model/LLaDA/instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)'
echo '  tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model/LLaDA/instruct", trust_remote_code=True)'
