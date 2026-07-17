#!/usr/bin/env bash
set -uo pipefail
queue_root=$1
queue_id=$(basename "${queue_root}")
run_base=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/${queue_id}
log=${queue_root}/monitor.log

while [[ ! -e "${queue_root}/DONE" && ! -e "${queue_root}/FAILED" ]]; do
  {
    echo "===== $(date --iso-8601=seconds) ====="
    if [[ -s "${queue_root}/controller.pid" ]]; then
      pid=$(cat "${queue_root}/controller.pid")
      ps -p "${pid}" -o pid=,ppid=,stat=,etime=,cmd= || true
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    df -h /root/autodl-tmp | tail -n 1
    pgrep -af "[e]val_llada.py" || true
    [[ -s "${queue_root}/formal_manifest.tsv" ]] && tail -n 12 "${queue_root}/formal_manifest.tsv"
    /root/miniconda3/bin/python - "${run_base}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for trace in sorted(root.glob("*/*/trace/rank_0.jsonl")):
    rows=[]
    try:
        rows=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    except Exception as exc:
        print(f"TRACE_PARSE_ERROR {trace} {exc}")
        continue
    nfes=[]; residual=[]; ids=[]
    for row in rows:
        ids.append(str(row.get("task_id", row.get("stable_task_id", ""))))
        diag=row.get("decode_diagnostics") or {}
        nfes.append(row.get("nfe", diag.get("total_nfe")))
        residual.append(diag.get("residual_mask_count", 0))
    clean=[x for x in nfes if isinstance(x, (int,float))]
    print("TRACE", trace, "records", len(rows), "nfe", (min(clean),max(clean)) if clean else None,
          "residual", sum(int(x or 0) for x in residual), "duplicate_ids", len(ids)-len(set(ids)))
PY
    for file in "${queue_root}"/*.log; do
      [[ -f "${file}" ]] || continue
      tail -n 200 "${file}" | grep -E 'Traceback|CUDA out of memory|OutOfMemory|AssertionError|HARD STOP|FAILED_SOURCE' || true
    done
    sha256sum -c "${queue_root}/FROZEN_SOURCE_SHA256" >/dev/null 2>&1 || echo SOURCE_HASH_DRIFT
  } >>"${log}" 2>&1
  sleep 300
done
echo "===== $(date --iso-8601=seconds) terminal =====" >>"${log}"
