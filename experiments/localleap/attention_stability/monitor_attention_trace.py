import json
import math
import sys


records = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))

ids = [record["task_id"] for record in records]
indices = [int(record["absolute_index"]) for record in records]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate task ids")
if indices != list(range(len(indices))):
    raise SystemExit(f"non-contiguous absolute indices: tail={indices[-5:]}")
if any(int(record["nfe"]) != 256 for record in records):
    raise SystemExit("NFE differs from fair baseline budget")

diagnostics = [record["decode_diagnostics"] for record in records]
numeric = [
    value
    for item in diagnostics
    for value in (item["dependency_max"], item["dependency_mean"])
]
if any(not math.isfinite(value) for value in numeric):
    raise SystemExit("non-finite dependency diagnostic")
if any(item["underfilled_steps"] != 0 for item in diagnostics):
    raise SystemExit("unexpected underfilled step with b_t=1")
if any(item["rejected_pairs"] != 0 for item in diagnostics):
    raise SystemExit("unexpected same-batch rejection with b_t=1")

summary = {
    "records": len(records),
    "latest_task_id": ids[-1] if ids else None,
    "nfe_min": min((record["nfe"] for record in records), default=None),
    "nfe_max": max((record["nfe"] for record in records), default=None),
    "unstable_candidates_total": sum(item["unstable_candidates"] for item in diagnostics),
    "unstable_candidates_mean": (
        sum(item["unstable_candidates"] for item in diagnostics) / len(diagnostics)
        if diagnostics else 0
    ),
    "dependency_mean": (
        sum(item["dependency_mean"] for item in diagnostics) / len(diagnostics)
        if diagnostics else None
    ),
    "dependency_max": max((item["dependency_max"] for item in diagnostics), default=None),
}
print(json.dumps(summary, sort_keys=True))
