#!/usr/bin/env python3
"""Write the exact stable-id subset named by a reference audit file."""

import argparse
import json
from pathlib import Path


def stable_id(row):
    values = [
        str(row[key])
        for key in ("stable_task_id", "task_id")
        if row.get(key) is not None
    ]
    if not values or len(set(values)) != 1:
        raise ValueError(f"invalid stable identity: {values}")
    return values[0]


def index(rows, label):
    output = {}
    for row in rows:
        identity = stable_id(row)
        if identity in output:
            raise ValueError(f"duplicate {label} identity: {identity}")
        output[identity] = row
    return output


def slice_matching(records, reference):
    source = index(records, "source")
    wanted = index(reference, "reference")
    missing = sorted(set(wanted) - set(source))
    if missing:
        raise ValueError(f"source is missing reference identities: {missing[:5]}")
    return [source[identity] for identity in wanted]


def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    selected = slice_matching(read(args.input), read(args.reference))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({"selected": len(selected), "output": str(output)}))


if __name__ == "__main__":
    main()
