#!/usr/bin/env python3
"""Write a contiguous absolute-index slice without overwriting outputs."""

import argparse
import json
from pathlib import Path


def slice_records(records, start, end):
    selected = [
        record for record in records if start <= int(record["absolute_index"]) < end
    ]
    indices = sorted(int(record["absolute_index"]) for record in selected)
    expected = list(range(start, end))
    if indices != expected:
        raise ValueError(f"slice indices mismatch: got={indices} expected={expected}")
    return sorted(selected, key=lambda record: int(record["absolute_index"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = slice_records(records, args.start, args.end)
    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
