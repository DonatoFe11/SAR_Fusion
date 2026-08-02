#!/usr/bin/env python3
"""Report the first divergence among SARFusion reproducibility JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IGNORED_KEYS = {"repetition"}


def load_trace(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalized(record):
    return {key: value for key, value in record.items() if key not in IGNORED_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", help="Two or more reproducibility_trace.jsonl files")
    args = parser.parse_args()
    if len(args.traces) < 2:
        parser.error("at least two traces are required")

    traces = [load_trace(path) for path in args.traces]
    reference_path, reference = args.traces[0], traces[0]
    failed = False
    for candidate_path, candidate in zip(args.traces[1:], traces[1:]):
        limit = min(len(reference), len(candidate))
        mismatch = None
        for index in range(limit):
            if normalized(reference[index]) != normalized(candidate[index]):
                mismatch = index
                break
        if mismatch is None and len(reference) != len(candidate):
            mismatch = limit

        if mismatch is None:
            print(f"MATCH: {reference_path} == {candidate_path} ({len(reference)} events)")
            continue

        failed = True
        print(f"DIVERGENCE at event {mismatch}: {reference_path} != {candidate_path}")
        left = reference[mismatch] if mismatch < len(reference) else "<missing>"
        right = candidate[mismatch] if mismatch < len(candidate) else "<missing>"
        print("  reference:", json.dumps(left, sort_keys=True))
        print("  candidate:", json.dumps(right, sort_keys=True))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
