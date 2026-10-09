"""Merges a sharded run's results directories back into one tree.

asv rewrites the whole ``<machine>/<commit>-<env>.json`` at the end of a run,
so each shard writes its own directory and they are combined here. Shards
split by whole benchmark, so every row belongs to one shard and the merge is a
union.

Usage::

    python -m benchmarks.helpers._merge --out results results.shard*
"""

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]

# Every shard's benchmarks.json lists the whole suite (asv saves all it
# discovers, not just what ``--bench`` selects), and machine.json is pinned to
# one name, so any shard's copy will do.
_WHOLE_FILES = frozenset({"benchmarks.json", "machine.json"})


def merge(shard_dirs, out_dir):
    """Merges ``shard_dirs`` into ``out_dir``; returns ``{relative path: data}``."""
    merged = {}
    for shard_dir in map(Path, shard_dirs):
        for path in sorted(shard_dir.rglob("*.json")):
            rel = path.relative_to(shard_dir)
            data = json.loads(path.read_text())
            if rel not in merged or path.name in _WHOLE_FILES:
                merged[rel] = data
                continue
            into = merged[rel]
            into["results"].update(data["results"])
            # ``<build>`` and ``<setup_cache ...>``, which every shard pays; the
            # max is what a single run would report.
            for key, value in data.get("durations", {}).items():
                into["durations"][key] = max(into["durations"].get(key, 0.0), value)
    for rel, data in merged.items():
        target = Path(out_dir) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._merge",
        description="Merge a sharded run's results directories into one tree.",
    )
    parser.add_argument("shards", nargs="+", help="Shard results directories to merge.")
    parser.add_argument(
        "--out",
        default=str(BENCHMARK_DIR / "results"),
        help="Directory to write the merged tree to (default benchmarks/results).",
    )
    args = parser.parse_args(argv)
    merged = merge(args.shards, args.out)
    if not merged:
        parser.error(f"no results under {', '.join(args.shards)}")
    for rel, data in sorted(merged.items()):
        if "results" in data:
            print(f"{rel}: {len(data['results'])} benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
