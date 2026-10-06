"""Merges a sharded run's results directories back into one tree.

asv rewrites the whole ``<machine>/<commit>-<env>.json`` at the end of a run,
so shards sharing a directory would overwrite one another; each writes its own
and they are combined here. Shards split by whole benchmark, so every row
belongs to one shard and the merge is a union.

Rows are written in the order a serial run would produce
(:func:`canonical_order`). Idempotent and tolerant of shards not yet landed, so
it can run as jobs come back.

Usage::

    python -m benchmarks.helpers._merge --out results results.shard*
    python -m benchmarks.helpers._merge --out results --quiet results.shard*
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

__all__ = ["canonical_order", "merge", "merge_benchmarks", "merge_result_files"]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]

MACHINE_FILE = "machine.json"
BENCHMARKS_FILE = "benchmarks.json"
_SPECIAL_FILES = frozenset({MACHINE_FILE, BENCHMARKS_FILE})

# asv stores its own format version alongside the data in both files.
_VERSION_KEY = "version"


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _dump(path, data):
    """Writes ``data`` unsorted, like asv's ``write_json(..., compact=True)``.

    Key order is the only record of what ran when.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def canonical_order(benchmarks):
    """Benchmark names in the order an unsharded ``asv run`` would produce them.

    Mirrors ``runner.run_benchmarks``: by name within each ``setup_cache_key``
    group, groups ordered by their first member's name.
    """
    groups = {}
    for name in sorted(benchmarks):
        key = benchmarks[name].get("setup_cache_key")
        groups.setdefault(key, []).append(name)
    return [name for group in groups.values() for name in group]


def merge_benchmarks(shard_dirs):
    """Union of the shards' ``benchmarks.json``.

    Each shard only lists its own benchmarks; ``_partition`` needs the full set
    to plan the next run.
    """
    merged, version = {}, None
    for shard_dir in shard_dirs:
        path = Path(shard_dir) / BENCHMARKS_FILE
        if not path.is_file():
            continue
        data = _load(path)
        version = data.get(_VERSION_KEY, version)
        for name, value in data.items():
            if name != _VERSION_KEY:
                merged[name] = value
    if version is not None:
        merged[_VERSION_KEY] = version
    return merged


def _pick(name, existing, candidate, report):
    """Which of two rows for one benchmark to keep.

    Only reached if shards ran different plans. Prefers a row with a result,
    then the later ``started_at``, so a re-run beats the run it replaced.
    """
    if existing == candidate:
        return existing

    def rank(row):
        return (row.get("result") is not None, row.get("started_at") or 0)

    keep, drop = (candidate, existing) if rank(candidate) > rank(existing) else (existing, candidate)
    report(
        f"{name}: found in more than one shard with different data; keeping the "
        f"row started at {keep.get('started_at')} over {drop.get('started_at')}"
    )
    return keep


def merge_result_files(datas, order, report):
    """One results file from several shards' copies, rows written in ``order``.

    Fields other than ``results`` and ``durations`` are identical across shards,
    so the first shard's are kept.
    """
    merged = dict(datas[0])
    columns = list(merged.get("result_columns") or [])

    rows, durations = {}, {}
    for data in datas:
        # Rows are bare lists, so align each to its own file's columns.
        shard_columns = data.get("result_columns") or columns
        for name, row in (data.get("results") or {}).items():
            values = dict(zip(shard_columns, row))
            rows[name] = (
                _pick(name, rows[name], values, report) if name in rows else values
            )
        # Only ``<build>`` and ``<setup_cache ...>`` entries, which every shard
        # pays; the max is what a single run would report.
        for key, value in (data.get("durations") or {}).items():
            durations[key] = max(durations.get(key, 0.0), float(value))

    known = [name for name in order if name in rows]
    extra = sorted(name for name in rows if name not in set(order))
    if extra:
        report(f"{len(extra)} row(s) not in benchmarks.json, appended: {', '.join(extra[:3])}...")

    results = {}
    for name in known + extra:
        row = [rows[name].get(column) for column in columns]
        # Matches asv, which drops trailing nulls.
        while row and row[-1] is None:
            row.pop()
        results[name] = row

    merged["results"] = results
    merged["durations"] = durations
    return merged


def merge(shard_dirs, out_dir, report=lambda message: None):
    """Merges ``shard_dirs`` into ``out_dir``. Returns a per-file row count."""
    shard_dirs = [Path(d) for d in shard_dirs]
    out_dir = Path(out_dir)
    resolved_out = out_dir.resolve()
    if any(d.resolve() == resolved_out for d in shard_dirs):
        raise ValueError(f"--out {out_dir} is also a shard directory; refusing to merge in place")

    present = [d for d in shard_dirs if d.is_dir()]
    for missing in [d for d in shard_dirs if not d.is_dir()]:
        report(f"{missing}: not there yet, skipped")
    if not present:
        raise ValueError("no shard directories to merge")

    benchmarks = merge_benchmarks(present)
    order = canonical_order({k: v for k, v in benchmarks.items() if k != _VERSION_KEY})
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(benchmarks) > (1 if _VERSION_KEY in benchmarks else 0):
        _dump(out_dir / BENCHMARKS_FILE, benchmarks)

    # Shards of one commit and environment write the same file name.
    groups = {}
    for shard_dir in present:
        for path in sorted(shard_dir.glob("*/*.json")):
            if path.name in _SPECIAL_FILES:
                continue
            groups.setdefault((path.parent.name, path.name), []).append(path)
        for machine_path in sorted(shard_dir.glob(f"*/{MACHINE_FILE}")):
            target = out_dir / machine_path.parent.name / MACHINE_FILE
            if target.is_file() and _load(target) != _load(machine_path):
                report(f"{machine_path}: disagrees with the machine.json already merged")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(machine_path, target)

    counts = {}
    for (machine, filename), paths in sorted(groups.items()):
        merged = merge_result_files(
            [_load(p) for p in paths],
            order,
            lambda message, f=filename: report(f"{f}: {message}"),
        )
        _dump(out_dir / machine / filename, merged)
        counts[f"{machine}/{filename}"] = (len(merged["results"]), len(paths))
    return counts


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
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file notes.")
    args = parser.parse_args(argv)

    def report(message):
        if not args.quiet:
            print(f"  {message}", file=sys.stderr)

    counts = merge(args.shards, args.out, report)
    for name, (rows, shards) in sorted(counts.items()):
        print(f"{name}: {rows} benchmarks from {shards} shard(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
