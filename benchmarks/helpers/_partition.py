"""Splits the suite into shards of roughly equal cost, one per runner.

asv's ``--parallel`` only parallelizes environment builds, and ``time_*``
results need an uncontended machine, so each shard gets its own runner.

Shards split by whole class: a class's benchmarks share the kernels its first
one compiles, so splitting it makes every shard recompile (``FaceBounds``: 59s
in one shard, 221s across four). Numba's ``cache=True`` does not help, as asv
reinstalls the wheel per commit. Whole benchmarks also leave every results row
owned by one shard, so :mod:`benchmarks.helpers._merge` can merge by union.

Each shard writes to its own ``results_dir``, since asv rewrites the whole
per-commit results file at the end of a run. Weights are the ``duration`` asv
records per benchmark; benchmarks without one get the median.

Usage::

    python -m benchmarks.helpers._partition --shards 4
    asv run $(python -m benchmarks.helpers._partition --shards 4 --shard 0 \
        --config asv.conf.hpc.json --asv-args)

    # Thread sweep over the whole suite (add ``--bench`` to narrow it).
    for n in 1 2 4 8; do
        asv run $(python -m benchmarks.helpers._partition --shards 1 --shard 0 \
            --config asv.conf.hpc.json --env NUMBA_NUM_THREADS=$n --asv-args)
    done
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

__all__ = [
    "bench_regexes",
    "load_benchmarks",
    "load_weights",
    "plan",
    "shard_results_dir",
    "write_shard_config",
]

BENCHMARK_DIR = Path(__file__).resolve().parents[1]

# ``setup_cache`` keys cheap enough to repeat per shard (``prime()`` is a stat
# per file once warm). Other groups would rerun in every shard, so stay together.
SPLITTABLE_PREFIX = "helpers._fixtures:"

_SKIP_FILES = frozenset({"machine.json", "benchmarks.json"})


def _splittable(setup_cache_key):
    """Whether benchmarks sharing this ``setup_cache_key`` may span shards."""
    return setup_cache_key is None or str(setup_cache_key).startswith(SPLITTABLE_PREFIX)


def _owner(name):
    """The class -- or the module, for a bare function -- a benchmark belongs to."""
    return name.rsplit(".", 1)[0]


def _units(benchmarks):
    """Benchmarks that must share a shard, as ``{root name: [names]}``.

    Unions same-class benchmarks with those sharing a non-splittable
    ``setup_cache``. Roots are the smallest member name, so grouping is
    deterministic.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for name, benchmark in benchmarks.items():
        find(name)
        groups.setdefault(("class", _owner(name)), []).append(name)
        key = benchmark.get("setup_cache_key")
        if not _splittable(key):
            groups.setdefault(("setup_cache", key), []).append(name)
    for members in groups.values():
        for other in members[1:]:
            union(members[0], other)

    units = {}
    for name in benchmarks:
        units.setdefault(find(name), []).append(name)
    return units


def load_benchmarks(results_dir):
    """The discovered benchmarks, as asv wrote them to ``benchmarks.json``."""
    path = Path(results_dir) / "benchmarks.json"
    with open(path) as handle:
        discovered = json.load(handle)
    # asv stores its own format version alongside the benchmarks.
    return {name: value for name, value in discovered.items() if name != "version"}


def load_weights(results_dirs):
    """Mean recorded duration per benchmark, in seconds.

    Mean, not latest: a cold numba cache can make one run hundreds of times
    slower than warm, and a single outlier would skew the plan.
    """
    samples = {}
    for results_dir in results_dirs:
        root = Path(results_dir)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.json")):
            if path.name in _SKIP_FILES:
                continue
            try:
                with open(path) as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            columns = data.get("result_columns") or []
            if "duration" not in columns:
                continue
            index = columns.index("duration")
            for name, row in (data.get("results") or {}).items():
                if len(row) <= index or row[index] is None:
                    continue
                samples.setdefault(name, []).append(float(row[index]))
    return {name: statistics.fmean(values) for name, values in samples.items()}


def plan(benchmarks, n_shards, weights=None):
    """Partitions ``benchmarks`` into ``n_shards`` lists of names.

    Greedy longest-first (LPT): within about 1% of optimal on this suite and
    easy to debug. Ties break by name, so a shard can compute its own membership.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be at least 1, got {n_shards}")
    weights = dict(weights or {})
    known = [value for value in weights.values() if value > 0]
    default = statistics.median(known) if known else 1.0

    units = _units(benchmarks)

    costs = {
        unit: sum(weights.get(name, default) for name in names)
        for unit, names in units.items()
    }

    shards = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for unit in sorted(units, key=lambda u: (-costs[u], u)):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        shards[target].extend(sorted(units[unit]))
        loads[target] += costs[unit]
    return shards


def bench_regexes(names):
    """``--bench`` patterns selecting exactly ``names``.

    asv matches parameterized benchmarks as ``name(p0, p1)``, so the trailing
    group admits ``(`` as well as end-of-string; ``^name$`` would match none.
    """
    return [f"^{re.escape(name)}($|\\()" for name in names]


def shard_config_path(base_config, shard):
    """Where shard ``shard``'s generated config goes, beside ``base_config``."""
    base = Path(base_config)
    return base.with_name(f"{base.stem}.shard{shard}{base.suffix}")


def shard_results_dir(results_dir, shard):
    """Where shard ``shard`` writes, given the run's ordinary ``results_dir``."""
    return f"{results_dir}.shard{shard}"


def write_shard_config(base_config, out_path, shard, env=None):
    """Copies ``base_config`` with the shard's ``results_dir``; returns that dir.

    ``env`` overrides ``env_nobuild`` variables. Overrides land in the
    environment name, so a sweep's runs do not overwrite one another.
    """
    # asv configs are JSON with JS comments; lazy so the module runs without asv.
    from asv import util

    config = util.load_json(str(base_config), js_comments=True)
    results_dir = shard_results_dir(config.get("results_dir", "results"), shard)
    config["results_dir"] = results_dir
    if env:
        matrix = config.setdefault("matrix", {}).setdefault("env_nobuild", {})
        # Several values would each be an environment running the whole suite.
        matrix.update({key: [value] for key, value in env.items()})
    with open(out_path, "w") as handle:
        json.dump(config, handle, indent=4)
    return results_dir


def _report(benchmarks, shards, weights):
    known = [value for value in weights.values() if value > 0]
    default = statistics.median(known) if known else 1.0
    total = sum(weights.get(name, default) for name in benchmarks)
    print(
        f"{len(benchmarks)} benchmarks, {len(weights)} with recorded durations, "
        f"{total / 60:.1f} min of work; median fallback {default:.1f}s"
    )
    loads = [sum(weights.get(name, default) for name in shard) for shard in shards]
    ideal = total / len(shards) if shards else 0.0
    for index, (shard, load) in enumerate(zip(shards, loads)):
        drift = 100 * (load - ideal) / ideal if ideal else 0.0
        print(f"  shard {index}: {len(shard):3} benchmarks  {load / 60:5.1f} min  {drift:+5.1f}%")
    if loads and ideal:
        print(
            f"  slowest shard {max(loads) / 60:.1f} min against an ideal "
            f"{ideal / 60:.1f}; speedup {total / max(loads):.2f}x of a possible {len(shards)}x"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._partition",
        description="Split the benchmark suite into shards of roughly equal cost.",
    )
    parser.add_argument("--shards", type=int, default=4, help="Number of shards (default 4).")
    parser.add_argument(
        "--shard", type=int, default=None, help="Report only this shard, by index."
    )
    parser.add_argument(
        "--results",
        action="append",
        default=None,
        help="Results directory to read (repeatable; default benchmarks/results).",
    )
    parser.add_argument(
        "--bench-args",
        action="store_true",
        help="Print the shard's --bench arguments instead of a report.",
    )
    parser.add_argument(
        "--config",
        default=str(BENCHMARK_DIR / "asv.conf.json"),
        help="Base asv config for --config-out (default benchmarks/asv.conf.json).",
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Where --asv-args writes the shard config (default: beside --config).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an env_nobuild variable, e.g. NUMBA_NUM_THREADS=4 (repeatable).",
    )
    parser.add_argument(
        "--asv-args",
        action="store_true",
        help="Write this shard's config and print its asv run arguments. Needs --shard.",
    )
    args = parser.parse_args(argv)

    results_dirs = args.results or [str(BENCHMARK_DIR / "results")]
    benchmarks = load_benchmarks(results_dirs[0])
    weights = load_weights(results_dirs)
    shards = plan(benchmarks, args.shards, weights)

    if args.bench_args or args.asv_args:
        if args.shard is None:
            parser.error("--bench-args and --asv-args need --shard")
        if args.asv_args:
            # Emitted with ``--bench`` so a shard can't pair its benchmarks with
            # another shard's results dir.
            env = {}
            for entry in args.env:
                key, sep, value = entry.partition("=")
                if not sep:
                    parser.error(f"--env wants KEY=VALUE, got {entry!r}")
                env[key] = value
            config_out = args.config_out or shard_config_path(args.config, args.shard)
            write_shard_config(args.config, config_out, args.shard, env)
            print("--config", config_out)
        for pattern in bench_regexes(shards[args.shard]):
            print("--bench", pattern)
        return 0

    if args.shard is None:
        _report(benchmarks, shards, weights)
    else:
        for name in shards[args.shard]:
            print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
