"""Splits the suite into shards of roughly equal cost, one per runner.

asv's ``--parallel`` only parallelizes environment builds, and ``time_*``
results need an uncontended machine, so each shard gets its own runner.

Shards split by whole class: a class's benchmarks share the kernels its first
one compiles, so splitting it makes every shard recompile (``FaceBounds``: 59s
in one shard, 221s across four). Numba's ``cache=True`` does not help, as asv
reinstalls the wheel per commit. Every ``setup_cache`` but the suite-wide one in
``_fixtures`` (cheap to repeat) lives on a single class, so stays in one shard.

Each shard writes to its own ``results_dir``, since asv rewrites the whole
per-commit results file at the end of a run. Weights are the ``duration`` asv
records per benchmark; benchmarks without one get the median.

Usage::

    python -m benchmarks.helpers._partition --shards 4
    asv run $(python -m benchmarks.helpers._partition --shards 4 --shard 0)

    # Thread sweep over the whole suite (add ``--bench`` to narrow it).
    for n in 1 2 4 8; do
        asv run $(python -m benchmarks.helpers._partition --shards 1 --shard 0 \
            --env NUMBA_NUM_THREADS=$n)
    done
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def load_benchmarks(results_dir):
    """The discovered benchmarks, as asv wrote them to ``benchmarks.json``."""
    discovered = json.loads((Path(results_dir) / "benchmarks.json").read_text())
    # asv stores its own format version alongside the benchmarks.
    discovered.pop("version", None)
    return discovered


def load_weights(results_dirs):
    """Mean recorded duration per benchmark, in seconds.

    Mean, not latest: a cold numba cache can make one run hundreds of times
    slower than warm, and a single outlier would skew the plan.
    """
    samples = {}
    for results_dir in results_dirs:
        for path in sorted(Path(results_dir).glob("*/*.json")):
            data = json.loads(path.read_text())
            columns = data.get("result_columns") or []
            if "duration" not in columns:
                continue
            index = columns.index("duration")
            for name, row in data["results"].items():
                if len(row) > index and row[index] is not None:
                    samples.setdefault(name, []).append(float(row[index]))
    return {name: statistics.fmean(values) for name, values in samples.items()}


def costs(benchmarks, weights):
    """Each benchmark's weight, the median for those without one."""
    known = [value for value in weights.values() if value > 0]
    default = statistics.median(known) if known else 1.0
    return {name: weights.get(name, default) for name in benchmarks}


def plan(benchmarks, n_shards, weights=None):
    """Partitions ``benchmarks`` into ``n_shards`` lists of names.

    Greedy longest-first (LPT): within about 1% of optimal on this suite and
    easy to debug. Ties break by name, so every shard computes the same plan.
    """
    cost = costs(benchmarks, weights or {})
    classes = {}
    for name in sorted(benchmarks):
        # The class -- or the module, for a bare function.
        classes.setdefault(name.rsplit(".", 1)[0], []).append(name)
    class_cost = {owner: sum(cost[name] for name in names) for owner, names in classes.items()}

    shards = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for owner in sorted(classes, key=lambda owner: (-class_cost[owner], owner)):
        target = loads.index(min(loads))
        shards[target].extend(classes[owner])
        loads[target] += class_cost[owner]
    return shards


def write_shard_config(base_config, shard, env):
    """Copies ``base_config`` beside itself with the shard's ``results_dir``.

    ``env`` overrides ``env_nobuild`` variables, which asv folds into the
    environment name, so a sweep's runs do not overwrite one another.
    """
    # asv configs are JSON with JS comments; lazy so the module runs without asv.
    from asv import util

    base = Path(base_config)
    config = util.load_json(str(base), js_comments=True)
    config["results_dir"] = f"{config.get('results_dir', 'results')}.shard{shard}"
    # Several values would each be an environment running the whole suite.
    config["matrix"]["env_nobuild"].update({key: [value] for key, value in env.items()})
    out = base.with_name(f"{base.stem}.shard{shard}{base.suffix}")
    out.write_text(json.dumps(config, indent=4))
    return out


def _report(benchmarks, shards, weights):
    cost = costs(benchmarks, weights)
    total = sum(cost.values())
    loads = [sum(cost[name] for name in shard) for shard in shards]
    ideal = total / len(shards)
    print(f"{len(benchmarks)} benchmarks, {len(weights)} with recorded durations, "
          f"{total / 60:.1f} min of work")
    for index, (shard, load) in enumerate(zip(shards, loads)):
        print(f"  shard {index}: {len(shard):3} benchmarks  {load / 60:5.1f} min  "
              f"{100 * (load - ideal) / ideal:+5.1f}%")
    print(f"  speedup {total / max(loads):.2f}x of a possible {len(shards)}x")
    for name in sorted(cost, key=cost.get, reverse=True)[:15]:
        print(f"  {cost[name]:8.1f}s  {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.helpers._partition",
        description="Split the benchmark suite into shards of roughly equal cost.",
    )
    parser.add_argument("--shards", type=int, default=4, help="Number of shards (default 4).")
    parser.add_argument(
        "--shard", type=int, help="Write this shard's config and print its asv run arguments."
    )
    parser.add_argument(
        "--results",
        action="append",
        help="Results directory to read (repeatable; default benchmarks/results).",
    )
    parser.add_argument(
        "--config",
        default=str(BENCHMARK_DIR / "asv.conf.json"),
        help="Base asv config for --shard (default benchmarks/asv.conf.json).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an env_nobuild variable, e.g. NUMBA_NUM_THREADS=4 (repeatable).",
    )
    args = parser.parse_args(argv)

    results_dirs = args.results or [str(BENCHMARK_DIR / "results")]
    benchmarks = load_benchmarks(results_dirs[0])
    weights = load_weights(results_dirs)
    shards = plan(benchmarks, args.shards, weights)
    if args.shard is None:
        _report(benchmarks, shards, weights)
        return 0

    env = dict(entry.split("=", 1) for entry in args.env)
    # ``--bench`` alongside ``--config``, so a shard can't pair its benchmarks
    # with another shard's results dir. asv matches parameterized benchmarks as
    # ``name(p0, p1)``, hence ``($|\()`` rather than ``$``.
    print("--config", write_shard_config(args.config, args.shard, env))
    for name in shards[args.shard]:
        print("--bench", f"^{re.escape(name)}($|\\()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
