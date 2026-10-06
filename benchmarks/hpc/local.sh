#!/usr/bin/env bash
# Runs the sharded suite concurrently on the current node (e.g. a ``qsub -I``
# session), each shard pinned to its own physical cores. Shared memory bandwidth
# and cache add noise: ``asv compare`` ratios stay usable, but take absolute
# timings from one-shard-per-node (``submit.sh``).
#
# Usage, from the repository root:
#
#   ./benchmarks/hpc/local.sh
#   SHARDS=8 THREADS=4 ./benchmarks/hpc/local.sh
#   REV=main^! ./benchmarks/hpc/local.sh
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHARDS="${SHARDS:-4}"
THREADS="${THREADS:-8}"
export REPO SHARDS THREADS
export REV="${REV:-HEAD^!}"
# Empty on purpose: stage.pbs derives it per cluster.
export ASV_MACHINE="${ASV_MACHINE:-}"
export ASV_ACTIVATE="${ASV_ACTIVATE:-true}"
# Activated here too so the core list below can come from python.
eval "$ASV_ACTIVATE"

STAGE_SCRIPT="$REPO/benchmarks/hpc/stage.pbs"
LOGS="${LOGS:-$REPO/benchmarks/hpc/logs}"
mkdir -p "$LOGS"

# One CPU per physical core this job may use, in socket order. Slicing that
# keeps shards off each other's SMT siblings (~1.28x slower per thread) and,
# where a shard fits, on one socket. Contiguous logical CPU ranges do neither
# when siblings are numbered N and N + cores.
CPUS=($(python - <<'PY'
import os, subprocess
allowed = os.sched_getaffinity(0)
lscpu = subprocess.run(
    ["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True, check=True
).stdout
first = {}
for line in lscpu.splitlines():
    if not line.startswith("#"):
        cpu, core, socket = map(int, line.split(","))
        if cpu in allowed:
            first.setdefault((socket, core), cpu)
print(*(first[key] for key in sorted(first)))
PY
))
PER=$((${#CPUS[@]} / SHARDS))
if [ "$PER" -lt 1 ]; then
    echo "found ${#CPUS[@]} physical cores, too few for $SHARDS shards" >&2
    exit 1
elif [ "$PER" -lt "$((THREADS + 1))" ]; then
    echo "warning: ${#CPUS[@]} cores over $SHARDS shards is $PER each, under the" >&2
    echo "         $THREADS threads a shard wants; they will oversubscribe" >&2
fi

echo "== setup =="
STAGE=setup bash "$STAGE_SCRIPT" 2>&1 | tee "$LOGS/setup.log"

echo "== $SHARDS shards, $PER cores each, $THREADS threads each =="
pids=()
for S in $(seq 0 $((SHARDS - 1))); do
    cpus=$(IFS=,; echo "${CPUS[*]:S * PER:PER}")
    SHARD=$S STAGE=shard taskset -c "$cpus" bash "$STAGE_SCRIPT" >"$LOGS/shard$S.log" 2>&1 &
    pids+=("$!")
    echo "  shard $S -> cpus $cpus, pid $!, log $LOGS/shard$S.log"
done

# A failed shard does not stop the merge (cf. ``afteranyarray`` in submit.sh).
failed=0
for S in $(seq 0 $((SHARDS - 1))); do
    if wait "${pids[$S]}"; then
        echo "  shard $S ok"
    else
        echo "  shard $S FAILED (see $LOGS/shard$S.log)" >&2
        failed=$((failed + 1))
    fi
done

echo "== merge =="
STAGE=merge bash "$STAGE_SCRIPT" 2>&1 | tee "$LOGS/merge.log"
[ "$failed" -eq 0 ] || echo "$failed shard(s) failed; merged what landed" >&2
