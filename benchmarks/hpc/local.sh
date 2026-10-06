#!/usr/bin/env bash
# Runs the sharded suite concurrently on the current node (e.g. a ``qsub -I``
# session), each shard pinned to its own core slice. Shared memory bandwidth
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
export CONFIG="${CONFIG:-asv.conf.hpc.json}"
# Empty on purpose: stage.pbs derives it per cluster.
export ASV_MACHINE="${ASV_MACHINE:-}"
export ASV_ACTIVATE="${ASV_ACTIVATE:-true}"
# Activated here too so the core count below can come from python.
eval "$ASV_ACTIVATE"

STAGE_SCRIPT="$REPO/benchmarks/hpc/stage.pbs"
LOGS="${LOGS:-$REPO/benchmarks/hpc/logs}"
mkdir -p "$LOGS"

# The affinity mask, not PBS's NCPUS (per-chunk request) or ``nproc`` (honours
# OMP_NUM_THREADS). Override with CORES.
CORES="${CORES:-$(python -c '
import os
try:
    print(len(os.sched_getaffinity(0)))
except AttributeError:   # not Linux
    print(os.cpu_count() or 1)
')}"
if ! [ "$CORES" -ge 1 ] 2>/dev/null; then
    echo "could not work out a core count (got ${CORES:-empty}); set CORES" >&2
    exit 1
fi
PER=$((CORES / SHARDS))
if [ "$PER" -lt "$((THREADS + 1))" ]; then
    echo "warning: $CORES cores over $SHARDS shards is $PER each, under the" >&2
    echo "         $THREADS threads a shard wants; they will oversubscribe" >&2
fi

PIN=""
if command -v taskset >/dev/null; then
    PIN="taskset"
else
    echo "warning: no taskset, shards will not be pinned and will drift across cores" >&2
fi

echo "== setup =="
STAGE=setup bash "$STAGE_SCRIPT" 2>&1 | tee "$LOGS/setup.log"

echo "== $SHARDS shards, $PER cores each, $THREADS threads each =="
pids=()
for S in $(seq 0 $((SHARDS - 1))); do
    lo=$((S * PER))
    hi=$((lo + PER - 1))
    if [ -n "$PIN" ]; then
        SHARD=$S STAGE=shard taskset -c "$lo-$hi" bash "$STAGE_SCRIPT" \
            >"$LOGS/shard$S.log" 2>&1 &
    else
        SHARD=$S STAGE=shard bash "$STAGE_SCRIPT" >"$LOGS/shard$S.log" 2>&1 &
    fi
    pid=$!
    pids+=("$pid")
    echo "  shard $S -> cores $lo-$hi, pid $pid, log $LOGS/shard$S.log"
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
