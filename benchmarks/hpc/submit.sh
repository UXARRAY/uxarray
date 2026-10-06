#!/usr/bin/env bash
# Submits a sharded asv run on derecho as three chained PBS jobs: setup (once,
# so shards do not race on the shared filesystem), a job array of shards (one
# node each, so ``time_*`` results are uncontended), and merge. The merge uses
# ``afteranyarray`` so one failed shard does not cost the others' results.
#
# Usage:
#
#   PBS_ACCOUNT=UXXX0001 ./benchmarks/hpc/submit.sh
#   PBS_ACCOUNT=UXXX0001 THREADS=8 ./benchmarks/hpc/submit.sh
#   PBS_ACCOUNT=UXXX0001 SHARDS=8 REV=main^! ./benchmarks/hpc/submit.sh
#
# THREADS overrides the config's NUMBA_NUM_THREADS and is recorded in the
# environment name, so thread counts do not overwrite each other's results.
#
set -euo pipefail

: "${PBS_ACCOUNT:?set PBS_ACCOUNT to your project code, e.g. PBS_ACCOUNT=UXXX0001}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHARDS="${SHARDS:-4}"
# An asv range: ``main^!`` is one commit, ``base..head`` every commit between.
REV="${REV:-HEAD^!}"
QUEUE="${QUEUE:-main}"
CONFIG="${CONFIG:-asv.conf.hpc.json}"
# Empty means stage.pbs derives it from NCAR_HOST or the node name.
ASV_MACHINE="${ASV_MACHINE:-}"
WALLTIME="${WALLTIME:-12:00:00}"
SETUP_WALLTIME="${SETUP_WALLTIME:-02:00:00}"
# On scratch so every shard and worktree shares one fixture cache.
CACHE_DIR="${UXARRAY_BENCH_CACHE_DIR:-/glade/derecho/scratch/$USER/uxarray-bench}"
# ``-V`` carries this shell's ``asv`` into the jobs. ASV_ACTIVATE, if set, is
# eval'd once per stage instead.
export ASV_ACTIVATE="${ASV_ACTIVATE:-true}"
command -v asv >/dev/null || [ "$ASV_ACTIVATE" != "true" ] || {
    echo "asv is not on PATH; activate your environment first, or set ASV_ACTIVATE" >&2
    exit 1
}

STAGE_SCRIPT="$REPO/benchmarks/hpc/stage.pbs"
# Via the environment, not ``-v``, whose comma-separated list cannot hold commas.
export REPO SHARDS REV CONFIG ASV_MACHINE
export THREADS="${THREADS:-}"
export UXARRAY_BENCH_CACHE_DIR="$CACHE_DIR"

mkdir -p "$CACHE_DIR"

setup=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-setup \
    -l select=1:ncpus=128 -l walltime="$SETUP_WALLTIME" \
    -V -v "STAGE=setup" "$STAGE_SCRIPT")
echo "setup  $setup"

shards=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-shard \
    -J "0-$((SHARDS - 1))" \
    -l select=1:ncpus=128 -l walltime="$WALLTIME" \
    -W "depend=afterok:$setup" \
    -V -v "STAGE=shard" "$STAGE_SCRIPT")
echo "shards $shards  ($SHARDS of them)"

merge=$(qsub -A "$PBS_ACCOUNT" -q "$QUEUE" -N asv-merge \
    -l select=1:ncpus=1 -l walltime=00:30:00 \
    -W "depend=afteranyarray:$shards" \
    -V -v "STAGE=merge" "$STAGE_SCRIPT")
echo "merge  $merge"
echo
echo "watch with:  qstat -u $USER -t"
echo "results in:  $REPO/benchmarks/results/$ASV_MACHINE/"
