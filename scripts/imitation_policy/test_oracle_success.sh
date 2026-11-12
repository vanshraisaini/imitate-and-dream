#!/bin/bash
# Run the pass@K oracle diagnostic sharded over several worker processes, then analyze.
#
# Usage (from the project root):
#   ./scripts/imitation_poilcy/test_oracle_success.sh [NUM_WORKERS=4] [NUM_SEEDS=100] [RUNS_PER_SEED=10]
#
# Examples:
#   ./scripts/imitation_poilcy/test_oracle_success.sh 4            # 4 workers, 100 seeds, 10 runs per seed
#   ./scripts/imitation_poilcy/test_oracle_success.sh 8 100 20     # 8 workers, 100 seeds, 20 runs per seed

set -u

NUM_WORKERS=${1:-4}
NUM_SEEDS=${2:-100}
RUNS_PER_SEED=${3:-10}

is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

for value in "$NUM_WORKERS" "$NUM_SEEDS" "$RUNS_PER_SEED"; do
    if ! is_pos_int "$value"; then
        echo "Error: all arguments must be positive integers (got '$value')." >&2
        echo "Usage: $0 [NUM_WORKERS=4] [NUM_SEEDS=100] [RUNS_PER_SEED=10]" >&2
        exit 1
    fi
done

if (( NUM_WORKERS > NUM_SEEDS )); then
    echo "Warning: $NUM_WORKERS workers but only $NUM_SEEDS seeds; some workers will have nothing to do." >&2
fi

if command -v nproc >/dev/null 2>&1; then
    CORES=$(nproc)
    if (( NUM_WORKERS > CORES )); then
        echo "Warning: $NUM_WORKERS workers but only $CORES CPU cores; expect little extra speedup." >&2
    fi
fi

echo "Launching $NUM_WORKERS workers: $NUM_SEEDS seeds x $RUNS_PER_SEED runs per seed"

pids=()
# Ctrl-C stops all workers instead of leaving them running in the background.
trap 'echo "Interrupted, stopping workers..." >&2; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

for ((w = 0; w < NUM_WORKERS; w++)); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.imitation_poilcy.test_oracle_success run \
        --worker "$w" --num-workers "$NUM_WORKERS" \
        --num-seeds "$NUM_SEEDS" --runs-per-seed "$RUNS_PER_SEED" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

if (( failed )); then
    echo "At least one worker failed; skipping analyze." >&2
    exit 1
fi

python -m scripts.imitation_poilcy.test_oracle_success analyze