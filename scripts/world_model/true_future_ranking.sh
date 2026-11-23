#!/bin/bash
# Run the true-future ranking check sharded over several worker processes, then analyze.
#
# Usage (from the project root):
#   ./scripts/world_model/true_future_ranking.sh [NUM_WORKERS=4] [K=8] [CONTINUATIONS=1] [SEED_FILTER=mixed] [NUM_SEEDS=100]
#
# SEED_FILTER: mixed       seeds the policy sometimes solves (from results/pass_at_k_iteration2)
#              not-perfect seeds it does not always solve
#              all         seeds 0 .. NUM_SEEDS-1
#
# Examples:
#   ./scripts/world_model/true_future_ranking.sh                 # 4 workers, K=8, mixed seeds
#   ./scripts/world_model/true_future_ranking.sh 4 8 2           # 2 continuations per candidate (less noisy)
#   ./scripts/world_model/true_future_ranking.sh 4 8 1 all 20    # first 20 seeds, no filtering
#
# Environment variables:
#   ENCODER   encoder checkpoint (default checkpoints/world_model_encoder/wm_encoder_best.pt)
#   OUT_DIR   results folder (default results/true_future_ranking_<encoder's folder name>)
#   e.g. ENCODER=checkpoints/encoder/wm_encoder_best.pt ./scripts/world_model/true_future_ranking.sh
#
# Rerunning with the same OUT_DIR skips decision points already done, so use a separate
# OUT_DIR for each encoder.

set -u

NUM_WORKERS=${1:-4}
K=${2:-8}
CONTINUATIONS=${3:-1}
SEED_FILTER=${4:-mixed}
NUM_SEEDS=${5:-100}
ENCODER=${ENCODER:-checkpoints/world_model_encoder/wm_encoder_best.pt}
OUT_DIR=${OUT_DIR:-results/true_future_ranking_$(basename "$(dirname "$ENCODER")")}
USAGE="Usage: $0 [NUM_WORKERS=4] [K=8] [CONTINUATIONS=1] [SEED_FILTER=mixed|not-perfect|all] [NUM_SEEDS=100]"

is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

for value in "$NUM_WORKERS" "$K" "$CONTINUATIONS" "$NUM_SEEDS"; do
    if ! is_pos_int "$value"; then
        echo "Error: numeric arguments must be positive integers (got '$value')." >&2
        echo "$USAGE" >&2
        exit 1
    fi
done

case "$SEED_FILTER" in
    mixed|not-perfect|all) ;;
    *) echo "Error: SEED_FILTER must be mixed, not-perfect or all (got '$SEED_FILTER')." >&2
       echo "$USAGE" >&2
       exit 1 ;;
esac

if command -v nproc >/dev/null 2>&1; then
    CORES=$(nproc)
    if (( NUM_WORKERS > CORES )); then
        echo "Warning: $NUM_WORKERS workers but only $CORES CPU cores; expect little extra speedup." >&2
    fi
fi

if [[ ! -f "$ENCODER" ]]; then
    echo "Error: encoder checkpoint '$ENCODER' not found." >&2
    exit 1
fi

echo "Encoder: $ENCODER -> results in $OUT_DIR"
echo "Launching $NUM_WORKERS workers: K=$K, $CONTINUATIONS continuation(s), seeds '$SEED_FILTER' (up to $NUM_SEEDS)"

pids=()
# Ctrl-C stops all workers instead of leaving them running in the background.
trap 'echo "Interrupted, stopping workers..." >&2; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

for ((w = 0; w < NUM_WORKERS; w++)); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.world_model.true_future_ranking --out-dir "$OUT_DIR" run \
        --encoder "$ENCODER" --worker "$w" --num-workers "$NUM_WORKERS" \
        --k "$K" --continuations "$CONTINUATIONS" \
        --seed-filter "$SEED_FILTER" --num-seeds "$NUM_SEEDS" &
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

python -m scripts.world_model.true_future_ranking --out-dir "$OUT_DIR" analyze
