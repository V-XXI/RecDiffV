#!/bin/bash
# ============================================================
# Efficiency analysis: measures per-epoch training time and
# per-evaluation test time as --steps (diffusion steps) and
# --n_hid (embedding dimension) vary.
#
# Uses --n_epoch 5 instead of the full 150, since we only need
# a handful of "Training complete in ...s" / "Testing complete
# in ...s" samples per configuration, not full convergence.
#
# Results layout:
#   History/<dataset>/efficiency/<tag>/eff_<tag>.console.log
#   (the .txt/.his files are also produced but timing is easier
#   to read straight from the console log / aggregate_results.py)
#
# After running, summarize with:
#   python scripts/aggregate_results.py History
#
# Usage:
#   bash run_efficiency.sh                # all datasets, paused
#   bash run_efficiency.sh ciao           # ciao only
#   bash run_efficiency.sh all all -y     # everything, no pauses
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_dataset_configs.sh" "$@"

DATASETS_FILTER="${1:-all}"
N_EPOCH_PROBE=5

if ! is_in_list "$DATASETS_FILTER" "${VALID_DATASETS[@]}"; then
    echo "Warning: '$DATASETS_FILTER' is not a recognized dataset."
    echo "Valid datasets: ${VALID_DATASETS[*]}"
    exit 1
fi

STEPS_VALUES=(10 50 100 200)
NHID_VALUES=(32 64 128 256)

RUNS_EXECUTED=0
FAILED_RUNS=0

run_point () {
    local dataset=$1 param=$2 value=$3
    get_base_args "$dataset" || return

    RUNS_EXECUTED=$((RUNS_EXECUTED + 1))
    local tag="eff_${param}_${value}"
    local result_dir="History/${dataset}/efficiency/${param}_${value}"
    local model_dir="Model/${dataset}/efficiency/${param}_${value}/"
    mkdir -p "$result_dir" "$model_dir"

    echo "============================================================"
    echo ">>> Dataset: $dataset | $param = $value | probing $N_EPOCH_PROBE epochs"
    echo "============================================================"

    local console_log="${result_dir}/${tag}.console.log"
    python main.py "${BASE_ARGS[@]}" --save_name "$tag" "--${param}" "$value" \
        --n_epoch "$N_EPOCH_PROBE" --model_dir "$model_dir" 2>&1 | tee "$console_log"
    local exit_code=${PIPESTATUS[0]}

    local logger_txt="History/${dataset}/${tag}.txt"
    local history_his="History/${dataset}/_${tag}_.his"
    [[ -f "$logger_txt" ]] && mv "$logger_txt" "$result_dir/"
    [[ -f "$history_his" ]] && mv "$history_his" "$result_dir/"

    if [[ "$exit_code" -ne 0 ]]; then
        echo ">>> FAILED (exit code $exit_code): $dataset / $param=$value  (log: $console_log)"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    else
        echo ">>> Done: $dataset / $param=$value  (log: $console_log)"
    fi
    pause_before_next
}

for dataset in ciao epinions yelp; do
    [[ "$DATASETS_FILTER" != "all" && "$DATASETS_FILTER" != "$dataset" ]] && continue
    for v in "${STEPS_VALUES[@]}"; do
        run_point "$dataset" "steps" "$v"
    done
    for v in "${NHID_VALUES[@]}"; do
        run_point "$dataset" "n_hid" "$v"
    done
done

echo "============================================================"
if [[ "$RUNS_EXECUTED" -eq 0 ]]; then
    echo "WARNING: no run matched dataset='$DATASETS_FILTER'."
    exit 1
fi
if [[ "$FAILED_RUNS" -gt 0 ]]; then
    echo "Efficiency probing finished with $FAILED_RUNS failure(s) out of $RUNS_EXECUTED run(s)."
else
    echo "Efficiency probing completed successfully ($RUNS_EXECUTED run(s) executed)."
fi
echo "Results are organized under History/<dataset>/efficiency/<tag>/"
echo "Run 'python scripts/aggregate_results.py History' to summarize timings."
echo "============================================================"
