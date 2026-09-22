#!/bin/bash
# ============================================================
# Hyperparameter sensitivity analysis for RecDiff.
# Varies ONE hyperparameter at a time (all others kept at the
# dataset's baseline from scripts/run_<dataset>.sh) and records
# Recall@K / NDCG@K for each value.
#
# Parameters swept (edit the GRID array below to add/remove):
#   steps        diffusion steps
#   noise_scale  noise magnitude scaling
#   n_hid        hidden embedding dimension
#   lr           GCN learning rate
#   s_layers     social GCN layers
#
# Results layout:
#   History/<dataset>/sensitivity/<param>/<value>/sens_<param>_<value>.console.log
#   History/<dataset>/sensitivity/<param>/<value>/sens_<param>_<value>.txt
#   History/<dataset>/sensitivity/<param>/<value>/_sens_<param>_<value>_.his
#   Model/<dataset>/sensitivity/<param>/<value>/_sens_<param>_<value>_.pth
#
# By default the script pauses after each run (Enter = continue,
# 'q' = stop), since a full "all/all" sweep is 3 datasets x 5
# params x 4 values = 60 runs. Pass -y / AUTO_CONFIRM=1 to skip
# pauses, or filter down to one dataset/param at a time.
#
# Usage:
#   bash run_sensitivity.sh                    # everything, paused
#   bash run_sensitivity.sh ciao                # ciao only, all params
#   bash run_sensitivity.sh ciao steps          # ciao, steps only
#   bash run_sensitivity.sh all n_hid -y        # n_hid on all datasets, no pauses
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_dataset_configs.sh" "$@"

DATASETS_FILTER="${1:-all}"
PARAM_FILTER="${2:-all}"

declare -A GRID
GRID[steps]="10 50 100 200"
GRID[noise_scale]="0.01 0.05 0.1 0.2"
GRID[n_hid]="32 64 128 256"
GRID[lr]="0.0001 0.001 0.005"
GRID[s_layers]="1 2 3"

VALID_PARAMS=("all" "${!GRID[@]}")

if ! is_in_list "$DATASETS_FILTER" "${VALID_DATASETS[@]}"; then
    echo "Warning: '$DATASETS_FILTER' is not a recognized dataset."
    echo "Valid datasets: ${VALID_DATASETS[*]}"
    exit 1
fi
if ! is_in_list "$PARAM_FILTER" "${VALID_PARAMS[@]}"; then
    echo "Warning: '$PARAM_FILTER' is not a recognized parameter."
    echo "Valid parameters: ${VALID_PARAMS[*]}"
    exit 1
fi

RUNS_EXECUTED=0

run_point () {
    local dataset=$1 param=$2 value=$3
    get_base_args "$dataset" || return

    RUNS_EXECUTED=$((RUNS_EXECUTED + 1))

    local tag="sens_${param}_${value}"
    local result_dir="History/${dataset}/sensitivity/${param}/${value}"
    local model_dir="Model/${dataset}/sensitivity/${param}/${value}/"
    mkdir -p "$result_dir" "$model_dir"

    echo "============================================================"
    echo ">>> Dataset: $dataset | Param: $param = $value"
    echo ">>> Results folder: $result_dir"
    echo "============================================================"

    local console_log="${result_dir}/${tag}.console.log"
    python main.py "${BASE_ARGS[@]}" --save_name "$tag" "--${param}" "$value" \
        --model_dir "$model_dir" 2>&1 | tee "$console_log"

    local logger_txt="History/${dataset}/${tag}.txt"
    local history_his="History/${dataset}/_${tag}_.his"
    [[ -f "$logger_txt" ]] && mv "$logger_txt" "$result_dir/"
    [[ -f "$history_his" ]] && mv "$history_his" "$result_dir/"

    echo ">>> Done: $dataset / $param=$value  (log: $console_log)"
    pause_before_next
}

for dataset in ciao epinions yelp; do
    [[ "$DATASETS_FILTER" != "all" && "$DATASETS_FILTER" != "$dataset" ]] && continue
    for param in "${!GRID[@]}"; do
        [[ "$PARAM_FILTER" != "all" && "$PARAM_FILTER" != "$param" ]] && continue
        for value in ${GRID[$param]}; do
            run_point "$dataset" "$param" "$value"
        done
    done
done

echo "============================================================"
if [[ "$RUNS_EXECUTED" -eq 0 ]]; then
    echo "WARNING: no run matched dataset='$DATASETS_FILTER' / param='$PARAM_FILTER'."
    exit 1
fi
echo "Sensitivity analysis completed ($RUNS_EXECUTED run(s) executed)."
echo "Results are organized under History/<dataset>/sensitivity/<param>/<value>/"
echo "============================================================"
