#!/bin/bash
# ============================================================
# Robustness / variance analysis: reruns the FULL model (baseline
# hyperparameters, no ablation) with several random seeds, so you
# can compute mean +/- std of Recall@K / NDCG@K per dataset.
#
# Results layout:
#   History/<dataset>/seeds/<seed>/seed_<seed>.console.log
#   History/<dataset>/seeds/<seed>/seed_<seed>.txt
#   History/<dataset>/seeds/<seed>/_seed_<seed>_.his
#   Model/<dataset>/seeds/<seed>/_seed_<seed>_.pth
#
# After running, summarize with:
#   python scripts/aggregate_results.py History
#
# Usage:
#   bash run_seed_robustness.sh                # all datasets, paused
#   bash run_seed_robustness.sh ciao           # ciao only
#   bash run_seed_robustness.sh all all -y     # everything, no pauses
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_dataset_configs.sh" "$@"

DATASETS_FILTER="${1:-all}"

if ! is_in_list "$DATASETS_FILTER" "${VALID_DATASETS[@]}"; then
    echo "Warning: '$DATASETS_FILTER' is not a recognized dataset."
    echo "Valid datasets: ${VALID_DATASETS[*]}"
    exit 1
fi

# 2023 is the original default seed (already covered by your baseline
# run); it's included here too so all 5 points land in the same
# folder structure and are easy to aggregate together.
SEEDS=(2023 2024 2025 2026 2027)

RUNS_EXECUTED=0

run_point () {
    local dataset=$1 seed=$2
    get_base_args "$dataset" || return

    RUNS_EXECUTED=$((RUNS_EXECUTED + 1))
    local tag="seed_${seed}"
    local result_dir="History/${dataset}/seeds/${seed}"
    local model_dir="Model/${dataset}/seeds/${seed}/"
    mkdir -p "$result_dir" "$model_dir"

    echo "============================================================"
    echo ">>> Dataset: $dataset | Seed: $seed"
    echo "============================================================"

    local console_log="${result_dir}/${tag}.console.log"
    python main.py "${BASE_ARGS[@]}" --save_name "$tag" --seed "$seed" \
        --model_dir "$model_dir" 2>&1 | tee "$console_log"

    local logger_txt="History/${dataset}/${tag}.txt"
    local history_his="History/${dataset}/_${tag}_.his"
    [[ -f "$logger_txt" ]] && mv "$logger_txt" "$result_dir/"
    [[ -f "$history_his" ]] && mv "$history_his" "$result_dir/"

    echo ">>> Done: $dataset / seed=$seed  (log: $console_log)"
    pause_before_next
}

for dataset in ciao epinions yelp; do
    [[ "$DATASETS_FILTER" != "all" && "$DATASETS_FILTER" != "$dataset" ]] && continue
    for seed in "${SEEDS[@]}"; do
        run_point "$dataset" "$seed"
    done
done

echo "============================================================"
if [[ "$RUNS_EXECUTED" -eq 0 ]]; then
    echo "WARNING: no run matched dataset='$DATASETS_FILTER'."
    exit 1
fi
echo "Seed robustness runs completed ($RUNS_EXECUTED run(s) executed)."
echo "Results are organized under History/<dataset>/seeds/<seed>/"
echo "Run 'python scripts/aggregate_results.py History' to summarize."
echo "============================================================"