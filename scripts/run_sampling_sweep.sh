#!/bin/bash
# ============================================================
# Sweep the reverse-diffusion starting point at inference time
# (args.sampling_steps / args.sampling_noise), reusing an already
# trained checkpoint instead of retraining from scratch.
#
# REQUIRES the param.py / main.py / models/diffusion_process.py
# patch described in the accompanying chat message:
#   - a --eval_only flag (param.py) that skips training and only
#     evaluates a saved checkpoint (main.py: Coach.evaluate_only)
#   - a q_sample() method on DiffusionProcess (missing in the
#     current models/diffusion_process.py; p_sample() calls it
#     whenever sampling_steps > 0 and otherwise crashes)
#
# Uses the checkpoint saved by the ORIGINAL scripts/run_<dataset>.sh
# runs: Model/<dataset>/_tem_.pth (default --save_name is 'tem').
# Run those scripts first if that file doesn't exist yet.
#
# NOTE: sampling_steps only changes how much noise is injected
# before the reverse pass starts; the reverse loop itself always
# runs the full number of steps used in training (see chat message
# for details). Keep this in mind when reading the results.
#
# Usage:
#   bash run_sampling_sweep.sh                # all datasets, paused
#   bash run_sampling_sweep.sh ciao            # ciao only
#   bash run_sampling_sweep.sh all all -y      # everything, no pauses
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

RUNS_EXECUTED=0
FAILED_RUNS=0
CANDIDATE_STEPS=(0 5 10 20 50 100 200)

run_point () {
    local dataset=$1 steps=$2 noisy=$3
    get_base_args "$dataset" || return
    local total_steps
    total_steps="$(get_base_arg_value steps)"
    if (( steps > total_steps )); then
        return   # skip values above this dataset's total diffusion steps
    fi

    local checkpoint="Model/${dataset}/_tem_.pth"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Skipping $dataset: checkpoint not found at $checkpoint"
        echo "  (run scripts/run_${dataset}.sh first to produce it)."
        return
    fi

    RUNS_EXECUTED=$((RUNS_EXECUTED + 1))
    local tag="steps${steps}_noise${noisy}"
    local result_dir="History/${dataset}/sampling_sweep/${tag}"
    mkdir -p "$result_dir"

    extra_args=(--eval_only --checkpoint "$checkpoint" --sampling_steps "$steps")
    [[ "$noisy" == "True" ]] && extra_args+=(--sampling_noise True)

    echo "============================================================"
    echo ">>> Dataset: $dataset | sampling_steps=$steps | sampling_noise=$noisy"
    echo ">>> Checkpoint: $checkpoint"
    echo "============================================================"

    local console_log="${result_dir}/sampling_sweep_${tag}.console.log"
    python main.py "${BASE_ARGS[@]}" --save_name "sampling_sweep_${tag}" "${extra_args[@]}" \
        2>&1 | tee "$console_log"
    local exit_code=${PIPESTATUS[0]}

    if [[ "$exit_code" -ne 0 ]]; then
        echo ">>> FAILED (exit code $exit_code): $dataset / $tag  (log: $console_log)"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    else
        echo ">>> Done: $dataset / $tag  (log: $console_log)"
    fi
    pause_before_next
}

for dataset in ciao epinions yelp; do
    [[ "$DATASETS_FILTER" != "all" && "$DATASETS_FILTER" != "$dataset" ]] && continue
    for steps in "${CANDIDATE_STEPS[@]}"; do
        run_point "$dataset" "$steps" "False"
        run_point "$dataset" "$steps" "True"
    done
done

echo "============================================================"
if [[ "$RUNS_EXECUTED" -eq 0 ]]; then
    echo "WARNING: no run matched dataset='$DATASETS_FILTER', or no checkpoints were found."
    exit 1
fi
if [[ "$FAILED_RUNS" -gt 0 ]]; then
    echo "Sampling-steps sweep finished with $FAILED_RUNS failure(s) out of $RUNS_EXECUTED run(s)."
else
    echo "Sampling-steps sweep completed successfully ($RUNS_EXECUTED run(s) executed)."
fi
echo "Results are organized under History/<dataset>/sampling_sweep/<tag>/"
echo "============================================================"
