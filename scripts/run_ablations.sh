#!/bin/bash
# ============================================================
# Ablation studies for RecDiff: -D (w/o Diffusion), -S (w/o Social),
# -RW (w/o Reweight), run on ciao, epinions, and yelp.
#
# Reuses the base hyperparameters from the corresponding
# scripts/run_*.sh files.
#
# Results layout (per dataset / per variant):
#   History/<dataset>/ablation/<variant>/ablation_<variant>.console.log   (full stdout/stderr)
#   History/<dataset>/ablation/<variant>/ablation_<variant>.txt           (logger output, moved here)
#   History/<dataset>/ablation/<variant>/_ablation_<variant>_.his         (pickled loss/Recall/NDCG history, moved here)
#   Model/<dataset>/ablation/<variant>/_ablation_<variant>_.pth           (best checkpoint, saved here directly)
#
# NOTE: main.py hardcodes the .txt and .his output paths to
# ./History/<dataset>/ (no subfolders). This script lets main.py
# write them there as usual, then moves them into the nested
# ablation/<variant>/ folder once the run finishes.
#
# Requires the --no_diffusion flag patch in param.py / main.py
# described earlier (needed for the "no_diffusion" variant only).
#
# By default, the script PAUSES after each run and waits for you
# to press Enter before starting the next one, so you can check
# system load / temperature / free up resources between jobs.
#   - Press Enter  -> start the next run
#   - Type 'q'     -> stop the whole script
# Pass -y (or set AUTO_CONFIRM=1) to skip the pauses and run
# everything back-to-back, as before.
#
# Usage:
#   bash run_ablations.sh                    # all datasets/variants, paused between each
#   bash run_ablations.sh ciao               # ciao only, paused between each
#   bash run_ablations.sh ciao no_social      # a single dataset/variant, no pause needed
#   bash run_ablations.sh all all -y          # everything, no pauses (old behavior)
#   AUTO_CONFIRM=1 bash run_ablations.sh      # same as above, via env var
# ============================================================

set -e

DATASETS_FILTER="${1:-all}"
VARIANT_FILTER="${2:-all}"
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"

# Allow -y as a 3rd positional arg or anywhere in the args
for arg in "$@"; do
    if [[ "$arg" == "-y" || "$arg" == "--yes" ]]; then
        AUTO_CONFIRM=1
    fi
done

VALID_DATASETS=("all" "ciao" "epinions" "yelp")
VALID_VARIANTS=("all" "no_diffusion" "no_social" "no_reweight")

is_in_list () {
    local needle="$1"; shift
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# If the first arg is not a valid dataset name, assume the user meant to
# filter by variant instead (e.g. `run_ablations.sh no_diffusion`) and
# shift it into VARIANT_FILTER automatically.
if ! is_in_list "$DATASETS_FILTER" "${VALID_DATASETS[@]}"; then
    if is_in_list "$DATASETS_FILTER" "${VALID_VARIANTS[@]}"; then
        echo "Note: '$DATASETS_FILTER' is a variant, not a dataset -> filtering all datasets for this variant."
        VARIANT_FILTER="$DATASETS_FILTER"
        DATASETS_FILTER="all"
    else
        echo "Warning: '$DATASETS_FILTER' is not a recognized dataset or variant."
        echo "Valid datasets: ${VALID_DATASETS[*]}"
        echo "Valid variants: ${VALID_VARIANTS[*]}"
        exit 1
    fi
fi

if [[ "$VARIANT_FILTER" != "all" ]] && ! is_in_list "$VARIANT_FILTER" "${VALID_VARIANTS[@]}"; then
    echo "Warning: '$VARIANT_FILTER' is not a recognized variant."
    echo "Valid variants: ${VALID_VARIANTS[*]}"
    exit 1
fi

RUNS_EXECUTED=0

pause_before_next () {
    if [[ "$AUTO_CONFIRM" == "1" ]]; then
        return
    fi
    echo
    read -r -p ">>> Press Enter to start the next run, or type 'q' to stop: " answer
    if [[ "$answer" == "q" || "$answer" == "Q" ]]; then
        echo "Stopped by user."
        exit 0
    fi
}

run_variant () {
    dataset=$1
    variant=$2      # variant name (used for save_name / folder name)
    extra_args=$3   # overrides relative to the base config

    if [[ "$DATASETS_FILTER" != "all" && "$DATASETS_FILTER" != "$dataset" ]]; then
        return
    fi
    if [[ "$VARIANT_FILTER" != "all" && "$VARIANT_FILTER" != "$variant" ]]; then
        return
    fi

    RUNS_EXECUTED=$((RUNS_EXECUTED + 1))

    save_name="ablation_${variant}"
    result_dir="History/${dataset}/ablation/${variant}"
    model_dir="Model/${dataset}/ablation/${variant}/"

    mkdir -p "$result_dir"
    mkdir -p "$model_dir"

    echo "============================================================"
    echo ">>> Dataset: $dataset | Variant: $variant"
    echo ">>> Extra args: $extra_args"
    echo ">>> Results folder: $result_dir"
    echo "============================================================"

    case $dataset in
        ciao)
            base_args="--n_hid 64 --dataset ciao --n_layers 2 --s_layers 2 --lr 5e-3 --difflr 1e-3 \
                --reg 1e-2 --batch_size 2048 --test_batch_size 1024 --emb_size 16 --steps 20 \
                --noise_scale 1"
            ;;
        epinions)
            base_args="--n_hid 64 --dataset epinions --n_layers 2 --s_layers 2 --lr 0.001 --difflr 0.001 \
                --reg 0.0001 --batch_size 4096 --test_batch_size 1024 --emb_size 16 --steps 200 \
                --noise_scale 0.1"
            ;;
        yelp)
            base_args="--n_hid 64 --dataset yelp --n_layers 2 --s_layers 2 --lr 0.001 --difflr 0.0001 \
                --reg 0.005 --batch_size 4096 --test_batch_size 2048 --emb_size 16 --steps 50 \
                --noise_scale 0.1"
            ;;
        *)
            echo "Unknown dataset: $dataset"
            return
            ;;
    esac

    console_log="${result_dir}/${save_name}.console.log"

    # --model_dir is placed after base_args/extra_args so it overrides
    # any default set elsewhere (argparse: last occurrence wins).
    python main.py $base_args --save_name "$save_name" $extra_args \
        --model_dir "$model_dir" 2>&1 | tee "$console_log"

    # main.py writes these two files directly under History/<dataset>/
    # (hardcoded path); move them into the nested ablation folder.
    logger_txt="History/${dataset}/${save_name}.txt"
    history_his="History/${dataset}/_${save_name}_.his"

    [[ -f "$logger_txt" ]] && mv "$logger_txt" "$result_dir/"
    [[ -f "$history_his" ]] && mv "$history_his" "$result_dir/"

    echo ">>> Done: $dataset / $variant"
    echo ">>> Console log : $console_log"
    echo ">>> Logger file : $result_dir/$(basename "$logger_txt")"
    echo ">>> History file: $result_dir/$(basename "$history_his")"

    pause_before_next
}

for dataset in ciao epinions yelp; do
    # --- (-D) w/o Diffusion: bypass the diffusion module via the --no_diffusion flag
    #     (requires the param.py / main.py patch: setting --noise_scale 0 alone
    #     will crash at init because the beta schedule becomes all-zero and
    #     fails the "betas > 0" assertion)
    run_variant "$dataset" "no_diffusion" "--no_diffusion"

    # --- (-S) w/o Social: disable the GCN layers on the user-user graph ---
    run_variant "$dataset" "no_social" "--s_layers 0"

    # --- w/o Reweight: uniform per-timestep weighting instead of SNR-based ---
    run_variant "$dataset" "no_reweight" "--no_reweight"
done

echo "============================================================"
if [[ "$RUNS_EXECUTED" -eq 0 ]]; then
    echo "WARNING: no run matched dataset='$DATASETS_FILTER' / variant='$VARIANT_FILTER'."
    echo "Nothing was executed. Check the filters and try again."
    exit 1
fi
echo "Ablation studies completed ($RUNS_EXECUTED run(s) executed)."
echo "Results are organized under History/<dataset>/ablation/<variant>/"
echo "Checkpoints are organized under Model/<dataset>/ablation/<variant>/"
echo "============================================================"
