#!/bin/bash

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# =========================================================
# Script: 2c_baselines_gpu_sar.sh
# Description: Computes SAR (Semantic Augmented Reasoning) baseline signals on GPU.
#              Uses SLURM array jobs (0-8) to parallelize across sigma values.
#              Array indices cover sigma values: 3, 5, 8
# Usage: sbatch scripts/2c_baselines_gpu_sar.sh [--force-recompute]
# Output: Signal files in output/signals/ with naming pattern sar_<sigma>.jsonl
# Hardware: Requires Nvidia A100 GPU
# Runtime: ~10-20 minutes per job
# =========================================================

#SBATCH --array=0-8
#SBATCH --account=yfw@a100
#SBATCH --job-name=ccp_sig
#SBATCH --partition gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/hallu_bench/logs/%x.%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/hallu_bench/logs/%x.%A_%a.out
#SBATCH --no-requeue

# =========================================================
# ======================== SETUP ==========================
# =========================================================

# Dynamically find project root by searching upwards for pyproject.toml
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT=$SCRIPT_DIR
while [ ! -f "$ROOT/pyproject.toml" ] && [ ! -d "$ROOT/.git" ]; do
    ROOT=$(dirname "$ROOT")
    [ "$ROOT" = "/" ] && { echo "Error: Project root not found"; exit 1; }
done

cd "$ROOT"

# =========================================================
# ====================== WARN USER ========================
# =========================================================

# Parse --force-recompute flag
FORCE_RECOMPUTE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-recompute)  FORCE_RECOMPUTE="--force-recompute"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Echo warning only if flag is NOT present
if [[ -z "$FORCE_RECOMPUTE" ]]; then
    echo "WARNING: Data is already committed in Git. Use --force-recompute to recompute"
fi

# =========================================================
# ========================== KTC ==========================
# =========================================================

# Possible models, seeds, temperatures
ALL_SIZES=("L" "M" "S")
ALL_SIGMAS=(3 5 8)

# Total number of combinations
TOTAL_COMBINATIONS=$(( ${#ALL_SIZES[@]} * ${#ALL_SIGMAS[@]} ))

# If SLURM_ARRAY_TASK_ID is set, process only that task
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_ARRAY=("$SLURM_ARRAY_TASK_ID")
else
    # Local mode: Process all combinations
    TASK_ARRAY=($(seq 0 $((TOTAL_COMBINATIONS - 1))))
fi

# Process each task
for TASK_IDX in "${TASK_ARRAY[@]}"; do

    SIZE_IDX=$((TASK_IDX % ${#ALL_SIZES[@]}))
    export PARAM_SIZE="${ALL_SIZES[$SIZE_IDX]}"
    TASK_IDX=$((TASK_IDX / ${#ALL_SIZES[@]}))

    SIGMA_IDX=$((TASK_IDX % ${#ALL_SIGMAS[@]}))
    export PARAM_SIGMA="${ALL_SIGMAS[$SIGMA_IDX]}"
    TASK_IDX=$((TASK_IDX / ${#ALL_SIGMAS[@]}))

    echo "Processing size=$PARAM_SIZE and sigma=$PARAM_SIGMA"
    uv run python -m src compute-signal $FORCE_RECOMPUTE sar
done;


