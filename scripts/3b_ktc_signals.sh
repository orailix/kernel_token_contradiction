#!/bin/bash

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# =========================================================
# Script: 3b_ktc_signals.sh
# Description: Computes Kernel Token Contradiction (KTC) signals with various hyperparameter configurations.
#              Explores the KTC parameter space across multiple batches.
# 
# Batch 1: Explores (Nu, Tau) with adjacency_method=wiki
#   - Nu: {3, 4, 5, 6, 8}
#   - Tau: {0.1, 0.2, 0.3, 0.4, 0.5, 0.7}
#   - Token Position: fixed at default (0)
#   - Generates 5 batches of 6 configs (30 total)
# 
# Batch 2: Ablation study with normalization methods
#   - Scale method: Nu in {3, 4, 5, 6, 8}, Tau fixed at 0.3
#   - Comb method: Nu in {3, 4, 5, 6, 8}, Alpha in {0, 0.25, 0.5, 0.75, 1.0}, Tau fixed at 0.3
#
# Default values (not varied in batches):
#   - kernel_method = heat
#   - embedding_weight = 0.0
#   - prefix_entail = true
#   - language_specific = false
#   - delta = 24
#   - token_position = 0
# Output: KTC signal files in output/signals/ with naming pattern ktc_<config_id>.jsonl
# =========================================================

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
# =================== COMPUTE SIGNALS =====================
# =========================================================

# Default values, no need to override: 
# kernel_method = heat
# prefix_entail = true
# language_specific = false
# delta = 24
export NU TAU

# Batch 1: for exploring various values of (Nu, Tau, Token Position)
# 30 hyperparameter values explored
echo "Starting 5 batches of 6 configs for batch 1";
for NU in 3 4 5 6 8; do
    for TAU in 0.1 0.2 0.3 0.4 0.5 0.7; do
        python -m src compute-signal ktc&
    done
    wait;
    echo "Finished a batch of 6 configs!";
done

# Batch 2 (Ablation in Table 1): impact of alpha with SCALE method
export NU ALPHA
export TAU=0.3

# SCALE
echo "Starting 1 batch of 5 configs for batch 2 - scale";
export NORMALIZATION_METHOD="scale"
for NU in 3 4 5 6 8; do
    python -m src compute-signal ktc &
done
wait;
echo "Finished a batch of 5 configs!";

# COMB
echo "Starting 5 batch of 5 configs for batch 2 - comb";
export NORMALIZATION_METHOD="comb"
for NU in 3 4 5 6 8; do
    for ALPHA in 0 0.25 0.5 0.75 1.0; do
        python -m src compute-signal ktc &
    done
    wait;
    echo "Finished a batch of 5 configs!";
done
