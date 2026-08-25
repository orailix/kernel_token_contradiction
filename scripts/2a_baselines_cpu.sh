#!/bin/bash

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# =========================================================
# Script: 2a_baselines_cpu.sh
# Description: Computes CPU-only baseline signals (no GPU required).
#              Includes: max_likelihood, token_likelihood, and token_entropy.
#              Token entropy is computed for delta values {5, 10, 24}.
# Usage: bash scripts/2a_baselines_cpu.sh [--force-recompute]
# Output: Signal files in output/signals/
#   - max_likelihood.jsonl
#   - token_likelihood.jsonl
#   - token_entropy_5.jsonl, token_entropy_10.jsonl, token_entropy_24.jsonl
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
# ======================= SIGNALS =========================
# =========================================================

# Likelihoods
uv run python -m src compute-signal $FORCE_RECOMPUTE max_likelihood
uv run python -m src compute-signal $FORCE_RECOMPUTE token_likelihood 

# Token entropy
for PARAM_DELTA in 5 10 24; do
    PARAM_DELTA=$PARAM_DELTA uv run python -m src compute-signal $FORCE_RECOMPUTE token_entropy 
done;
