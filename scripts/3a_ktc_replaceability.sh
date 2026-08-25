#!/bin/bash

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# =========================================================
# Script: 3a_ktc_replaceability.sh
# Description: Computes Wikipedia-based token replaceability for all 16 models
#              and both position settings (0 and 1). This is a prerequisite for KTC.
#              Computes token neighbors from Wikipedia corpus for each model's vocabulary.
# Expected runtime: ~6-11 hours (parallelized across models, ~3-5 min per model/position)
# Output: Pre-computed neighbor files in output/replaceability/
# Dependencies: Requires Wikipedia corpus access
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

echo "WARNING: You're about to launch tokenization of a subsequent part of"
echo "Wikipedia in 4 languages for 16 models and for 2 offset choices."
echo "Computation usually takes 3-5 min per setting, leading to a total"
echo "of 6-11 hours of computation."
read -p "Data is already committed in Git. Do you want to re-compute? [y/N]: " answer
case $answer in
    [yY]*) ;;
    *) echo "Aborted."; exit 0 ;;
esac

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
# ======================== CONFIG =========================
# =========================================================


ALL_MODELS=(
    ""meta-llama/Llama-3.2-3B-Instruct""
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Ministral-8B-Instruct-2410"
    "google/gemma-3-4b-it"
    "tiiuae/falcon-7b-instruct"
    "togethercomputer/Pythia-Chat-Base-7B"
    "occiglot/occiglot-7b-de-en-instruct"
    "malteos/bloom-6b4-clp-german-oasst-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "croissantllm/CroissantLLMChat-v0.1"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "occiglot/occiglot-7b-eu5-instruct"
    "Iker/Llama-3-Instruct-Neurona-8b-v2"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2-7B-Instruct"
)

# =========================================================
# =================== COMPUTE SIGNALS =====================
# =========================================================

for MODEL_NAME in "${ALL_MODELS[@]}"; do
    echo "Evaluating $MODEL_NAME"
    uv run python -m src compute-replaceability $FORCE_RECOMPUTE "$MODEL_NAME" --position 0 --num-proc 12;
    uv run python -m src compute-replaceability $FORCE_RECOMPUTE "$MODEL_NAME" --position 1 --num-proc 12;
done

# Clipping to 8 neighbors to be lighter.
uv run python -m src clip-neighbors --max-neighbors 8 --position 0
uv run python -m src clip-neighbors --max-neighbors 8 --position 1