#!/bin/bash

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# =========================================================
# Script: 1_data_setup.sh
# Description: Setup script to download and import required datasets.
#              Imports MUCH benchmark and MU-SHroom dataset for all 4 languages (en, fr, de, es).
#              Cleans existing output/generations and output/annotations directories first.
# Usage: bash scripts/1_data_setup.sh
# Output: Dataset files in output/annotations/ and output/generations/
# Dependencies: HuggingFace Hub access, python -m src modules
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

# Warn user: data is already in Git repo
read -p "WARNING: Data is already committed in Git. Recomputing will overwrite output/. Continue? [y/N]: " answer
case $answer in
    [yY]*) ;;
    *) echo "Aborted."; exit 0 ;;
esac

# Clean existing output
if [ -d "output/generations" ] || [ -d "output/annotations" ]; then
    echo "Removing output/generations and output/annotations..."
    rm -rf output/generations output/annotations
fi

# =========================================================
# ======================= IMPORTS =========================
# =========================================================

# Import MUCH
uv run python -m src import-much

# Import MU-SHroom in all languages
uv run python -m src import-mushroom en
uv run python -m src import-mushroom fr
uv run python -m src import-mushroom de
uv run python -m src import-mushroom es
