"""
HuggingFace Setup module - Data import/export utilities.

This module provides functionality for importing and exporting data to/from
the HuggingFace Hub. It handles dataset management for the MUCH benchmark
and related resources.

Key functions:
    - import_much: Import MUCH dataset from HuggingFace Hub
    - import_mushroom: Import MU-SHroom dataset for all languages
    - import_signals: Import pre-computed signal values
    - export_much: Export MUCH data to HuggingFace Hub
"""

from .cli import export_much, import_much, import_mushroom, import_signals

__all__ = [
    "export_much",
    "import_much",
    "import_signals",
    "import_mushroom",
]
