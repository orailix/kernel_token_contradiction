"""
Replaceability module - Token neighbor computation and caching.

This module provides functionality for computing token replaceability using
Wikipedia corpus statistics. It identifies neighboring tokens that can be used
as alternatives for computing contradiction scores in KTC.

Key components:
    - compute_neighbors: Compute token neighbors for a specific model
    - load_neighbors: Load pre-computed neighbors from cache
    - load_sorted_neighbors: Load and sort neighbors by similarity

Key functions:
    - compute_replaceability: CLI command to compute replaceability for all models
    - clip_neighbors: Clip neighbor lists to a maximum size
"""

from .cli import clip_neighbors, compute_replaceability
from .utils import compute_neighbors, load_neighbors, load_sorted_neighbors

__all__ = [
    "compute_replaceability",
    "clip_neighbors",
    "compute_neighbors",
    "load_neighbors",
    "load_sorted_neighbors",
]
