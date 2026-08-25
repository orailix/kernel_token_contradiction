"""
Generation module - LLM generation utilities and data structures.

This module provides functionality for generating LLM outputs and handling
the resulting data structures. It includes:
    - Generation: Main generation class and data structures
    - GenerationConfig: Configuration for generation parameters
    - loaders: Functions to load and retrieve generations
    - utils: Generation utilities including model loading and tokenization

Key functions:
    - generate: CLI command to generate LLM outputs
    - get_generation_from_idx: Retrieve a generation by its ID
    - get_all_generations: Retrieve all generations
    - get_tokenizer: Get tokenizer for a specific model
    - get_model: Load a model for generation
"""

from .cli import generate
from .loaders import (
    get_all_generations,
    get_generation_from_idx,
    get_generations_from_config,
)
from .utils import (
    Generation,
    GenerationConfig,
    generate_reproducible,
    get_model,
    get_tokenizer,
    template_question,
    get_tokenizer_inverse_vocab
)

__all__ = [
    "generate",
    "get_all_generations",
    "get_generation_from_idx",
    "get_generations_from_config",
    "Generation",
    "GenerationConfig",
    "generate_reproducible",
    "get_model",
    "get_tokenizer",
    "get_tokenizer_inverse_vocab",
    "template_question",
]
