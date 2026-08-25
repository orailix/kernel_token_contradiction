"""
Kernel Token Contradiction (KTC) - Main Python module.

This module provides the core implementation for the KTC approach to claim-level
Uncertainty Quantification (UQ) for Large Language Models (LLMs).

Submodules:
    - annotation: Claim annotation handling and GPT-based labeling
    - evaluation: Evaluation metrics and comparison with ground truth
    - generation: LLM generation utilities and token handling
    - hf_setup: HuggingFace Hub integration for data import/export
    - questions: Question generation and processing
    - replaceability: Token replaceability computation using Wikipedia corpus
    - signals: UQ signal implementations (KTC, baselines)
    - utils: Configuration, paths, constants, and utilities
"""

import os

from dotenv import load_dotenv

# Load env
load_dotenv(override=True)

# Additional imports
# isort:skip
from loguru import logger

from . import annotation, evaluation, generation, hf_setup, questions, signals, replaceability
from .utils import paths

logger.add(
    paths.logs_path / "main.log",
    rotation="10 MB",
)
