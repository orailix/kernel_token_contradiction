"""
Evaluation module - Signal evaluation and performance metrics.

This module provides functionality for evaluating UQ signals against ground
truth annotations and computing performance metrics.

Key components:
    - SignalEvaluation: Class for evaluating signal performance
    - Supervised evaluation: Compute metrics using labeled data
    - NoChunkEval: Evaluation without chunking

Key functions:
    - evaluate_signal: CLI command to evaluate a specific signal
    - compute_evaluation: Compute evaluation metrics for signals
    - get_evaluation: Retrieve evaluation results
"""

from .cli import evaluate_signal
from .signal_evaluation import SignalEvaluation
from .supervized_evaluator import NoChunkEval, compute_evaluation, get_evaluation

__all__ = [
    "evaluate_signal",
    "SignalEvaluation",
    "compute_evaluation",
    "get_evaluation",
    "NoChunkEval",
]
