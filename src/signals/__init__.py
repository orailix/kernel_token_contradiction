"""
Signals module - Uncertainty Quantification signal implementations.

This module provides implementations for various UQ signals used to evaluate
claim-level uncertainty in LLM outputs. Each signal is implemented as a subclass
of UQSignal and computes token-level uncertainty scores.

Available signals:
    - SignalKTC: Kernel Token Contradiction (main contribution)
    - SignalCCP: Cross-Encoder Probability baseline
    - SignalSAR: Semantic Augmented Reasoning baseline
    - SignalMaxLikelihood: Maximum Likelihood baseline
    - SignalTokenLikelihood: Token Likelihood baseline
    - SignalTokenEntropy: Token Entropy baseline
    - SignalSemanticVolume: Semantic Volume baseline
    - SignalToReFact: ToReFact baseline

Usage:
    python -m src compute-signal <signal_name> [--force-recompute]
"""

from .abc_signal import UQSignal
from .cli import compute_signal
from .plot_utils import get_plot, get_plot_debug_token_proba, plot_on_ax
from .sig_ccp import SignalCCP
from .sig_ktc import KTCConfig, SignalKTC
from .sig_max_likelihood import SignalMaxLikelihood
from .sig_sar import SignalSAR
from .sig_semantic_volume import SignalSemanticVolume
from .sig_token_entropy import SignalTokenEntropy
from .sig_token_likelihood import SignalTokenLikelihood
from .signal_value import UQSignalValue

__all__ = [
    "UQSignal",
    "compute_signal",
    "get_plot",
    "get_plot_debug_token_proba",
    "plot_on_ax",
    "KTCConfig",
    "SignalCCP",
    "SignalKTC",
    "SignalMaxLikelihood",
    "SignalTokenEntropy",
    "SignalTokenLikelihood",
    "SignalSAR",
    "UQSignalValue",
    "SignalSemanticVolume",
]
