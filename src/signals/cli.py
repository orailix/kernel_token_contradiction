import typing as t
from time import time

from loguru import logger

from ..generation import get_generation_from_idx
from ..utils.constants import KTC_DEBUG
from ..utils.typer_app import app
from .abc_signal import UQSignal
from .sig_ccp import SignalCCP
from .sig_ktc import SignalKTC
from .sig_max_likelihood import SignalMaxLikelihood
from .sig_sar import SignalSAR
from .sig_token_entropy import SignalTokenEntropy
from .sig_token_likelihood import SignalTokenLikelihood

SIGNAL_NAME_DICT: t.Dict[str, UQSignal] = {
    "max_likelihood": SignalMaxLikelihood,
    "token_likelihood": SignalTokenLikelihood,
    "token_entropy": SignalTokenEntropy,
    "ccp": SignalCCP,
    "sar": SignalSAR,
    "ktc": SignalKTC,
}


@app.command()
def compute_signal(
    signal_name: str,
    force_recompute: bool = False,
):
    """Computed a signal value (token-level UQ score of a baseline)."""

    # Warming up - only in DEBUG mode
    signal = SIGNAL_NAME_DICT[signal_name].from_env()
    if KTC_DEBUG:
        signal.warm_up()

    # Computation core
    t0 = time()
    signal.compute_on_all_generations(force_recompute)
    logger.info(f"Computed {signal_name} in {time() - t0:.3f} seconds")

    # Flop
    if hasattr(signal, "count_nn_flop") and signal.count_nn_flop:
        logger.info(
            f"FLOP: {signal._flop_total:_} and {signal._total_model_call:_} model call"
        )


@app.command()
def compute_one_signal(
    signal_name: str, generation_id: str, force_recompute: bool = False
):
    """Computed a signal value (token-level UQ score of a baseline) for one generation."""
    t0 = time()
    generation = get_generation_from_idx(generation_id)
    SIGNAL_NAME_DICT[signal_name].from_env().compute_on_generation(
        generation, force_recompute
    )
    logger.info(f"Computed {signal_name} in {time() - t0:.3f} seconds")
