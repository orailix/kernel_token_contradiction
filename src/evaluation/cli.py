from ..signals.cli import SIGNAL_NAME_DICT
from ..utils.typer_app import app
from .supervized_evaluator import get_evaluation


@app.command()
def evaluate_signal(
    signal_name: str,
    force_recompute: bool = False,
    lang: str = None,
):
    """Evaluates a baseline by comparing its token-level signal to the annotations."""
    get_evaluation(
        SIGNAL_NAME_DICT[signal_name].from_env(),
        force_recompute,
        verbose=True,
        lang=lang,
    )
