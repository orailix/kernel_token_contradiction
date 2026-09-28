import typing as t

from ..generation import Generation
from .abc_signal import UQSignal


class SignalMaxLikelihood(UQSignal):

    """UQSignal containing the Maximum Likelihood baseline [3].

    The token-level UQ score corresponds to the probability of the most likely token at each generation step.

    [3] Lukas Aichberger, Kajetan Schweighofer, and Sepp Hochreiter. *Rethinking Uncertainty Estimation in Natural Language Generation* QUESTION workshop at ICLR. 2025. [https://openreview.net/forum?id=iBKWqXCSFA](https://openreview.net/forum?id=iBKWqXCSFA)"""

    @property
    def signal_name(self) -> str:
        return "max_likelihood"

    def compute_signal_value(self, generation: Generation) -> list:

        # Init
        one_minus_max_p = []

        # Iter tokens
        for idx_tok, _ in enumerate(generation.output_tokens):
            one_minus_max_p.append(
                1 - max(generation.topn_prob_dicts[idx_tok].values())
            )

        # Output
        return one_minus_max_p
