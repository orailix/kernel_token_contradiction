import typing as t

from ..generation import Generation
from .abc_signal import UQSignal


class SignalTokenLikelihood(UQSignal):

    """UQSignal containing the Token Likelihood baseline [4].

    The score is defined as the likelihood of the sampled token.

    [4] Nuno M. Guerreiro, Elena Voita, and André Martins. *Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation* EACL, pages 1059–1075. 2023. [https://aclanthology.org/2023.eacl-main.75/](https://aclanthology.org/2023.eacl-main.75/)
    """

    @property
    def signal_name(self) -> str:
        return "token_likelihood"

    def compute_signal_value(self, generation: Generation) -> list:

        # Init
        one_minus_p_sampled = []

        # Iter tokens
        for idx_tok, token in enumerate(generation.output_tokens):

            one_minus_p_sampled.append(1 - generation.topn_prob_dicts[idx_tok][token])

        # Output
        return one_minus_p_sampled
