import os
import typing as t

import numpy as np
import torch
from scipy.stats import entropy

from ..generation import Generation, get_model
from ..utils.constants import DEFAULT_DELTA
from .abc_signal import UQSignal


class SignalTokenEntropy(UQSignal):

    """UQSignal containing the Token Entropy baseline [5].

    The score is the entropy of the probability distribution over the top-24 tokens provided in MUCH.

    [5] Andrey Malinin and Mark Gales. *Uncertainty Estimation in Autoregressive Structured Prediction* ICLR. 2021. [https://openreview.net/forum?id=jN5y-zb5Q7m](https://openreview.net/forum?id=jN5y-zb5Q7m)
    """

    def __init__(self, delta: int = DEFAULT_DELTA):
        super().__init__()
        self.delta: int = delta

    @classmethod
    def from_env(cls) -> t.Self:
        delta: int = int(os.getenv("PARAM_DELTA", DEFAULT_DELTA))
        return cls(delta=delta)

    @property
    def signal_name(self) -> str:
        return f"token_entropy_{self.delta}"

    def compute_signal_value(self, generation: Generation) -> list:

        # Getting sorted values and token indices
        sorted_values, sorted_token_indices = [], []
        for idx_token in range(len(generation.output_tokens)):
            sorted_values.append(
                sorted(generation.topn_prob_dicts[idx_token].values(), reverse=True)[
                    : self.delta
                ]
            )
            sorted_token_indices.append(
                sorted(
                    generation.topn_prob_dicts[idx_token],
                    key=generation.topn_prob_dicts[idx_token].get,
                    reverse=True,
                )[: self.delta]
            )

        sorted_values = torch.Tensor(sorted_values).float()
        sorted_token_indices = torch.Tensor(sorted_token_indices).int()

        # Non-semantic entropy
        top_probas = sorted_values[:, : self.delta]
        token_entropy = entropy(top_probas, axis=1) / np.log(self.delta)

        # Output
        return token_entropy.tolist()
