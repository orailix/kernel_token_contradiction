import os
import typing as t
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..generation.utils import Generation, get_tokenizer, get_tokenizer_inverse_vocab
from ..replaceability import load_sorted_neighbors
from ..utils.configs import BaseConfig
from ..utils.constants import LLMS_INFOS
from .abc_signal import UQSignal
from .utils import (
    augment_diagonal,
    clip_neg_eigs,
    get_wiki_adjacency,
    heat_kernel,
    square_matrix,
    von_neumann_entropy,
)


@dataclass
class KTCConfig(BaseConfig):
    """UQSignal containing the Kernel Token Contradiction (KTC) baseline."""

    @classmethod
    def class_output_name(cls) -> str:
        return "ktc_configs"

    @classmethod
    def key_ignored_in_hash(cls):
        return [
            "_base_output_dir",
        ]

    @classmethod
    def ignore_in_hash_if_equal_to_default(cls):
        return False

    def __repr__(self):
        return super().__repr__()

    # Methods
    kernel_method: str = "heat"
    """Method to compute the PSD kernel from the adjacency matrix.
    Should be one of ["clip", "augm", "square", "heat"]"""
    normalization_method: str = "scale"
    """Method to normalize the PSD kernel using the conditional distribution.
    Should be one of ["scale", "comb", "none"]"""
    prefix_entail: bool = True
    """Whether or not being a prefix implies entailment."""
    language_specific: bool = False
    """Whether or not we use language-specific tokenizers for the contradiction."""
    token_position: int = 0
    """The position of neighbors for wiki adjacency."""

    # Hyperparameters - adjacency
    delta: int = 24
    """Number of candidate token (in the generation process) to account
    for the computation of the adjacency matrix and the kernels."""
    nu: int = 8
    """For the wikipedia tokenization method to compute adjacency matrices,
    size of the neighbor set to be considered."""
    sigma: int = 3
    """For the NLI method to compute adjacency matrices, number of
    preceeding tokens to consider and pass as context to the NLI model."""

    # Hyperparameters - kenel method
    tau: float = 0.3
    """For the HEAT kernel, diffusion time : K = exp(-TAU x Laplacian)"""

    # Hyperparameters - normalization
    alpha: float = 0.5
    """For the normalization method that involves normalization with
    the conditional distribution, weight of K: ALPHA x K + (1-ALPHA) x PRIOR"""


class SignalKTC(UQSignal):
    def __init__(self, ktc_config_id: str):
        super().__init__()
        self.ktc_config = KTCConfig.autoconfig(ktc_config_id)
        self.ktc_config_id = self.ktc_config.id

    @classmethod
    def from_env(cls) -> t.Self:
        if "PARAM_KTC_CONFIG_ID" in os.environ:
            ktc_config_id: str = str(os.getenv("PARAM_KTC_CONFIG_ID"))
            return cls(ktc_config_id=ktc_config_id)
        else:
            return cls(ktc_config_id=KTCConfig.from_env().id)

    def warm_up(self):
        super().warm_up()
        for model_name in tqdm(LLMS_INFOS):
            _ = get_tokenizer(model_name)
            _ = get_tokenizer_inverse_vocab(model_name)

            for lang in [None, "fr", "en", "de", "es"]:
                for position in [0, 1]:
                    _ = load_sorted_neighbors(model_name, lang=lang, position=position)

    @property
    def signal_name(self) -> str:
        return f"ktc_{self.ktc_config_id}"

    def compute_signal_value(self, generation: Generation) -> list:
        # Tokenizer
        tokenizer = get_tokenizer(generation.generation_cfg.model_name)

        # Output tokens
        output_tokens = generation.output_tokens

        # Iterating over the tokens
        result = []
        for idx_token in range(len(output_tokens)):

            # Top n candidate
            topn_local = generation.topn_prob_dicts[idx_token]
            candidates, conditional = [], []
            for key in sorted(topn_local, key=topn_local.__getitem__, reverse=True):
                if len(candidates) == self.ktc_config.delta:
                    break

                candidates.append(key)
                conditional.append(topn_local[key])

            # Getting adjacency
            adjacency = get_wiki_adjacency(
                candidates,
                nu=self.ktc_config.nu,
                model_name=generation.generation_cfg.model_name,
                prefix_entail=self.ktc_config.prefix_entail,
                lang=generation.lang if self.ktc_config.language_specific else None,
            )

            # Conditional normalizer
            conditional_kernel_sqrt = torch.diag(torch.Tensor(conditional) ** (0.5))

            # Kernel matrices
            if self.ktc_config.kernel_method == "heat":
                graph_kernel = heat_kernel(adjacency, tau=self.ktc_config.tau)
            elif self.ktc_config.kernel_method == "square":
                graph_kernel = square_matrix(adjacency)
            elif self.ktc_config.kernel_method == "augm":
                graph_kernel = augment_diagonal(adjacency)
            elif self.ktc_config.kernel_method == "clip":
                graph_kernel = clip_neg_eigs(adjacency)
            else:
                raise ValueError(
                    f"Unknown kernel_method '{self.ktc_config.kernel_method}'"
                )

            # Normalization
            if self.ktc_config.normalization_method == "scale":
                final_kernel = (
                    conditional_kernel_sqrt @ graph_kernel @ conditional_kernel_sqrt
                )
            elif self.ktc_config.normalization_method == "comb":
                alpha = self.ktc_config.alpha
                final_kernel = (
                    alpha * conditional_kernel_sqrt**2 + (1 - alpha) * graph_kernel
                )
            elif self.ktc_config.normalization_method == "none":
                final_kernel = graph_kernel
            else:
                raise ValueError(
                    f"Unknown normalization_method '{self.ktc_config.normalization_method}'"
                )

            # VNE
            result.append(von_neumann_entropy(final_kernel, force_trace=True))

        return result
