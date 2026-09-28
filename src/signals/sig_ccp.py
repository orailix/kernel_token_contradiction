import os
import typing as t
from collections import defaultdict
from functools import cached_property

import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.flop_counter import FlopCounterMode
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaForSequenceClassification,
    DebertaTokenizerFast,
)

from ..generation import Generation, get_tokenizer
from ..utils.constants import (
    DEFAULT_DELTA,
    DEFAULT_SIGMA,
    DEFAULT_SIZE,
    DEVICE,
    NLI_MODEL_NAME,
    NLI_MODEL_NAME_M,
    NLI_MODEL_NAME_S,
)
from .abc_signal import UQSignal


class SignalCCP(UQSignal):
    """UQSignal containing the CCP baseline [1].

    The main adaptation arises from the fact that the meaning of claims in MUCH is often not self-contained, whereas the original method operates on self-contained sentences. To preserve the performance of the Natural Language Inference (NLI) model used in CCP, we provide the model with the `sigma=8` preceding and succeeding tokens of the token for which the CCP score is computed. Moreover, we evaluated the `delta=24` possible alternative for each token (the original paper used `delta = 10`).

    [1] Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, Sergey Petrakov, Haonan Li, Hamdy Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, et al. *Fact-Checking the Output of Large Language Models via Token-Level Uncertainty Quantification* Findings of the ACL pages 9367–9385. 2024. [https://aclanthology.org/2024.findings-acl.558/](https://aclanthology.org/2024.findings-acl.558/)
    """

    def __init__(
        self,
        delta: int = DEFAULT_DELTA,
        sigma: int = DEFAULT_SIGMA,
        size: str = DEFAULT_SIZE,
        count_nn_flop: bool = False,
    ):
        super().__init__()
        self.delta: int = delta
        self.sigma: int = sigma
        self.size: str = size

        # Count flop
        self.count_nn_flop = count_nn_flop
        if count_nn_flop:
            self._flop_total = 0
            self._total_model_call = 0

        if size not in ["L", "M", "S"]:
            raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

    @classmethod
    def from_env(cls) -> t.Self:
        delta: int = int(os.getenv("PARAM_DELTA", DEFAULT_DELTA))
        sigma: int = int(os.getenv("PARAM_SIGMA", DEFAULT_SIGMA))
        size: int = os.getenv("PARAM_SIZE", DEFAULT_SIZE)
        count_nn_flop: bool = os.getenv("COUNT_NN_FLOP", "0") == "1"
        return cls(delta=delta, sigma=sigma, size=size, count_nn_flop=count_nn_flop)

    def warm_up(self):
        super().warm_up()
        self.nli_model
        self.nli_tokenizer

    @property
    def signal_name(self) -> str:
        if self.size == "L":
            size_addon = ""
        elif self.size == "M":
            size_addon = "_M"
        elif self.size == "S":
            size_addon = "_S"
        else:
            raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

        return f"ccp_{self.delta}_{self.sigma}{size_addon}"

    @property
    def nli_model_name(self) -> str:
        if self.size == "L":
            return NLI_MODEL_NAME[0]
        elif self.size == "M":
            return NLI_MODEL_NAME_M[0]
        elif self.size == "S":
            return NLI_MODEL_NAME_S[0]

        raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

    @property
    def nli_class_label(self) -> str:
        if self.size == "L":
            return NLI_MODEL_NAME[1]
        elif self.size == "M":
            return NLI_MODEL_NAME_M[1]
        elif self.size == "S":
            return NLI_MODEL_NAME_S[1]

        raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

    @cached_property
    def flop_counter(self):
        return FlopCounterMode(depth=None, display=False)

    @cached_property
    def nli_model(self) -> DebertaForSequenceClassification:
        model: DebertaForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).to(
                DEVICE
            )
        )
        model.eval()
        return model

    @cached_property
    def nli_tokenizer(self) -> DebertaTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        return tokenizer

    def get_ccp_nli_values(
        self,
        selected_token: str,
        token_list: t.List[str],
        preceding_str: str,
        succeding_str: str,
    ) -> dict[str, list[int]]:
        """
        Gets the contradictions wrt the chosen token.

        Parameters
        ----------
        selected_token : the token selected by the LLM at inference time
        token_list : the list of candidate tokens at inference time (usually include `selected_tokens`)
        preceding_str : the string preceding this token choice at inference time
        succeding_str : the string succeding this token choice at inference time

        Returns
        -------
        A dict{"contradict":list[int], "neutral":list[int], "entail":list[int]}
        For each possible label, the list of int represent the `rank` (ie position in `token_list`)
        of all candidate token that contradict (resp neural, entail) the token that was selected at inference time.

        """
        # Init result
        n_elt = len(token_list)
        result = torch.zeros(n_elt, n_elt).float()

        # Forming the batch
        batch_idx_to_rank = []
        premises = []
        hypotheses = []
        for rank, candidate in enumerate(token_list):
            batch_idx_to_rank.append(rank)
            premises.append(preceding_str + selected_token + succeding_str)
            hypotheses.append(preceding_str + candidate + succeding_str)

        batch = self.nli_tokenizer(
            premises, hypotheses, truncation=False, padding=True, return_tensors="pt"
        ).to(DEVICE)

        # Inference - For some models, flop_counter is incompatible with torch.no_grad()
        if self.count_nn_flop:
            with self.flop_counter:
                logits = self.nli_model(**batch).logits
            self._flop_total += self.flop_counter.get_total_flops()
            self._total_model_call += 1
        else:
            with torch.no_grad():
                logits = self.nli_model(**batch).logits

        labels = logits.argmax(dim=1)

        # Getting class label
        nli_class_labels = self.nli_class_label
        for key, label in nli_class_labels.items():
            if label == "entail":
                entail_class = key

        # Ensuring the label of selected_token is correct
        for rank, candidate in enumerate(token_list):
            if candidate == selected_token:
                labels[rank] = entail_class
                break

        # Forming result
        result = defaultdict(list)
        for rank, label in zip(batch_idx_to_rank, labels.cpu().numpy().tolist()):
            result[nli_class_labels[label]].append(rank)

        # Output
        return result

    def compute_signal_value(self, generation: Generation) -> list:

        # Init
        ccp_scores = []

        # Iterating over tokens
        for idx_tok, token in enumerate(generation.output_tokens):

            # Candidates and their corresponding likelihoods
            sorted_candidates = sorted(
                generation.topn_prob_dicts[idx_tok],
                key=generation.topn_prob_dicts[idx_tok].get,
                reverse=True,
            )[: self.delta]
            sorted_likelihoods = sorted(
                generation.topn_prob_dicts[idx_tok].values(), reverse=True
            )[: self.delta]

            # If the selected token is not in :delta, we must add it
            if token not in sorted_candidates:
                sorted_candidates.append(token)
                sorted_likelihoods.append(generation.topn_prob_dicts[idx_tok][token])

            # Computing CCP NLI values
            llm_tokenizer = get_tokenizer(generation.generation_cfg.model_name)
            selected_token = llm_tokenizer.decode(token)
            token_list = [
                llm_tokenizer.decode(candidate) for candidate in sorted_candidates
            ]
            preceding_str = llm_tokenizer.decode(
                generation.output_tokens[max(0, idx_tok - self.sigma) : idx_tok]
            )
            succeding_str = llm_tokenizer.decode(
                generation.output_tokens[idx_tok + 1 : idx_tok + self.sigma]
            )
            ccp_nli_values = self.get_ccp_nli_values(
                selected_token, token_list, preceding_str, succeding_str
            )

            # Computing CCP score
            numerator = sum(
                sorted_likelihoods[rank] for rank in ccp_nli_values["entail"]
            )
            denominator = numerator + sum(
                sorted_likelihoods[rank] for rank in ccp_nli_values["contradict"]
            )

            ccp_scores.append(1 - numerator / denominator)

        # Output
        return ccp_scores
