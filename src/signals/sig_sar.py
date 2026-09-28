import os
import typing as t
from functools import cached_property

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaForSequenceClassification,
    DebertaTokenizerFast,
)

from ..generation import Generation, get_tokenizer
from ..utils.constants import (
    DEFAULT_SIGMA,
    DEFAULT_SIZE,
    DEVICE,
    NLI_MODEL_NAME,
    NLI_MODEL_NAME_M,
    NLI_MODEL_NAME_S,
)
from .abc_signal import UQSignal


class SignalSAR(UQSignal):
    """UQSignal containing the SAR baseline [2].

    Like CCP, it relies on an entailment score computed on sentences. For the same reason, we fed the NLI model with the `sigma=8` preceding and succeeding tokens around the target token.

    [2] Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. *Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models* ACL, volume 1, pages 5050–5063. 2024. [https://aclanthology.org/2024.acl-long.276/](https://aclanthology.org/2024.acl-long.276/)
    """

    def __init__(
        self,
        sigma: int = DEFAULT_SIGMA,
        count_nn_flop: bool = False,
        size: str = DEFAULT_SIZE,
    ):
        super().__init__()
        self.sigma: int = sigma
        self.size: str = size

        # Count flop
        self.count_nn_flop = count_nn_flop
        if count_nn_flop:
            self._flop_total = 0
            self._total_model_call = 0

        if size not in ["L", "M", "S"]:
            raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

    def warm_up(self):
        super().warm_up()
        self.nli_model
        self.nli_tokenizer

    @classmethod
    def from_env(cls) -> t.Self:
        sigma: int = int(os.getenv("PARAM_SIGMA", DEFAULT_SIGMA))
        size: int = os.getenv("PARAM_SIZE", DEFAULT_SIZE)
        count_nn_flop: bool = os.getenv("COUNT_NN_FLOP", "0") == "1"
        return cls(sigma=sigma, size=size, count_nn_flop=count_nn_flop)

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

        return f"sar_{self.sigma}{size_addon}"

    @cached_property
    def flop_counter(self):
        return FlopCounterMode(depth=None, display=False)

    @property
    def nli_model_name(self) -> str:
        if self.size == "L":
            return NLI_MODEL_NAME[0]
        elif self.size == "M":
            return NLI_MODEL_NAME_M[0]
        elif self.size == "S":
            return NLI_MODEL_NAME_S[0]

        raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

    @cached_property
    def entail_class(self) -> str:
        if self.size == "L":
            nli_class_labels = NLI_MODEL_NAME[1]
        elif self.size == "M":
            nli_class_labels = NLI_MODEL_NAME_M[1]
        elif self.size == "S":
            nli_class_labels = NLI_MODEL_NAME_S[1]
        else:
            raise ValueError(f"Size={self.size} should be in ['L', 'M', 'S']")

        for key, label in nli_class_labels.items():
            if label == "entail":
                return key

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

    def compute_signal_value(self, generation: Generation) -> list:

        # Init
        result = []

        # Computing relevance score
        premises = []
        hypotheses = []
        llm_tokenizer = get_tokenizer(generation.generation_cfg.model_name)
        for idx_tok, token in enumerate(generation.output_tokens):
            decoded_token = llm_tokenizer.decode(token)
            preceding_str = llm_tokenizer.decode(
                generation.output_tokens[max(0, idx_tok - self.sigma) : idx_tok]
            )
            succeding_str = llm_tokenizer.decode(
                generation.output_tokens[idx_tok + 1 : idx_tok + self.sigma]
            )
            premises.append(preceding_str + decoded_token + succeding_str)
            premises.append(preceding_str + succeding_str)
            hypotheses.append(preceding_str + succeding_str)
            hypotheses.append(preceding_str + decoded_token + succeding_str)

        # Inference - For some models, flop_counter is incompatible with torch.no_grad()
        # For count nn flop, we need to split in two batches for memory reasons
        if self.count_nn_flop:
            batch_0 = self.nli_tokenizer(
                premises[: len(premises) // 2],
                hypotheses[: len(hypotheses) // 2],
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            batch_1 = self.nli_tokenizer(
                premises[len(premises) // 2 :],
                hypotheses[len(hypotheses) // 2 :],
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            with self.flop_counter:
                logits_0 = self.nli_model(**batch_0).logits
            self._flop_total += self.flop_counter.get_total_flops()

            logits_0 = logits_0.detach()
            self.nli_model.zero_grad()
            torch.cuda.empty_cache()

            with self.flop_counter:
                logits_1 = self.nli_model(**batch_1).logits
            self._flop_total += self.flop_counter.get_total_flops()

            logits = torch.cat([logits_0, logits_1])

            self._total_model_call += 1
        else:
            batch = self.nli_tokenizer(
                premises,
                hypotheses,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.nli_model(**batch).logits

        logits = torch.softmax(logits.detach(), dim=1)
        entail_scores = logits[:, self.entail_class]
        sar_scores = (1 - entail_scores).cpu().numpy()

        # Tolist
        sar_scores_processed = []
        for idx in range(0, len(sar_scores), 2):
            sar_scores_processed.append(
                ((sar_scores[idx] + sar_scores[idx + 1]) / 2).item()
            )

        # Iterating over tokens
        for idx_tok, token in enumerate(generation.output_tokens):

            # Candidates and their corresponding likelihoods
            selected_likelihood = generation.topn_prob_dicts[idx_tok][token]
            result.append((1 - selected_likelihood) * sar_scores_processed[idx_tok])

        # Output
        return result
