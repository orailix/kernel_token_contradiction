import base64
import hashlib
import json
import random
import typing as t
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import torch
from filelock import FileLock
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.generation.utils import GenerateOutput

from ..questions import Question
from ..utils.configs import BaseConfig
from ..utils.constants import DEVICE, LLMS_INFOS, N_LOGITS_SAVED


@lru_cache
def get_model(model_name: str, eager_attn: bool, device=DEVICE) -> LlamaForCausalLM:
    """Gets the model.

    By default, eager attention is used, because it's needed for white-box approaches
    based on Attention weights and latent representations."""

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        **(dict(attn_implementation="eager") if eager_attn else dict()),
    )


@lru_cache(maxsize=256)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Gets the tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, legacy=True)

@lru_cache(maxsize=256)
def get_tokenizer_inverse_vocab(model_name: str) -> dict[int, str]:
    """Given a model name referring to a tokenizer, builds the inverse vocab {idx: str}"""
    return {value: key for key, value in get_tokenizer(model_name).vocab.items()}

def template_question(
    question: str,
    tokenizer: PreTrainedTokenizerFast,
    brief: bool = True,
) -> str:
    """Template a question.

    Note that for the moment we only verified reproducibility with Llama-3.2 chat template.

    Output
    ------
        prompt: The prompt as string."""

    # Template
    order_for_brief = (
        " Your answers should be very concise and precise." if brief else ""
    )
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always answer questions directly."
            + order_for_brief,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Chat template - in some case we need to fix the date
    if LLMS_INFOS[tokenizer.name_or_path]["filter_date"]:
        chat_template = "\n".join(
            [
                elt
                if "Today Date:" not in elt
                else r'{{- "Today Date: 10 Jun 2025\n\n" }}'
                for elt in tokenizer.chat_template.split("\n")
            ]
        )
    else:
        chat_template = tokenizer.chat_template

    # Removing today date from template for reproducibility
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )

    return prompt


def generate_reproducible(
    prompt: str,
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    brief: bool = True,
    greedy: bool = True,
    seed: int = 42,
    temperature: float = 1,
    top_p: float = 0.9,
    top_k: int = 20,
    max_new_tokens: int = 200,
) -> t.Tuple[t.Union[GenerateOutput, torch.LongTensor], str, torch.Tensor, int]:
    """Generates the model's response in a reproducible way.

    Output
    ------
        outputs: The output of model.generate
        output_str: The generation as string.
        tokens: The genetation as token (int tensor)
        len_prompt: The number of tokens in the prompt
    """

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Templating
    prompt_formatted = template_question(prompt, tokenizer, brief=brief)

    # Tokenization
    prompt_inputs = tokenizer.encode(
        prompt_formatted, add_special_tokens=False, return_tensors="pt"
    )
    len_prompt: int = prompt_inputs.size(-1)

    # Pad token
    pad_token_id = LLMS_INFOS[tokenizer.name_or_path]["pad_tok_id"]

    # Generation
    outputs = model.generate(
        input_ids=prompt_inputs.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        pad_token_id=pad_token_id,
        output_attentions=False,
        return_dict_in_generate=True,
        output_logits=True,
        **(
            dict(temperature=None, top_p=None, top_k=None)
            if greedy
            else dict(top_p=top_p, temperature=temperature, top_k=top_k)
        ),
    )

    # Output tokens
    output_tokens: torch.Tensor = outputs.sequences[0][len_prompt:]

    return (
        outputs,
        tokenizer.decode(output_tokens),
        output_tokens,
        len_prompt,
    )


@dataclass
class GenerationConfig(BaseConfig):
    @classmethod
    def class_output_name(cls) -> str:
        return "generations"

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

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    eager: bool = False
    brief: bool = True
    greedy: bool = False
    seed: int = 1234
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 20
    max_new_tokens: int = 500


class Generation(BaseModel):
    """Represents a single model generation with metadata.

    Fields
    ------
    generation_config : str
        Identifier of the configuration used to produce the generation.
    prompt : str
        The input text given to the model.
    output : str
        The model's generated output as text.
    lang : str
        Language of the prompt/output (e.g., "en", "fr").
    output_tokens : List[int]
        Token IDs of the generated output.
    topn_prob_dicts : List[Dict[int, float]]
        For each generated token, a dictionary mapping candidate token IDs
        to their probabilities (restricted to top-N).
    wiki_url : str, optional
        Associated Wikipedia URL providing context.
    """

    generation_config: str
    prompt: str
    output: str
    lang: str
    output_tokens: t.List[int]
    wiki_url: str
    topn_prob_dicts: t.List[t.Dict[int, float]]

    @cached_property
    def generation_cfg(self) -> GenerationConfig:
        return GenerationConfig.autoconfig(self.generation_config)

    def write_disk(self):
        """Append this generation to the JSONL file with file locking."""
        output_path = self.generation_cfg.get_output_dir() / "generations.jsonl"

        with FileLock(output_path.with_suffix(".lock")):
            is_nonempty = output_path.exists() and output_path.stat().st_size > 0
            with output_path.open("a") as f:
                if is_nonempty:
                    f.write("\n")
                self_as_dict = self.model_dump()
                self_as_dict = dict(id=self.id, **self_as_dict)
                f.write(json.dumps(self_as_dict))

    @property
    def id(self):
        """Unique, stable hash identifier for this generation."""
        return base64.urlsafe_b64encode(
            hashlib.md5(self.model_dump_json().encode("utf-8")).digest()
        ).decode()[:22]


def generate_one_question(
    question: Question,
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    generation_cfg: GenerationConfig,
    save_disk: bool = True,
) -> Generation:
    """Generates a model output for a single question.

    Parameters
    ----------
    question : Question
        The input question.
    model : LlamaForCausalLM
        The language model used for generation.
    tokenizer : PreTrainedTokenizerFast
        The tokenizer associated with the model.
    generation_cfg : GenerationConfig
        Configuration for text generation.
    save_disk : bool, optional
        If True, saves the generated output to disk.

    Returns
    -------
    Generation
        The generated output.
    """

    # Generation
    (outputs, output_str, output_tokens, len_prompt,) = generate_reproducible(
        prompt=question.prompt,
        model=model,
        tokenizer=tokenizer,
        brief=generation_cfg.brief,
        greedy=generation_cfg.greedy,
        seed=generation_cfg.seed,
        temperature=generation_cfg.temperature,
        top_p=generation_cfg.top_p,
        top_k=generation_cfg.top_k,
        max_new_tokens=generation_cfg.max_new_tokens,
    )

    # Compute token probabilities
    logits_cat = torch.cat(outputs.logits)
    probas = torch.softmax(logits_cat / generation_cfg.temperature, dim=1)

    # Sort tokens by probability (descending)
    _, sorted_indices = torch.sort(probas, dim=1, descending=True)

    # Rank of each sampled token
    token_match = sorted_indices == output_tokens.unsqueeze(1)
    rank_of_sampled = token_match.float().argmax(dim=1).cpu().tolist()

    # Collect top-N probabilities per generated token
    topn_prob_dicts: list[dict[int, float]] = []
    for output_idx in range(len(output_tokens)):
        rank = rank_of_sampled[output_idx]
        dict_size = max(N_LOGITS_SAVED, rank + 1)  # ensure sampled token is included

        top_indices = sorted_indices[output_idx, :dict_size]
        top_probas = probas[output_idx, top_indices]

        topn_prob_dicts.append(
            {int(idx): float(p) for idx, p in zip(top_indices, top_probas)}
        )

    # Building the annotation
    generation = Generation(
        generation_config=generation_cfg.id,
        prompt=question.prompt,
        output=output_str,
        lang=question.lang,
        output_tokens=list(output_tokens.cpu().numpy()),
        wiki_url=question.wiki_url,
        topn_prob_dicts=topn_prob_dicts,
    )

    # Saving
    if save_disk:
        generation.write_disk()

    return generation
