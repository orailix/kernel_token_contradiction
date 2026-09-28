import gc
import itertools
import typing as t
from functools import lru_cache

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from much_segmenter import much_segmentation
from much_segmenter.utils import get_tokens_and_spans
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from ..annotation import Annotation, get_all_annotations
from ..generation import Generation, GenerationConfig, get_tokenizer
from ..generation.utils import N_LOGITS_SAVED

DEVICE = torch.device("cpu")
MODEL_CONVERTER = {
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": "mistralai/Mistral-7B-Instruct-v0.2"
}
UNSUPPORTED_MODELS = ["TheBloke/SauerkrautLM-7B-v1-GGUF", "bofenghuang/vigogne-2-13b-chat"]
LANG_TO_SEED = {"en": 0, "fr": 1, "es": 2, "de": 3}


class LogitNotInTop24(Exception):
    pass


@lru_cache
def get_mushroom_per_lang() -> dict[str, Dataset]:
    """Gets Mu-Shroom dataset per lang"""
    mushrooms = {}
    for lang in ["en", "fr", "de", "es"]:
        mushrooms[lang] = load_dataset("Helsinki-NLP/mu-shroom", name=lang)["test"]

    return mushrooms


@lru_cache
def get_mushroom_model(model_name: str) -> LlamaForCausalLM:
    model_name = MODEL_CONVERTER.get(model_name, model_name)
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False).to(
        DEVICE
    )


def clear_mushroom_model_cache():
    get_mushroom_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


def apply_model_specific_templtate(
    model_name: str, question: str, output_text: str, tokenizer
) -> tuple[int, str, t.Any]:
    "Applies model-specific template for generation"
    tokenizer = get_tokenizer(model_name)
    if model_name == "togethercomputer/Pythia-Chat-Base-7B":
        new_prompt = f"<human>: {question}\n<bot>:"
        new_output = output_text

    elif model_name == "tiiuae/falcon-7b-instruct":
        new_prompt = question + "\n"
        new_output = output_text + tokenizer.eos_token

    elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        new_prompt = f"[INST] {question} [/INST]"
        new_output = output_text + tokenizer.eos_token

    elif model_name == "malteos/bloom-6b4-clp-german-oasst-v0.1":
        new_prompt = question + "\n"
        new_output = output_text + tokenizer.eos_token

    elif model_name == "occiglot/occiglot-7b-de-en-instruct":
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please give short and concise answers.",
            },
            {"role": "user", "content": question},
        ]
        new_prompt = tokenizer.decode(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                return_tensors="pt",
            )[0]
        )
        new_output = output_text.split("<|im_end|>")[0] + "<|im_end|>"

    elif model_name in [
        "croissantllm/CroissantLLMChat-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "occiglot/occiglot-7b-eu5-instruct",
        "Iker/Llama-3-Instruct-Neurona-8b-v2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        messages = [{"role": "user", "content": question}]
        new_prompt = tokenizer.decode(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                return_tensors="pt",
            )[0]
        )
        new_output = output_text + tokenizer.eos_token

    else:
        raise ValueError(f"Unupported model_name: {model_name}")

    prompt_len = tokenizer(new_prompt, return_tensors="pt")["input_ids"].size(-1)
    sentence = new_prompt + new_output
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

    return prompt_len, new_output, inputs


def get_generation_and_annotation(
    lang: str,
    model_id: str,
    question: str,
    output_text: str,
    soft_labels: dict,
) -> tuple[Generation, Annotation]:
    """Gets the Generation and Annotation related to a sample"""

    # Sanity check
    if model_id in UNSUPPORTED_MODELS:
        raise ValueError(f"Unuspported model: {model_id}")

    # Model and tokenizer
    model_name = MODEL_CONVERTER.get(model_id, model_id)
    model = get_mushroom_model(model_name)
    tokenizer = get_tokenizer(model_name)

    # Re-computing logits
    prompt_len, new_output, inputs = apply_model_specific_templtate(
        model_name, question, output_text, tokenizer
    )
    output_tokens = inputs["input_ids"][0, prompt_len:]

    # Compute token probabilities
    outputs = model(**inputs)
    logits = outputs.logits[:, prompt_len - 1 : -1]
    probas = torch.softmax(logits[0], dim=1)

    # Sort tokens by probability (descending)
    _, sorted_indices = torch.sort(probas, dim=1, descending=True)

    # Rank of each sampled token
    token_match = sorted_indices == inputs["input_ids"][
        :, prompt_len:
    ].squeeze().unsqueeze(1)
    rank_of_sampled = token_match.float().argmax(dim=1).cpu().tolist()

    # Collect top-N probabilities per generated token
    all_tokens_are_ok = True
    topn_prob_dicts: list[dict[int, float]] = []
    for output_idx in range(len(output_tokens)):
        rank = rank_of_sampled[output_idx]
        if rank >= N_LOGITS_SAVED:
            all_tokens_are_ok = False

        top_indices = sorted_indices[output_idx, :N_LOGITS_SAVED]
        top_probas = probas[output_idx, top_indices]

        topn_prob_dicts.append(
            {int(idx): float(p) for idx, p in zip(top_indices, top_probas)}
        )

    # All tokens are OK?
    if not all_tokens_are_ok:
        logger.warning(f"Some tokens are NOK: {question} | {new_output}")
        raise LogitNotInTop24

    # Getting hallucination indices
    filtered_soft_label = [
        (lab["start"], lab["end"]) for lab in soft_labels if lab["prob"] == 1.0
    ]
    hallu_indices = set(
        itertools.chain.from_iterable(
            [list(range(span[0], span[1] + 1)) for span in filtered_soft_label]
        )
    )

    # Chunking
    _, token_spans = get_tokens_and_spans(new_output, tokenizer)
    output_chunks = much_segmentation(new_output, tokenizer)
    chunk_chars = [
        set(range(token_spans[chunk[0]][0], token_spans[chunk[-1]][1]))
        for chunk in output_chunks
    ]

    # Hallucination spans
    chunk_hallu = [len(chars.intersection(hallu_indices)) > 0 for chars in chunk_chars]
    chunk_hallu = [-1 if item else 1 for item in chunk_hallu]

    # Forming the result
    generation_cfg = GenerationConfig(
        model_name=model_name,
        seed=LANG_TO_SEED[lang],
    )
    generation = Generation(
        generation_config=generation_cfg.id,
        prompt=question,
        output=new_output,
        lang=lang,
        output_tokens=output_tokens,
        wiki_url="None",
        topn_prob_dicts=topn_prob_dicts,
    )
    annotation = Annotation(
        generation_id=generation.id,
        token_chunks=output_chunks,
        labels={"mushroom": chunk_hallu},
    )

    # Output
    return generation, annotation


def recompute_one_mushroom_label(annotation: Annotation) -> None:
    """Recomputes the label of a given Mu-Shroom annotation"""

    # Finding the corresponding sample in MU-Shroom dataset
    current_mushroom = get_mushroom_per_lang()[annotation.generation.lang]
    current_prompt = annotation.generation.prompt
    current_output = annotation.generation.output

    corresponding_rank = None
    for rank in range(len(current_mushroom["model_input"])):
        if (
            current_mushroom["model_input"][rank] in current_prompt
            and current_mushroom["model_output_text"][rank] in current_output
        ):
            corresponding_rank = rank

    if corresponding_rank is None:
        logger.warning(f"Found no corresponding rank: {current_prompt} | {current_output}")
        return

    # Getting the soft labels
    soft_labels = current_mushroom["soft_labels"][corresponding_rank]

    # Filtering high proba
    filtered_soft_label = [
        (lab["start"], lab["end"]) for lab in soft_labels if lab["prob"] >= 0.90
    ]

    # Indices of hallucination
    hallu_indices = set(
        itertools.chain.from_iterable(
            [list(range(span[0], span[1] + 1)) for span in filtered_soft_label]
        )
    )

    # Tokenizer and output
    tokenizer = get_tokenizer(annotation.generation.generation_cfg.model_name)
    output = annotation.generation.output

    # Chunking
    _, token_spans = get_tokens_and_spans(output, tokenizer)
    output_chunks = much_segmentation(output, tokenizer)
    chunk_chars = [
        set(range(token_spans[chunk[0]][0], token_spans[chunk[-1]][1]))
        for chunk in output_chunks
    ]

    # Chunk hallu
    chunk_hallu = [len(chars.intersection(hallu_indices)) > 0 for chars in chunk_chars]
    chunk_hallu = [-1 if item else 1 for item in chunk_hallu]

    # Forming the result adn saving
    annotation.labels = {"mushroom": chunk_hallu}
    annotation.write_disk()
