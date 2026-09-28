import os

import torch
from loguru import logger

# CPU only?
CPU_ONLY = os.getenv("CPU_ONLY", "0") == "1"
if CPU_ONLY:
    logger.warning(f"Using CPU_ONLY mode.")

if (not CPU_ONLY) and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (not CPU_ONLY) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Saved logits for generations
N_LOGITS_SAVED = 24

# Authorized lang for questions
AUTHORIZED_LANG = {"en", "fr", "es", "de"}

# LLMs
LLMS_INFOS = {
    "meta-llama/Llama-3.2-3B-Instruct": dict(
        filter_date=True, pad_tok_id=128011, embed_key="model.embed_tokens.weight"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": dict(
        filter_date=True, pad_tok_id=128011, embed_key="model.embed_tokens.weight"
    ),
    "mistralai/Ministral-8B-Instruct-2410": dict(
        filter_date=False, pad_tok_id=11, embed_key="model.embed_tokens.weight"
    ),
    "google/gemma-3-4b-it": dict(
        filter_date=False,
        pad_tok_id=6,
        embed_key="language_model.model.embed_tokens.weight",
    ),
    "tiiuae/falcon-7b-instruct": dict(embed_key="transformer.word_embeddings.weight"),
    "togethercomputer/Pythia-Chat-Base-7B": dict(embed_key="gpt_neox.embed_in.weight"),
    "occiglot/occiglot-7b-de-en-instruct": dict(embed_key="model.embed_tokens.weight"),
    "malteos/bloom-6b4-clp-german-oasst-v0.1": dict(
        embed_key="transformer.word_embeddings.weight"
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": dict(embed_key="model.embed_tokens.weight"),
    "croissantllm/CroissantLLMChat-v0.1": dict(embed_key="model.embed_tokens.weight"),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": dict(
        embed_key="model.embed_tokens.weight"
    ),
    "mistralai/Mistral-Nemo-Instruct-2407": dict(embed_key="model.embed_tokens.weight"),
    "occiglot/occiglot-7b-eu5-instruct": dict(embed_key="model.embed_tokens.weight"),
    "Iker/Llama-3-Instruct-Neurona-8b-v2": dict(embed_key="model.embed_tokens.weight"),
    "meta-llama/Meta-Llama-3-8B-Instruct": dict(embed_key="model.embed_tokens.weight"),
    "Qwen/Qwen2-7B-Instruct": dict(embed_key="model.embed_tokens.weight"),
}

# Wikipedia articles
NUM_WIKI_ARTICLES = 1_500_000

# Signals defaults
DEFAULT_DELTA = 8
DEFAULT_SIGMA = -1
DEFAULT_SIZE = "L"
NLI_MODEL_NAME = (
    "microsoft/deberta-large-mnli",
    {0: "contradict", 1: "neutral", 2: "entail"},
)
NLI_MODEL_NAME_M = (
    "microsoft/deberta-base-mnli",
    {0: "contradict", 1: "neutral", 2: "entail"},
)
NLI_MODEL_NAME_S = (
    "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",
    {0: "contradict", 1: "entail"},
)

# HF export
HF_MUCH_PATH = "orailix/MUCH"
HF_CONFIGS_PATH = "orailix/MUCH-configs"
HF_SIGNALS_PATH = "orailix/MUCH-signals"
HF_TRASH_PATH = "orailix/MUCH-trash-only-for-reproducibility"

# DEBUG MODE?
KTC_DEBUG = os.getenv("KTC_DEBUG", "0") == "1"
