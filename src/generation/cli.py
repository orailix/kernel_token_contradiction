import gc
import typing as t
from time import time

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..questions import get_all_questions
from ..utils.typer_app import app
from .loaders import get_all_generations
from .utils import GenerationConfig, generate_one_question, get_model, get_tokenizer


@app.command()
def generate(
    generation_config: t.Optional[str] = None,
):
    """Computes generations for all available questions."""

    # Loading generation config
    logger.info("Loading generation config")
    if generation_config is not None:
        generation_cfg = GenerationConfig.autoconfig(generation_config)
    else:
        generation_cfg = GenerationConfig.from_env()

    # Model, tokenizer
    logger.info("Loading model an tokenizer")
    model = get_model(generation_cfg.model_name, eager_attn=generation_cfg.eager)
    tokenizer = get_tokenizer(generation_cfg.model_name)

    # Getting questions
    logger.info("Loading all questions")
    all_questions = get_all_questions()

    # Logging
    logger.info(f"Computing generation for {len(all_questions)} available questions...")
    logger.info(generation_cfg)

    # Infinite while
    for question in tqdm(all_questions):
        _ = generate_one_question(
            question, model, tokenizer, generation_cfg=generation_cfg, save_disk=True
        )


@app.command()
def evaluate_generation_time(approx_factor: float = 0.05):
    """Evaluates the generation time of the 4.8k samples in MUCH."""

    # Creating all configs
    all_model_names = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Ministral-8B-Instruct-2410",
    ]
    all_temperatures = [1.0, 0.7]
    all_configs = [
        GenerationConfig(seed=1234, model_name=model_name, temperature=temperature)
        for model_name in all_model_names
        for temperature in all_temperatures
    ]

    # Init container
    question_processing_time: float = 0

    # Loading questions
    all_questions = get_all_questions()
    fist_question = all_questions[0]
    split_size = int(np.ceil(approx_factor * len(all_questions)))
    rg = np.random.RandomState(42)

    for generation_cfg in all_configs:
        logger.info(f"Processing config {generation_cfg.id}")

        # Loading of the model
        model = get_model(generation_cfg.model_name, eager_attn=generation_cfg.eager)
        tokenizer = get_tokenizer(generation_cfg.model_name)

        # Generating the first question separately in case there is a special overhead
        _ = generate_one_question(
            fist_question,
            model=model,
            tokenizer=tokenizer,
            generation_cfg=generation_cfg,
            save_disk=False,
        )

        # Timing a given number of questions
        random_split = rg.choice(all_questions, split_size, replace=False)
        delta_t0 = time()
        for question in tqdm(random_split):
            _ = generate_one_question(
                question,
                model=model,
                tokenizer=tokenizer,
                generation_cfg=generation_cfg,
                save_disk=False,
            )
        question_processing_time += time() - delta_t0

        # Clearing
        get_model.cache_clear()
        get_tokenizer.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()

    # Aggregating - we need to consider that MUCH is a subset of the total generation (after filtering)
    # We first multiply by len(all_questions) / split_size to estimate for all questions
    # Then, we multiply by 2270/1108 because we observed that LLama 3.2 + Ministral took 1108s out of the
    # 2270s required for the whole generation
    # For much generation, we multiply by (4884 / 6448) to accont for the dataset size, and then by
    # (4.26/5.30) because MUCH samples are 4.26 claims in average vs 5.30 for the others.
    total_generation_time = (
        question_processing_time * (len(all_questions) / split_size) * (2270 / 1108)
    )
    much_generation_time = total_generation_time * (4884 / 6448) * (4.26 / 5.30)
    logger.debug(
        f"Question processing time: {question_processing_time} | Num samples: {split_size} / {len(all_questions)}"
    )
    logger.debug(
        f"Total generation time (estimated for the 6448 samples before filtering): {total_generation_time}"
    )
    logger.info(
        f"Final generation time (estimated for the 4884 samples of MUCH): {much_generation_time}"
    )


@app.command()
def reload_all_generations():
    """Loads all generations from the disk, and saves them back. Can serve to:
    - Recompute all hashed of the generations
    - Add attributes to the generations in case another one was implemented"""

    all_generations = get_all_generations()
    all_configs = GenerationConfig.get_all_configs()

    # Removing all existing generations
    for generation_cfg in all_configs:
        (generation_cfg.get_output_dir() / "generations.jsonl").unlink()

    # Re-writing all generations
    for _, generation in all_generations.items():
        generation.write_disk()
