import gc
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from ..utils import paths
from ..utils.constants import AUTHORIZED_LANG, LLMS_INFOS, NUM_WIKI_ARTICLES
from ..utils.typer_app import app
from .utils import POSITION_TO_SUFFIX, compute_neighbors, load_neighbors


@app.command()
def compute_replaceability(
    model_name: str,
    force_recompute: bool = False,
    num_proc: int = None,
    only_lang: str = None,
    position: int = 0,
):

    # Need recompute?
    export_dir = paths.output / "replaceability"
    export_dir.mkdir(exist_ok=True, parents=True)

    # Init before processing each lang separately
    logger.info(
        f"Computing neighbor tokens on wikimedia/wikipedia with {NUM_WIKI_ARTICLES} random articles per language."
    )

    # Iterating over languages
    for lang in AUTHORIZED_LANG:
        if only_lang is not None and lang != only_lang:
            continue

        # Export path
        gc.collect()
        logger.info(f"Processing lang={lang}")
        export_path = (
            export_dir
            / f"{model_name}_{lang}{POSITION_TO_SUFFIX[position]}.pkl".replace(
                "/", "__"
            )
        )
        if export_path.is_file() and not force_recompute:
            logger.info(f"Replaceability file already exists: {export_path}")
            continue

        # Loading Wiki dataset
        ds = (
            load_dataset("wikimedia/wikipedia", f"20231101.{lang}")["train"]
            .shuffle(seed=42)
            .select(list(range(int(NUM_WIKI_ARTICLES))))
        )
        gc.collect()

        # Parralelizm
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Splitting the dataset
        num_proc = os.cpu_count() if num_proc is None else num_proc
        logger.info(f"Running on {num_proc} processes")
        num_sample = len(ds)
        start_stop_idx = [
            (k * (num_sample // num_proc), (k + 1) * (num_sample // num_proc))
            for k in range(num_proc)
        ]
        start_stop_idx[-1] = (start_stop_idx[-1][0], num_sample)
        process_args = [
            (ds.select(range(start, stop)).to_dict()["text"], model_name, idx, position)
            for idx, (start, stop) in enumerate(start_stop_idx)
        ]

        # Deploying the pool
        with Pool(num_proc) as p:
            per_chunk_result = p.map(compute_neighbors, process_args)

        # Merging
        logger.info(f"Merging result of all processes")
        final_neighbors = [defaultdict(int) for _ in range(len(per_chunk_result[0]))]

        for local_neighbors in per_chunk_result:
            for token_idx, token_neighbors in enumerate(local_neighbors):
                for k, v in token_neighbors.items():
                    final_neighbors[token_idx][k] += v

        # Saving
        logger.info(f"Saving token neighbors at {export_path}")
        with export_path.open("wb") as f:
            pickle.dump(final_neighbors, f)

        # Cleaning
        del per_chunk_result
        del final_neighbors
        del ds
        gc.collect()

    # Merging all lang - needed?
    export_path = (
        export_dir
        / f"{model_name}{POSITION_TO_SUFFIX[position]}.pkl".replace("/", "__")
    )
    if export_path.is_file() and not force_recompute:
        logger.info(f"Replaceability file already exists: {export_path}")
        return

    # Merging all lang
    logger.info(f"Merging all lang")
    final_neighbors = None
    for lang in AUTHORIZED_LANG:
        export_path = (
            export_dir
            / f"{model_name}_{lang}{POSITION_TO_SUFFIX[position]}.pkl".replace(
                "/", "__"
            )
        )
        if not export_path.is_file():
            raise ValueError(f"Unfound export path: {export_path}")

        with export_path.open("rb") as f:
            local_neighbors = pickle.load(f)
            logger.debug(f"Loaded {len(local_neighbors)} neighbors from {export_path}")

        # Merging
        if final_neighbors is None:
            final_neighbors = local_neighbors
        else:
            for token_idx, token_neighbors in enumerate(local_neighbors):
                for k, v in token_neighbors.items():
                    if k in final_neighbors[token_idx]:
                        final_neighbors[token_idx][k] += v
                    else:
                        final_neighbors[token_idx][k] = v

    export_path = (
        export_dir
        / f"{model_name}{POSITION_TO_SUFFIX[position]}.pkl".replace("/", "__")
    )
    logger.info(f"Saving token neighbors at {export_path}")
    with export_path.open("wb") as f:
        pickle.dump(final_neighbors, f)

    logger.info(f"Merging done, token replaceability completed!")


@app.command()
def clip_neighbors(max_neighbors: int = 100, position: int = 0):
    """Clips the neighbors dict to a maximum number of neighbors."""

    for model_name in LLMS_INFOS:
        for lang in [None] + list(AUTHORIZED_LANG):
            logger.debug(f"Clipping neighbors with model={model_name}, lang={lang}")
            try:
                neighbors = load_neighbors(model_name, lang=lang, position=position)
            except ValueError:
                logger.info(f"Did not find any neighbor data for model={model_name}, lang={lang}. Skipping.")
                continue

            # Computing
            new_neighbors = []
            for token_neighbors in tqdm(neighbors):
                sorted_neighbors = sorted(
                    token_neighbors, key=lambda k: token_neighbors[k], reverse=True
                )[:max_neighbors]
                new_neighbors.append(
                    {tok: token_neighbors[tok] for tok in sorted_neighbors}
                )

            # Saving
            export_dir = paths.output / "replaceability"
            export_dir.mkdir(exist_ok=True, parents=True)
            if lang is None:
                export_path = (
                    export_dir
                    / f"{model_name}{POSITION_TO_SUFFIX[position]}.pkl".replace(
                        "/", "__"
                    )
                )
            else:
                export_path = (
                    export_dir
                    / f"{model_name}_{lang}{POSITION_TO_SUFFIX[position]}.pkl".replace(
                        "/", "__"
                    )
                )

            # Reading and outputing
            logger.debug(f"Saving to {export_path}")
            with export_path.open("wb") as f:
                pickle.dump(new_neighbors, f)
