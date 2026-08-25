import os
import typing as t
from time import time

from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from ..generation import get_all_generations, get_generation_from_idx
from ..utils.typer_app import app
from .annot_procedure import ANNOTATION_INSTRUCTIONS
from .get_wiki import download_wiki_language, get_question_per_language
from .gpt_evaluator import add_gpt_labels, log_costs
from .human_evaluator import add_human_labels
from .utils import (
    get_all_annotations,
    get_annotation_from_idx,
    get_fixed_random_annotations,
    get_labelfree_annotation,
)


@app.command()
def segment_all_generations(force_recompute: bool = False) -> None:
    """Computes the segmentation of all existing annotations."""
    generations_to_segment = get_all_generations(reload=True)
    logger.info(
        f"Computing segmentation for all {len(generations_to_segment)} generations with force_recompute={force_recompute}"
    )

    # Remove some of them?
    if not force_recompute:
        existing_annotation_id = set(
            annotation_id for annotation_id in get_all_annotations(reload=True)
        )

        generations_to_segment_list = [
            generation
            for generation_id, generation in generations_to_segment.items()
            if generation_id not in existing_annotation_id
        ]

        # Logging
        logger.info(
            f"It remains {len(generations_to_segment_list)} after filtering existing generations"
        )
    else:
        generations_to_segment_list = [
            generation for _, generation in generations_to_segment.items()
        ]

    for generation in tqdm(generations_to_segment_list):
        result = get_labelfree_annotation(generation)
        result.write_disk()


@app.command()
def segment_some_generations(idx_path: str) -> None:
    """Computes the segmentation of all generation whose idx is a line of "idx_path" file."""
    # Reading
    logger.info(f"Reading {idx_path}...")
    generations_to_segment_list = []
    with open(idx_path, "r") as f:
        for line in f:
            if line:
                generations_to_segment_list.append(
                    get_generation_from_idx(line.strip())
                )

    # Segmenting
    logger.info(
        f"Computing segmentation for all {len(generations_to_segment_list)} generations."
    )
    for generation in tqdm(generations_to_segment_list):
        result = get_labelfree_annotation(generation)
        result.write_disk()


@app.command()
def download_all_wiki_references():
    """Downloads the wiki content for all questions from all languaes in the datasets."""
    for lang, question_list in tqdm(get_question_per_language().items()):
        download_wiki_language(question_list, lang)

    logger.info(f"Download wiki content: done")


@app.command()
def add_all_gpt_labels(
    force_recompute: bool = False,
    max_annot: t.Optional[int] = None,
    model: str = None,
    lang: str = None,
):
    """Adds GPT labels to all annotations with stopwords method."""

    # Loading annotations (they don't have labels yet)
    annotations_to_label = get_all_annotations(reload=True)

    # Max annotation?
    if max_annot is not None:
        annotations_to_label = {
            key: value for key, value in list(annotations_to_label.items())[:max_annot]
        }

    # Filter lang?
    if lang is not None:
        annotations_to_label = {
            key: value
            for key, value in annotations_to_label.items()
            if value.generation.lang == lang
        }

    # Iterating
    try:
        Parallel(n_jobs=os.cpu_count(), backend="multiprocessing")(
            delayed(add_gpt_labels)(
                annotation, force_recompute=force_recompute, model=model
            )
            for _, annotation in annotations_to_label.items()
        )

    except Exception as e:
        log_costs()
        raise e

    # Logging costs
    log_costs()


@app.command()
def add_some_gpt_labels(
    lang: str = "en",
    size: int = 100,
    force_recompute: bool = False,
    annot_id: str | None = None,
    idx_path: str | None = None,
    model: str = None,
):
    """Add some GPT labels, chosen similarly to human's one, to compare."""

    # Loading annotations (they don't have labels yet)
    if annot_id is None and idx_path is None:
        logger.info(f"Annotating {size} samples random samples in {lang}...")
        annotations_to_label = get_fixed_random_annotations(lang=lang, size=size)
    elif annot_id is not None:
        logger.info(f"Annotating samples {annot_id}")
        annotations_to_label = {
            annot_id: get_annotation_from_idx(annot_id, reload_all=True)
        }
    elif idx_path is not None:
        logger.info(f"Reading annotation idx from {idx_path}")
        annotations_to_label = {}
        with open(idx_path, "r") as f:
            for line in f:
                if line:
                    annotations_to_label[line.strip()] = get_annotation_from_idx(
                        line.strip()
                    )

    # Iterating
    logger.info(f"Starting annotation of {len(annotations_to_label)} annotations...")
    try:
        Parallel(n_jobs=os.cpu_count(), backend="multiprocessing")(
            delayed(add_gpt_labels)(
                annotation, force_recompute=force_recompute, model=model
            )
            for _, annotation in annotations_to_label.items()
        )

    except Exception as e:
        log_costs()
        raise e

    # Logging costs
    log_costs()


@app.command()
def add_human_annotations(
    lang: str = "en",
    size: int = 100,
    annotator: str = "an0",
    force_recompute: bool = False,
    annot_id: str | None = None,
):
    """Add humuman evaluations."""

    # Loading annotations (they don't have labels yet)
    if annot_id is None:
        annotations_to_label = get_fixed_random_annotations(
            lang=lang, size=size, filter_gpt_accordance=True
        )
    else:
        annotations_to_label = {
            annot_id: get_annotation_from_idx(annot_id, reload_all=True)
        }

    # Intro message
    print(ANNOTATION_INSTRUCTIONS)

    # Iterating
    t0 = time()
    count_annotations = 0
    try:
        for count, (_, annotation) in enumerate(annotations_to_label.items()):
            if (not force_recompute) and (annotator in annotation.labels):
                print(
                    f"Skipping already-annotated annotation: {annotation.generation_id}"
                )
                continue

            print(f"{40*'='}\n\nAnnotation {count+1} / {size}\n")
            add_human_labels(annotation, annotator)
            count_annotations += 1

    except Exception as e:
        t1 = time()
        rate = (t1 - t0) / count_annotations if count_annotations else 0
        print(
            f"Thanks a lot! You annotated {count_annotations} in {t1-t0:.2f}sec [{rate:.2f}sec/annot]"
        )
        raise e

    t1 = time()
    rate = (t1 - t0) / count_annotations if count_annotations else 0
    print(
        f"Thanks a lot! You annotated {count_annotations} in {t1-t0:.2f}sec [{rate:.2f}sec/annot]"
    )


@app.command()
def remove_annotations(
    lang: str | None = None,
    annotator: str | None = "an0",
    annot_id: str | None = None,
):
    """Removes annotations."""

    # Loading annotations (they don't have labels yet)
    if annot_id is None:
        annotations_to_remove = get_all_annotations(reload=True)
    else:
        annotations_to_remove = {
            annot_id: get_annotation_from_idx(annot_id, reload_all=True)
        }

    # Iterating
    for annotation in annotations_to_remove.values():
        if lang is not None and annotation.generation.lang != lang:
            continue

        if annotator in annotation.labels:
            del annotation.labels[annotator]
            annotation.write_disk()
