import numpy as np
from loguru import logger
from tqdm import tqdm

from ..annotation import get_all_annotations, Annotation
from ..generation import get_all_generations, Generation
from ..utils.typer_app import app
from .import_mushroom import (
    UNSUPPORTED_MODELS,
    LogitNotInTop24,
    clear_mushroom_model_cache,
    get_generation_and_annotation,
    get_mushroom_model,
    get_mushroom_per_lang,
    recompute_one_mushroom_label,
)
from .utils import export_much_data, import_much_data, import_much_signals


@app.command()
def export_much():
    """Export MUCH files in ./output/much"""
    export_much_data()


@app.command()
def import_much(force_reimport: bool = False, include_trash: bool = False):
    """Import MUCH files from Hugging Face Hub.

    Parameter
    ---------
    force_reimport : bool
        If True, existing annotations and generations will be removed before the import.
    include_trash : bool
        By default, only the final samples of MUCH dataset are imported.
        If include_trash is True, the 1575 samples that were filtered out from MUCH
        for quality reasons will also be imported. Note that these samples SHOULD NOT
        be used for evaluated claim-level UQ methods.
    """
    import_much_data(force_reimport, include_trash=include_trash)


@app.command()
def import_signals():
    """Import MUCH signals from Hugging Face Hub."""
    import_much_signals()


@app.command()
def import_mushroom(lang: str):
    """Import Mu-Shroom for one language."""
    logger.info(f"Importing Mu-shroom: {lang}")
    mushrooms = get_mushroom_per_lang()
    all_models = np.unique(mushrooms[lang]["model_id"])

    for processing_model_id in all_models:
        if processing_model_id in UNSUPPORTED_MODELS:
            logger.debug(f"Skipping unsupported model: {processing_model_id}")
            continue

        logger.debug(f"Processing model: {processing_model_id}")
        ranks_to_process = []
        for rank in range(len(mushrooms[lang])):
            model_id = mushrooms[lang]["model_id"][rank]
            if model_id == processing_model_id:
                ranks_to_process.append(rank)

        generations_to_write: list[Generation] = []
        annotations_to_write: list[Annotation] = []

        for rank in tqdm(ranks_to_process):
            model_id = mushrooms[lang]["model_id"][rank]
            question = mushrooms[lang]["model_input"][rank]
            output_text = mushrooms[lang]["model_output_text"][rank]
            soft_labels = mushrooms[lang]["soft_labels"][rank]

            try:
                generation, annotation = get_generation_and_annotation(
                    lang, model_id, question, output_text, soft_labels
                )
            except LogitNotInTop24:
                continue
            generations_to_write.append(generation)
            annotations_to_write.append(annotation)

        # Writing
        for generation in generations_to_write:
            generation.generation_cfg.get_output_dir()
            generation.write_disk()
        _ = get_all_generations(reload=True)
        for annotation in annotations_to_write:
            annotation.write_disk()

        # Clearing model cache
        clear_mushroom_model_cache()


@app.command()
def recompute_mushroom_labels():
    """Recomputes Mu-shroom labels for all annotations."""

    annotations_to_process = []
    for _, annotation in get_all_annotations().items():
        if "mushroom" in annotation.labels:
            annotations_to_process.append(annotation)

    logger.info(f"Re-computing label for {len(annotations_to_process)} annotations")
    for annotation in tqdm(annotations_to_process):
        recompute_one_mushroom_label(annotation)
