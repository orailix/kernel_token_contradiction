import json
import shutil
import typing as t

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from ..annotation import Annotation, get_all_annotations
from ..generation import Generation, GenerationConfig, get_all_generations
from ..questions import Question
from ..signals import UQSignalValue
from ..utils import paths
from ..utils.constants import (
    HF_CONFIGS_PATH,
    HF_MUCH_PATH,
    HF_SIGNALS_PATH,
    HF_TRASH_PATH,
)


def export_much_data():
    """Export MUCH dataset in outputs/much"""
    # Gathering
    export_train = []
    export_test = []
    export_trash = []
    for _, annotation in get_all_annotations(reload=True).items():

        if annotation.labels["gpt-4o"] != annotation.labels["gpt-4.1"]:
            export_trash.append(get_annotation_export(annotation, is_trash=True))
        elif "an1" not in annotation.labels:
            export_train.append(get_annotation_export(annotation))
        else:
            export_test.append(get_annotation_export(annotation))

    # Exporting train and test
    export_dir = paths.output / "much"
    export_dir.mkdir(parents=True, exist_ok=True)

    for export_list, name in zip(
        [export_train, export_test, export_trash], ["train", "test", "trash"]
    ):
        logger.info(
            f"Exporting {len(export_list)} annotations in {export_dir / f'{name}.jsonl'}"
        )
        with (export_dir / f"{name}.jsonl").open("w") as f:
            f.write("\n".join([json.dumps(exportable) for exportable in export_list]))

    # Exporting configs
    existing_config_ids = set()
    for exportable in export_train:
        existing_config_ids.add(exportable["generation_config"])
    existing_config_ids = sorted(list(existing_config_ids))
    all_configs_as_dicts = [
        GenerationConfig.autoconfig(config_id).get_as_dict()
        for config_id in existing_config_ids
    ]
    for config_id, item in zip(existing_config_ids, all_configs_as_dicts):
        del item["_base_output_dir"]
        item["generation_config"] = config_id

    logger.info(
        f"Exporting {len(all_configs_as_dicts)} configurations in {export_dir / 'configs.jsonl'}"
    )
    with (export_dir / "configs.jsonl").open("w") as f:
        f.write(
            "\n".join([json.dumps(exportable) for exportable in all_configs_as_dicts])
        )

    # Logging
    logger.info("End of MUCH export!")


def get_annotation_export(annotation: Annotation, is_trash: bool = False) -> dict:
    """Gets an exportable dict representing the annotation."""

    # Dealing with the logits
    logits_token_ids = []
    logits_values = []
    for topn_dict in annotation.generation.topn_prob_dicts:
        tokens_ids_to_add = sorted(topn_dict, key=topn_dict.__getitem__, reverse=True)
        values_to_add = [topn_dict[token_id] for token_id in tokens_ids_to_add]
        logits_token_ids.append(tokens_ids_to_add)
        logits_values.append(values_to_add)

    # Forming the result
    result = dict(
        generation_id=annotation.generation_id,
        generation_config=annotation.generation.generation_config,
        lang=annotation.generation.lang,
        wiki_url=annotation.generation.wiki_url,
        prompt=annotation.generation.prompt,
        output=annotation.generation.output,
        output_tokens=annotation.generation.output_tokens,
        logits_token_ids=logits_token_ids,
        logits_values=logits_values,
        token_chunks=annotation.token_chunks,
        labels=annotation.labels["gpt-4o"],
        human_labels_0=annotation.labels["an0"] if "an0" in annotation.labels else [0],
        human_labels_1=annotation.labels["an1"] if "an1" in annotation.labels else [0],
    )

    # Is trash?
    if is_trash:
        result["gpt-4o"] = annotation.labels["gpt-4o"]
        result["gpt-4.1"] = annotation.labels["gpt-4.1"]
        del result["labels"]

    return result


def import_much_data(force_reimport: bool, include_trash: bool = False):
    """Imports MUCH data

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

    # Checks if MUCH is already imported
    if force_reimport:
        if (paths.output / "generations").is_dir():
            shutil.rmtree(paths.output / "generations")
        if (paths.output / "annotations").is_dir():
            shutil.rmtree(paths.output / "annotations")
    else:
        if (len(get_all_annotations()) == 4873 and not include_trash) or (
            len(get_all_annotations()) == 6448 and include_trash
        ):
            logger.info(f"It seems that MUCH is already imported, skipping import.")
            return
        else:
            logger.info(f"You do not have all samples, re-importing MUCH data...")
            return import_much_data(force_reimport=True, include_trash=include_trash)

    # Trash warning
    if include_trash:
        logger.warning(
            "You are importing MUCH 'trash' split. This dataset is made available for "
            "research reproducibility and SHOULD NOT be used for claim-level UQ evaluation."
        )

    # Importing the configs
    logger.info(f"Importing MUCH configs...")
    much_configs = load_dataset(HF_CONFIGS_PATH, split="train")
    for item in much_configs:
        gen_id = item["generation_config"]
        del item["generation_config"]

        logger.debug(f"Importing {gen_id}")
        imported_cfg = GenerationConfig(**item)

        if gen_id != imported_cfg.id:
            logger.warning(
                f"Inconsistent generation config ID: {gen_id} (dataset) != {imported_cfg.id} (local)"
            )

    # Importing annotations and generations
    logger.info(f"Importing MUCH generations and annotations...")
    much_dataset = load_dataset(HF_MUCH_PATH)
    split_container = [much_dataset["train"], much_dataset["test"]]
    split_names = ["train", "test"]
    if include_trash:
        split_container.append(load_dataset(HF_TRASH_PATH, split="train"))
        split_names.append("trash")

    for split, name in zip(split_container, split_names):

        logger.debug(f"Importing the generations of {name} split...")
        for sample in tqdm(split):
            # Re-building the topn_prob_dicts ...
            topn_prob_dicts = []
            for line_token_ids, line_values in zip(
                sample["logits_token_ids"], sample["logits_values"]
            ):
                topn_prob_dicts.append(
                    {
                        token_id: value
                        for token_id, value in zip(line_token_ids, line_values)
                    }
                )

            # Forming the generation
            generation = Generation(
                generation_config=sample["generation_config"],
                prompt=sample["prompt"],
                output=sample["output"],
                lang=sample["lang"],
                output_tokens=sample["output_tokens"],
                wiki_url=sample["wiki_url"],
                topn_prob_dicts=topn_prob_dicts,
            )
            generation.write_disk()

            # Sanity check
            if generation.id != sample["generation_id"]:
                logger.warning(
                    f"Inconsistent generation ID: {sample['generation_id']} (dataset) != {generation.id} (local) for generation\n{generation}"
                )

        # Importing the annotations
        # We do all generations and THEN all annotations because the annotation process
        # uses `annotation.generation` property, which would need to re-loads all generations
        # to find the correct one each time an annotation needs to be created.
        # This is inefficient. Consequently, we first write all generations, then read them
        # once, and then write all annotations.
        logger.debug(f"Importing the annotations of {name} split...")
        _ = get_all_generations(reload=True)
        for sample in tqdm(split):
            # Forming the annotation
            if "labels" in sample:  # Normal samples in MUCH
                labels = {"gpt-4o": sample["labels"], "gpt-4.1": sample["labels"]}
            else:  # Trash samples that got filtered out from MUCH
                labels = {"gpt-4o": sample["gpt-4o"], "gpt-4.1": sample["gpt-4.1"]}

            # Human-annotated samples
            if sample["human_labels_0"] != [0]:
                labels["an0"] = sample["human_labels_0"]
                labels["an1"] = sample["human_labels_1"]

            Annotation(
                generation_id=sample["generation_id"],
                token_chunks=sample["token_chunks"],
                labels=labels,
            ).write_disk()

    # Importing the questions
    logger.info(f"Importing the questions contained in imported samples")
    logger.debug(f"Gathering all questions")
    detected_questions = set()
    for _, annotation in tqdm(get_all_annotations(reload=True).items()):
        detected_questions.add(
            (
                annotation.generation.lang,
                annotation.generation.prompt,
                annotation.generation.wiki_url,
            )
        )

    logger.debug(f"Writing questions on disk")
    for lang, prompt, wiki_url in tqdm(detected_questions):
        Question(lang=lang, prompt=prompt, wiki_url=wiki_url).write_disk()

    # Logging
    logger.info("End of MUCH import!")


def import_much_signals():
    """Imports MUCH signals"""

    all_signals = load_dataset(HF_SIGNALS_PATH, split="train")
    logger.info(f"Importing the {len(all_signals)} signal values of MUCH...")
    for signal_value in tqdm(all_signals):
        UQSignalValue(**signal_value).write_disk()
