import json
import typing as t

from loguru import logger
from tqdm import tqdm

from ..annotation import get_all_annotations, get_annotation_from_idx
from ..generation import get_tokenizer
from ..signals import UQSignal
from ..utils import paths
from .signal_evaluation import SignalEvaluation, MANUAL_ADDON


class NoChunkEval(Exception):
    pass


def compute_evaluation(
    uq_signal: UQSignal,
    verbose: bool = False,
    lang: t.Optional[str] = None,
    manual: int = 0,
    include_much: bool = True,
    include_mushroom: bool = False,
) -> SignalEvaluation:
    """Computes a SignalEvaluation object for this signal using all evaluations."""

    # Logging
    logger.info(f"Computing evaluation on all annotations: {uq_signal.signal_name}")

    # Init evaluation
    signal_evaluation: SignalEvaluation = None

    # One reload to rule them all
    _ = get_all_annotations(reload=True)

    # Iterating
    count_annotation = 0
    count_chunks = 0
    for signal_value in tqdm(uq_signal.get_all_signal_values(), disable=not verbose):

        # Lang filter?
        if (
            lang is not None
            and get_annotation_from_idx(signal_value.generation_id).generation.lang
            != lang
        ):
            continue

        # Init signal evaluation at fist signal_value
        if signal_evaluation is None:
            signal_evaluation = SignalEvaluation(
                signal_name=uq_signal.signal_name,
                lang=lang,
                manual=manual,
                include_much=include_much,
                include_mushroom=include_mushroom,
            )

        # Getting the annotations and snippets
        annotation = get_annotation_from_idx(signal_value.generation_id)

        # Nullifying the values_np for stopwords
        values_np = signal_value.nullify_stopwords().values_np

        # MUCH case
        if annotation.generation.generation_cfg.seed == 1234:
            if not include_much:
                continue

            # Skipping if this annotator is not on this annotation
            if (
                "gpt-4o" not in annotation.labels
                or "gpt-4.1" not in annotation.labels
                or annotation.labels["gpt-4o"] != annotation.labels["gpt-4.1"]
            ):
                continue

            # Manual filter
            if manual == 1 and (
                "an0" not in annotation.labels
                or "an1" not in annotation.labels
                or annotation.labels["an0"] != annotation.labels["an1"]
            ):
                continue
        
            if manual == -1 and (
                "an0" in annotation.labels
                or "an1" in annotation.labels
            ):
                continue


            evaluation_labels = annotation.labels["gpt-4o"][:-1]

        # Mushroom case
        elif "mushroom" in annotation.labels:
            if not include_mushroom or manual == -1:
                continue

            evaluation_labels = annotation.labels["mushroom"][:-1]

        else:
            raise ValueError(f"Sample is neither MUCH nor MUSHROOM")

        # Count
        count_annotation += 1

        # Iterating over chunks - skipping the last one because it's EOS
        for chunk_tokens, chunk_label in zip(
            annotation.token_chunks[:-1], evaluation_labels
        ):
            # Fetching start and stop
            start, stop = chunk_tokens[0], chunk_tokens[-1] + 1

            # Count
            count_chunks += 1

            # Hallunication parts
            if chunk_label == -1:
                signal_chunk = values_np[start:stop]
                signal_evaluation.update_values_in(signal_chunk)

            # Non-hallucination
            else:
                signal_chunk = values_np[start:stop]
                signal_evaluation.update_values_out(signal_chunk)

    # Raise error if there are no chunks
    if count_chunks == 0:
        raise NoChunkEval

    # Logging counts
    logger.info(
        f"Evaluation computed on {count_annotation} annotations, including {count_chunks} chunks"
    )

    # Output
    return signal_evaluation


def get_evaluation(
    uq_signal: UQSignal,
    force_recompute: bool = False,
    verbose: bool = False,
    lang: t.Optional[str] = None,
    manual: int = 0,
    include_much: bool = True,
    include_mushroom: bool = False,
) -> SignalEvaluation:
    """Gets the evaluation of the signal. By default, tries to load from disk."""

    # File exists and we should use it?
    lang_addon = "" if lang is None else f"_{lang}"
    manual_addon = MANUAL_ADDON[manual]
    much_addon = "" if not include_much else "_much"
    mushroom_addon = "" if not include_mushroom else "_mushroom"
    output_path = (
        paths.output
        / "evaluations"
        / f"{uq_signal.signal_name}{lang_addon}{much_addon}{manual_addon}{mushroom_addon}.json"
    )
    if output_path.exists() and not force_recompute:
        logger.info(f"Loading evaluation from {output_path}")
        return SignalEvaluation.from_dict(json.loads(output_path.read_text()))

    # Else, recompute, save and return
    result = compute_evaluation(
        uq_signal,
        verbose,
        lang,
        manual=manual,
        include_much=include_much,
        include_mushroom=include_mushroom,
    )
    result.write_disk()
    return result
