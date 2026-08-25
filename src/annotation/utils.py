import json
import typing as t
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from filelock import FileLock
from much_segmenter import much_segmentation

from ..generation import Generation, get_generation_from_idx, get_tokenizer
from ..utils import paths

annotation_output = paths.output / "annotations"


@dataclass
class Annotation:
    """
    Stores the result of a annotation process.

    Attributes:
        generation_id (str): IDX of the annotation from which annotation is derived.
        token_chunks (List[np.ndarray]): List of token indices of each chunks.
            All tokens in the array are included
        labels (List[int]]): Optional label indices corresponding to each chunk
            Labels: 1=hallucination, 0=not hallucination
    """

    generation_id: str
    token_chunks: list[list[int]] = field(default_factory=list)
    labels: dict[str, list[int]] = field(default_factory=dict)

    @cached_property
    def generation(self) -> Generation:
        return get_generation_from_idx(self.generation_id)

    @cached_property
    def output_path(self) -> Path:
        annotation_output.mkdir(exist_ok=True, parents=True)
        return annotation_output / f"{self.generation.generation_config}.jsonl"

    def write_disk_no_check(self):
        """Writes on the disk without checking if it already exists."""
        is_nonempty = self.output_path.exists() and self.output_path.stat().st_size > 0

        with self.output_path.open("a") as f:
            if is_nonempty:
                f.write("\n")
            f.write(json.dumps(asdict(self)))

    def write_disk(self):
        """Writes on the disk and checks if it exists."""
        with FileLock(self.output_path.with_suffix(".lock")):
            # Write to disk
            self.write_disk_no_check()

            # Remove olders
            lines = self.output_path.read_text().split("\n")
            lines_filtered = [
                l
                for idx, l in enumerate(lines)
                if (l[19 : 19 + 22] != self.generation_id or idx == len(lines) - 1)
            ]
            self.output_path.write_text("\n".join(lines_filtered))

    def get_repr_string(
        self,
        with_labels: bool = False,
        with_question: bool = False,
        remove_latest: bool = False,
        remove_linebreaks: bool = True,
    ) -> str:
        tokenizer = get_tokenizer(self.generation.generation_cfg.model_name)
        result = (
            f"Annotation[{self.generation.id}] {self.generation.prompt}"
            if with_question
            else ""
        )
        for idx_chunk, chunk in enumerate(self.token_chunks):
            if remove_latest and idx_chunk == len(self.token_chunks) - 1:
                continue

            # Label
            if len(self.labels) == 0:
                label = "<empty>"
            else:
                label = ""
                for annotator in sorted(self.labels, reverse=True):
                    label += f"|{annotator}:{self.labels[annotator][idx_chunk]:2}"
                label = "<" + label[1:] + ">"

            chunk_str = tokenizer.decode(
                self.generation.output_tokens[chunk[0] : chunk[-1] + 1]
            )
            if remove_linebreaks:
                chunk_str = chunk_str.replace("\n", " ")
            line_break = "\n" if result != "" else ""

            if with_labels:
                result += f"{line_break}#{idx_chunk:2} Labels={label} : {chunk_str}"
            else:
                result += f"{line_break}#{idx_chunk:2} : {chunk_str}"

        return result

    def __repr__(self) -> str:
        return self.get_repr_string(
            with_labels=True, with_question=True, remove_latest=False
        )


_all_annotations_cache = {}


def get_all_annotations(reload: bool = False) -> t.Dict[str, Annotation]:
    """Gets all existing annotations that are stored on the disk."""
    if reload:
        _all_annotations_cache.clear()

        # Iterating on files in annotation_output
        annotation_output.mkdir(exist_ok=True, parents=True)
        for child in annotation_output.iterdir():
            if child.is_file() and child.suffix == ".jsonl":

                # Open the annotation file and read all annotations
                with child.open("r") as f:
                    for line in f:
                        if line.strip():
                            annotation = Annotation(**json.loads(line))
                            _all_annotations_cache[
                                annotation.generation_id
                            ] = annotation

        return _all_annotations_cache

    elif len(_all_annotations_cache) != 0:
        return _all_annotations_cache

    else:
        return get_all_annotations(reload=True)


def get_fixed_random_annotations(
    lang: str,
    size: int,
    filter_gpt_accordance: bool = False,
    seed: int = 53,
) -> t.Dict[str, Annotation]:
    """Gets a dict or random annotations, but in a fixed order so we always get the same."""
    all_annotations = get_all_annotations(reload=True)
    all_annotations_filtered = {
        key: value
        for key, value in all_annotations.items()
        if value.generation.lang == lang
    }

    # Shuffling idx
    all_annot_idx = np.array(sorted(list(all_annotations_filtered)))
    rg = np.random.RandomState(seed=seed)
    rg.shuffle(all_annot_idx)

    # Filter GPT accordance:
    if filter_gpt_accordance:
        all_annot_idx = [
            annot_id
            for annot_id in all_annot_idx
            if (
                "gpt-4o" in all_annotations[annot_id].labels
                and "gpt-4.1" in all_annotations[annot_id].labels
                and all_annotations[annot_id].labels["gpt-4o"]
                == all_annotations[annot_id].labels["gpt-4.1"]
            )
        ]

    # Selecting
    selected_annot_idx = set(all_annot_idx[:size])
    return {
        key: value
        for key, value in all_annotations_filtered.items()
        if key in selected_annot_idx
    }


def get_annotation_from_idx(
    generation_id: str,
    reload_all: bool = False,
) -> Annotation:
    """Finds a annotation if it exists"""
    existing_annotations = get_all_annotations(reload=reload_all)

    # Exact match?
    if generation_id in existing_annotations:
        return existing_annotations[generation_id]

    # Searching
    candidates = []
    for candidate_id, annotation in existing_annotations.items():
        if candidate_id[: len(generation_id)] == generation_id:
            candidates.append(annotation)

    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple annotations whose prefix is the one you provided: {generation_id}"
        )

    if len(candidates) == 1:
        return candidates[0]

    if not reload_all:
        return get_annotation_from_idx(generation_id, reload_all=True)

    raise ValueError(
        f"Found no annotations whose prefix is the one you provided: {generation_id}"
    )


def get_labelfree_annotation(generation: Generation) -> Annotation:
    """Gets a label-free annotation, ready for annotating labels.

    Note that we do not fill the labels now. Only the stopword segmentation.
    """

    llm_tokenizer = get_tokenizer(generation.generation_cfg.model_name)
    token_chunks = much_segmentation(
        generation.output,
        llm_tokenizer=llm_tokenizer,
        precomputed_tokens=generation.output_tokens,
    )
    return Annotation(generation_id=generation.id, token_chunks=token_chunks)
