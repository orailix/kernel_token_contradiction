import pickle
import typing as t
from collections import defaultdict
from functools import lru_cache

from tqdm import tqdm

from ..generation.utils import get_tokenizer
from ..utils import paths

POSITION_TO_SUFFIX = {0: "", 1: "_right", 2: "_left"}


def compute_neighbors(process_args) -> t.List[t.Dict[int, int]]:
    """Returns a list (same length as the tokenizer) of dict.
    result[idx_source][idx_target] count the number of time `idx_target` token
    appeared right after `idx_source` in the Wikipedia text processed"""

    # Unpacking
    ds_chunk, model_name, process_idx, position = process_args

    # Tokenizer
    tokenizer = get_tokenizer(model_name)

    # Init result
    neighbors = [defaultdict(int) for _ in range(len(tokenizer))]

    for sample in tqdm(ds_chunk, disable=process_idx != 0):
        encoded = tokenizer.encode(sample)

        for tok_idx in range(1, len(encoded) - 1):
            before, target, after = encoded[tok_idx - 1 : tok_idx + 2]
            if position == 0:
                neighbors[target][before] += 1
                neighbors[target][after] += 1
            elif position == 1:
                neighbors[target][after] += 1
            elif position == -1:
                neighbors[target][before] += 1
            else:
                raise ValueError(f"Unknown position: {position}")

    return neighbors


def load_neighbors(
    model_name: str, lang: str = None, position: int = 0
) -> t.List[t.Dict[int, int]]:
    """Returns a list (same length as the tokenizer) of dict.
    result[idx_source][idx_target] count the number of time `idx_target` token
    appeared right after `idx_source` in the Wikipedia text processed"""

    # Are available?
    export_dir = paths.output / "replaceability"
    export_dir.mkdir(exist_ok=True, parents=True)
    if lang is None:
        export_path = (
            export_dir
            / f"{model_name}{POSITION_TO_SUFFIX[position]}.pkl".replace("/", "__")
        )
    else:
        export_path = (
            export_dir
            / f"{model_name}_{lang}{POSITION_TO_SUFFIX[position]}.pkl".replace(
                "/", "__"
            )
        )
    if not export_path.is_file():
        raise ValueError(
            f"We did not found any replaceability files at {export_path}.\n"
            "Maybe you forgot to compute them with `compute-replaceability` command?"
        )

    # Reading and outputing
    with export_path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=256)
def load_sorted_neighbors(
    model_name: str, lang: str = None, position: int = 0
) -> t.List[t.List[int]]:
    """Returns a list (same length as the tokenizer) of list.
    result[idx_source] contains the ordered list of most frequent token
    neighbors in the Wikipedia text processed."""

    neighbors = load_neighbors(model_name=model_name, lang=lang, position=position)
    result = []
    for tok in range(len(neighbors)):
        token_neighbors = neighbors[tok]
        result.append(
            sorted(token_neighbors, key=lambda k: token_neighbors[k], reverse=True)
        )

    return result
