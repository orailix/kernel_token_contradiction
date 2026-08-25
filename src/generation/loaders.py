import json
import typing as t

from .utils import Generation, GenerationConfig


def get_generations_from_config(generation_cfg=GenerationConfig) -> t.List[Generation]:
    """Load all generations associated with a specific GenerationConfig."""
    output_file = generation_cfg.get_output_dir() / "generations.jsonl"
    if not output_file.is_file():
        return []

    result = []
    with output_file.open("r") as f:
        for line in f:
            if line.strip():
                result.append(Generation(**json.loads(line)))

    return result


_all_generations_cache = {}


def get_all_generations(reload: bool = False) -> t.Dict[str, Generation]:
    """Load all generations from all available configs."""
    if reload:
        _all_generations_cache.clear()

        for generation_cfg in GenerationConfig.get_all_configs():
            for generation in get_generations_from_config(generation_cfg):
                _all_generations_cache[generation.id] = generation

        return _all_generations_cache

    elif len(_all_generations_cache) != 0:
        return _all_generations_cache

    else:
        return get_all_generations(reload=True)


def get_generation_from_idx(generation_id: str, reload_all: bool = False) -> Generation:
    """Retrieve a generation by exact or prefix-matching its ID."""
    existing_generations = get_all_generations()

    # Exact match?
    if generation_id in existing_generations:
        return existing_generations[generation_id]

    # Searching...
    candidates = []
    for candidate_id, generation in existing_generations.items():
        if candidate_id[: len(generation_id)] == generation_id:
            candidates.append(generation)

    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple generations whose prefix is the one you provided: {generation_id}"
        )

    if len(candidates) == 1:
        return candidates[0]

    if not reload_all:
        return get_generation_from_idx(generation_id, reload_all=True)

    raise ValueError(
        f"Found no generations whose prefix is the one you provided: {generation_id}"
    )
