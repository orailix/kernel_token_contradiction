import os
import typing as t
from collections import Counter
from functools import lru_cache
from threading import Lock
from time import time

import numpy as np
import tiktoken
from joblib import Parallel, delayed
from loguru import logger
from much_segmenter import get_known_eos_or_eot
from openai import OpenAI
from pydantic import BaseModel, create_model

from ..generation import get_tokenizer
from .annot_procedure import ANNOTATION_INSTRUCTIONS, RAG_CHUNK_SIZE, RAG_TOP_K
from .get_wiki import get_wiki_content_from_cache
from .utils import Annotation

GPT_REDUNDANCY: int = 1
CHAT_MODELS = {
    "4o": "gpt-4o-2024-11-20",
    "4.1": "gpt-4.1-2025-04-14",
}
EMBED_MODEL = "text-embedding-3-large"
GPT_T0 = time()

# Costs monitoring
COSTS = {
    "embedding_tokens": 0,
    "embedding_cost": 0.0,
    "generation_input_tokens": 0,
    "generation_output_tokens": 0,
    "generation_cost": 0.0,
}
GPT_LOCK = Lock()

# Model pricing (as of 2025-08 — update if OpenAI changes it)
PRICES = {
    "text-embedding-3-large": {"input": 0.065 / 1_000_000},
    "gpt-4o-2024-11-20": {"input": 2.00 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4.1-2025-04-14": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
}


@lru_cache()
def get_client():
    return OpenAI()


def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE) -> t.List[str]:
    """Split text into approximately token-sized chunks with half-overlapping windows."""
    enc = tiktoken.encoding_for_model(EMBED_MODEL)
    tokens = enc.encode(text)

    chunks = []
    step = chunk_size  # full stride
    overlap = chunk_size // 2

    # primary chunks
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunks.append(enc.decode(chunk_tokens))

    # overlapping chunks (half-step shifted)
    for i in range(overlap, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunks.append(enc.decode(chunk_tokens))

    return chunks


@lru_cache(maxsize=4096)
def get_embedding(text: str) -> t.List[float]:
    """Gets the embedding of a text."""
    resp = get_client().embeddings.create(input=[text], model=EMBED_MODEL)

    with GPT_LOCK:
        tokens = resp.usage.total_tokens
        COSTS["embedding_tokens"] += tokens
        COSTS["embedding_cost"] += tokens * PRICES[EMBED_MODEL]["input"]

    return resp.data[0].embedding


def cosine_similarity(a: t.List[float], b: t.List[float]) -> float:
    """Computes the cosine similarity between norm-1 vectors"""
    a, b = np.array(a), np.array(b)
    return np.sum(a * b)


def rank_chunks_by_similarity(
    chunks: t.List[str], query_text: str, top_k: int = RAG_TOP_K
) -> t.List[str]:
    """Gets the top matching chunks."""
    try:
        query_emb = get_embedding(query_text)
    except Exception as e:
        logger.info(f"Exception with the following query: {query_text}")
        raise e

    chunks_embeddings = Parallel(n_jobs=os.cpu_count(), backend="threading")(
        delayed(get_embedding)(chunk) for chunk in chunks
    )

    chunk_scores = [
        (chunk, cosine_similarity(query_emb, chunks_embed))
        for chunk, chunks_embed in zip(chunks, chunks_embeddings)
    ]
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in chunk_scores[:top_k]]


class SegmentationFactualityList(BaseModel):
    """
    Represents the factuality score for each chunk in the segmented output.

    Scoring:
        -1 → Factually incorrect / contradicts previous ideas / unverifiable / approximative
         1 → Factually correct

    Attributes:
        factuality_scores (List[int]): A list of factuality scores, one per output chunk.
    """

    factuality_scores: t.List[int] = []


def make_annotation_factuality_model(n_chunks: int) -> t.Type[BaseModel]:
    fields = {
        f"score_chunk_{i}": (int, ...)  # require an int for each field
        for i in range(n_chunks)
    }
    model = create_model("SegmentationFactuality", **fields)
    model.__doc__ = (
        "Represents the factuality score for each chunk in the segmented output.\n\n"
        "Scoring:\n"
        "  -1 → Factually incorrect / contradicts previous ideas / unverifiable / approximative\n"
        "   1 → Factually correct\n\n"
        "Each attribute corresponds to one output chunk, e.g., score_chunk_0, score_chunk_1, ..."
    )

    return model


def get_one_gpt_evaluation(
    annotation: Annotation,
    model_name: str,
    seed: int = 0,
) -> SegmentationFactualityList:
    """Gets one evaluation for the score of each chunk."""

    main_content, infobox = get_wiki_content_from_cache(annotation.generation.wiki_url)
    chunks = chunk_text(main_content)

    # Step 2: Select relevant knowledge
    relevant_chunks = rank_chunks_by_similarity(
        chunks, annotation.generation.output, top_k=RAG_TOP_K
    )
    context_knowledge = "\n\n".join(relevant_chunks)

    # Step 3: Build GPT prompt
    system_message = "You are a factuality evaluation assistant."

    user_prompt = f"""{ANNOTATION_INSTRUCTIONS}

# Question

{annotation.generation.prompt}

# Model Output

{annotation.generation.output.strip()}

# Segmented Output

{annotation.get_repr_string(with_labels=False, remove_latest=True)}

# Reference Knowledge

## Wikipedia Infobox

{infobox}

## Relevant chunks from Wikipedia text

{context_knowledge.strip()}
""".strip()

    # Step 4: Call GPT-4o
    response = get_client().chat.completions.parse(
        model=model_name,
        temperature=0,
        response_format=make_annotation_factuality_model(
            len(annotation.token_chunks) - 1
        ),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        seed=seed,
    )

    # Price updating
    with GPT_LOCK:
        usage = response.usage
        COSTS["generation_input_tokens"] += usage.prompt_tokens
        COSTS["generation_output_tokens"] += usage.completion_tokens
        COSTS["generation_cost"] += (
            usage.prompt_tokens * PRICES[model_name]["input"]
            + usage.completion_tokens * PRICES[model_name]["output"]
        )

    # Step 5: Parse response into Pydantic
    result_pydantic = response.choices[0].message.parsed

    # Step 6: convert into integer list
    return SegmentationFactualityList(
        factuality_scores=[
            value
            for _, value in sorted(
                result_pydantic.model_dump().items(),
                key=lambda kv: int(kv[0][len("score_chunk_") :]),
            )
        ]
    )


def aggregate_factuality_scores(
    all_scores: t.List[SegmentationFactualityList],
) -> SegmentationFactualityList:
    """
    Aggregates multiple SegmentationFactualityList objects into a single consensus object.

    For each chunk index:
      - If a majority label exists (i.e., occurs more than half the time), that label is selected.
      - If there is no majority, the min label value among the scores is used.

    Args:
        all_scores (List[SegmentationFactualityList]):
            A list of SegmentationFactualityList objects to aggregate. All must have the same number of chunks.

    Returns:
        SegmentationFactualityList:
            A new SegmentationFactualityList object containing the aggregated factuality scores.

    Raises:
        ValueError: If no input is provided, or if the number of chunks differs between inputs.
    """
    if not all_scores:
        raise ValueError("No SegmentationFactualityList objects provided.")

    num_chunks = len(all_scores[0].factuality_scores)

    # Sanity check: ensure all annotations have same number of chunks
    for s in all_scores:
        if len(s.factuality_scores) != num_chunks:
            raise ValueError(
                "All SegmentationFactualityList instances must have the same number of chunks."
            )

    aggregated_scores = []
    for i in range(num_chunks):
        labels_at_i = [s.factuality_scores[i] for s in all_scores]
        counter = Counter(labels_at_i)
        most_common = counter.most_common()

        if len(most_common) == 1 or most_common[0][1] > len(labels_at_i) // 2:
            # Majority found
            aggregated_scores.append(most_common[0][0])
        else:
            # No majority → fallback to max
            aggregated_scores.append(min(labels_at_i))

    return SegmentationFactualityList(factuality_scores=aggregated_scores)


def get_robust_gpt_evaluation(
    annotation: Annotation,
    model_name: str,
    num_redundancy: int = GPT_REDUNDANCY,
) -> SegmentationFactualityList:
    """
    Runs redundant GPT evaluations to robustly estimate factuality scores for a segmented output.

    Performs multiple evaluations in parallel, filters out inconsistent responses, and aggregates
    the valid results using a majority voting scheme (with fallback to min).

    Args:
        annotation (Annotation): The segmented output to evaluate.
        model_name (str): The reference of the GPT model to use
        num_redundancy (int): Number of parallel redundant GPT evaluations to perform.

    Returns:
        SegmentationFactualityList: Aggregated factuality scores for each chunk.
    """
    # Run multiple evaluations in parallel
    all_factuality_scores: t.List[SegmentationFactualityList] = [
        get_one_gpt_evaluation(annotation, model_name=model_name, seed=seed)
        for seed in range(num_redundancy)
    ]

    # Filter out evaluations with incorrect number of chunks
    expected_len = len(annotation.token_chunks) - 1  # Ignoring trailing EOS chunk
    all_factuality_scores = [
        score
        for score in all_factuality_scores
        if len(score.factuality_scores) == expected_len
    ]

    # Retry if none are valid
    if not all_factuality_scores:
        logger.info(
            f"No factuality_scores with correct length, retrying: {annotation.generation_id}"
        )
        return get_robust_gpt_evaluation(annotation, num_redundancy)

    # Aggregate and add score for <|eot_id|> token
    result = aggregate_factuality_scores(all_factuality_scores)
    result.factuality_scores.append(1)  # Assume end-of-text token is factual
    return result


def add_gpt_labels(
    annotation: Annotation, model: str | None, force_recompute: bool = False
) -> None:
    """Adds GPT labels to annotation (expected method: stopwords)."""

    # Checking model
    if model is not None:
        if (not force_recompute) and (f"gpt-{model}" in annotation.labels):
            return

        model_name = CHAT_MODELS[model]
    else:
        for model in CHAT_MODELS:
            add_gpt_labels(
                annotation=annotation, model=model, force_recompute=force_recompute
            )

        return

    # Sanity check - if there exist a token for which the target token is not in top-24, we
    # assign gpt-4.1 label 1 and gpt-4o label -1 for every token, to ensure that this sample will be removed
    # from the final MUCH dataset. NB: this only happens for 4 samples in the original 6.4k
    # generations obtained in MUCH before filtering.
    # Similarly, we assigne gpt-4.1 label 1 and gpt-4o label -1 when the generation was cut before EOS
    # token, because the model was stuck in a loop generation. This only happens for 3 samples in the original
    # 6.4k generations obtained in MUCH before filtering.
    for item in annotation.generation.topn_prob_dicts:
        tokenizer = get_tokenizer(annotation.generation.generation_cfg.model_name)
        if len(item) > 24:
            logger.info(
                f"Filtering out annotation {annotation.generation_id} for token logit reasons."
            )
            need_filtering = True
        elif annotation.generation.output_tokens[-1] not in get_known_eos_or_eot(
            tokenizer
        ):
            logger.info(
                f"Filtering out annotation {annotation.generation_id} for EOS not found reason."
            )
            need_filtering = True
        else:
            need_filtering = False

        # Doing the actual filtering
        if need_filtering:
            if model == "4.1":
                annotation.labels[f"gpt-{model}"] = [1 for _ in annotation.token_chunks]
            elif model == "4o":
                annotation.labels[f"gpt-{model}"] = [
                    -1 for _ in annotation.token_chunks
                ]
            annotation.write_disk()

            return

    # Getting robust label
    factuality_scores = get_robust_gpt_evaluation(annotation, model_name=model_name)

    # Writing to disk
    annotation.labels[f"gpt-{model}"] = [
        score for score in factuality_scores.factuality_scores
    ]
    annotation.write_disk()


def log_costs():
    """Logs the costs of the API calls."""
    logger.info(f"Embedding tokens:{COSTS['embedding_tokens']}")
    logger.info(f"Embedding cost: ${COSTS['embedding_cost']}")
    logger.info(f"Generation input tokens:{COSTS['generation_input_tokens']}")
    logger.info(f"Generation output tokens:{COSTS['generation_output_tokens']}")
    logger.info(f"Generation cost: ${COSTS['generation_cost']}")
    logger.info(f"TOTAL time: {time() - GPT_T0:.4f}")
    logger.info(f"TOTAL COST: ${COSTS['embedding_cost'] + COSTS['generation_cost']}")
