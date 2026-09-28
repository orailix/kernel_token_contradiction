import textwrap

from ..generation import get_tokenizer
from .gpt_evaluator import (
    RAG_TOP_K,
    chunk_text,
    get_wiki_content_from_cache,
    rank_chunks_by_similarity,
)
from .utils import Annotation


def add_human_labels(annotation: Annotation, annotator: str) -> None:

    # Tokenizer
    tokenizer = get_tokenizer(annotation.generation.generation_cfg.model_name)

    # Select relevant knowledge
    main_content, infobox = get_wiki_content_from_cache(annotation.generation.wiki_url)
    chunks = chunk_text(main_content)
    relevant_chunks = rank_chunks_by_similarity(
        chunks, annotation.generation.output, top_k=RAG_TOP_K
    )
    relevant_chunks = ["\n".join(textwrap.wrap(item)) for item in relevant_chunks]
    context_knowledge = "\n\n".join(relevant_chunks)

    print(
        f"""
# Question <Annotation[{annotation.generation_id}]>
{annotation.generation.prompt}

# Wiki url
{annotation.generation.wiki_url}

# Model Output
{"\n".join(textwrap.wrap(annotation.generation.output.strip()))}

# Segmented Output
{annotation.get_repr_string(with_labels=False, remove_latest=True)}

# Reference Knowledge

## Wikipedia Infobox

{infobox}

## Relevant chunks from Wikipedia text
{context_knowledge.strip()}""".strip()
        + "\n\n"
    )

    all_labels = []
    for idx_chunk, chunk in enumerate(annotation.token_chunks):
        # Skipping the last one
        if idx_chunk == len(annotation.token_chunks) - 1:
            all_labels.append(1)
            continue
        chunk_str = tokenizer.decode(
            annotation.generation.output_tokens[chunk[0] : chunk[-1] + 1]
        )
        all_labels.append(int(input(f"> {chunk_str.replace('\n', ' ')}    | Label=")))

    save = input("To cancel this annotation, type 'c'. Else, hit Enter.")
    if save == "c":
        print("Re-annotating this annotation...")
        add_human_labels(annotation, annotator)
    else:
        print("Saving...")
        annotation.labels[annotator] = all_labels
        annotation.write_disk()
