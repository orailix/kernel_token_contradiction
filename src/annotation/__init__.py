"""
Annotation module - Claim annotation and labeling utilities.

This module provides functionality for annotating LLM-generated claims with
factuality labels. It supports both GPT-based automatic annotation and manual
human annotation.

Key components:
    - Annotation: Data class for claim annotations with labels
    - GPT-based annotation: Automatic labeling using GPT-4 models
    - Human annotation: Interface for manual labeling
    - Wikipedia references: Download and cache Wikipedia pages for context

Key functions:
    - add_all_gpt_labels: Annotate all claims using GPT-4o and GPT-4.1
    - download_all_wiki_references: Cache Wikipedia pages for all questions
    - segment_all_generations: Segment LLM outputs into individual claims
    - get_annotation_from_idx: Retrieve an annotation by its ID
    - get_all_annotations: Retrieve all annotations
"""

from .cli import (
    add_all_gpt_labels,
    download_all_wiki_references,
    segment_all_generations,
)
from .utils import Annotation, get_all_annotations, get_annotation_from_idx

__all__ = [
    "add_all_gpt_labels",
    "download_all_wiki_references",
    "segment_all_generations",
    "Annotation",
    "get_all_annotations",
    "get_annotation_from_idx",
]
