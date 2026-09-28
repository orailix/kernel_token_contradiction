"""
Questions module - Question generation and management.

This module provides functionality for downloading, processing, and managing
questions from the MU-SHroom dataset. Questions are the input prompts used
for LLM generation in the MUCH benchmark.

Key components:
    - Question: Data class for question metadata and content
    - get_mushroom_questions: CLI command to download questions
    - get_all_questions: Retrieve all questions from the dataset
"""

from .cli import get_mushroom_questions
from .utils import Question, get_all_questions

__all__ = [
    "get_mushroom_questions",
    "Question",
    "get_all_questions",
]
