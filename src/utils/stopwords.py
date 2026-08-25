import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords

@lru_cache()
def get_nltk_stopwords_set():

    # Collect all stopwords NLTK knows
    all_langs = ["english", "spanish", "french", "german"]
    stopword_set = set()
    for lang in all_langs:
        try:
            stopword_set |= set(stopwords.words(lang))
        except (OSError, LookupError):
            nltk.download("stopwords")
            stopword_set |= set(stopwords.words(lang))

    # Extra stopwords - when needed

    return stopword_set


@lru_cache()
def get_punctuation_set():
    return set(string.punctuation) | {
        "،",
        "؛",
        "؟",
        "，",
        "。",
        "、",
        "？",
        "！",
        "：",
        "；",
        "（",
        "）",
        "《",
        "》",
        "“",
        "”",
        "¿",
        "¡",
        "«",
        "»",
        "…",
        "''",
        "``",
        "\n",
        "",
        "''",
        '""',
    }


@lru_cache()
def get_extended_stopwords_set():
    punctuation = get_punctuation_set()
    stopword_set = get_nltk_stopwords_set()
    extra_tokens = {"'s", "'re", "'ve", "'d", "'ll", "n't", "''", "``", " ", ""}

    return stopword_set | punctuation | extra_tokens