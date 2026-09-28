import json
import os
import re
import typing as t
import urllib.parse
from collections import defaultdict

import wikipedia
import wptools
from diskcache import Cache
from joblib import Parallel, delayed
from loguru import logger

from ..questions import Question, get_all_questions
from ..utils import paths

WIKI_CACHE = paths.output / "questions" / "wiki_cache"
INFOBOX_PREFIX = "## Infobox<special token uvDivFMeJDp>\n\n"


def normalize_wikipedia_title(title: str) -> str:
    """
    Normalize a Wikipedia title so it can be safely used with the MediaWiki API.

    - Decodes any percent-encoding (so %27 → ')
    - Leaves the raw characters (MediaWiki API expects them)
    - Replaces spaces with underscores
    """
    # First decode any %XX escapes in case the input was already encoded
    decoded = urllib.parse.unquote(title)
    # Wikipedia convention: spaces → underscores
    normalized = decoded.replace(" ", "_")
    return normalized


def get_question_per_language() -> t.Dict[str, t.List[Question]]:
    """Gets all questions, sorted per language."""

    result = defaultdict(list)
    for question in get_all_questions():
        result[question.lang].append(question)

    return result


def download_wiki_language(question_list: t.List[Question], lang: str):
    """Downloads all content from wiki for a given language and set of questions."""
    logger.info(
        f"Downloading the {len(question_list)} pages of questions with lang={lang}..."
    )

    # Setting the language
    wikipedia.set_lang(lang)

    # Converting in URLS
    wiki_url_list = [question.wiki_url for question in question_list]

    wiki_content_list = Parallel(n_jobs=os.cpu_count(), backend="threading")(
        delayed(get_wiki_content)(wiki_url) for wiki_url in wiki_url_list
    )

    # Writing result
    with Cache(WIKI_CACHE) as reference:
        for wiki_url, wiki_content in zip(wiki_url_list, wiki_content_list):
            reference.set(wiki_url, wiki_content)


def get_wiki_content_from_cache(wiki_url) -> tuple[str, str]:
    """Gets the wiki content of an URL from the cache."""

    with Cache(WIKI_CACHE) as reference:
        if not (wiki_url in reference):
            raise KeyError(
                f"Wiki page {wiki_url} not in cache. Maybe you forgot using `download-wiki-language` command?"
            )

        result = reference.get(wiki_url)

    main_content, infobox = result.split(INFOBOX_PREFIX)

    return main_content, infobox


# Inspired from: https://github.com/erictherobot/wikipedia-markdown-generator
def get_wiki_content(wiki_url: str) -> str:
    """Gets some wiki content to bring factual knowledge to an annotation"""

    # "wikipedia" librairy part (for a convenient markdown)

    # Get topic
    topic_non_normalized = "/".join(
        wiki_url.split("/")[4:]
    )  # To support urls with "/" in them
    topic_normalized = normalize_wikipedia_title(topic_non_normalized)

    page = wikipedia.page(topic_normalized, auto_suggest=False)
    markdown_text = f"# {topic_normalized}\n\n"

    page_content = re.sub(r"=== ([^=]+) ===", r"### \1", page.content)
    page_content = re.sub(r"== ([^=]+) ==", r"## \1", page_content)

    sections = re.split(r"\n(## .*)\n", page_content)
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections) and any(
            line.strip() for line in sections[i + 1].split("\n")
        ):
            markdown_text += f"{sections[i]}\n{sections[i+1]}\n\n"

    # "wptools" librairy part
    lang = wiki_url.split("/")[2].split(".")[0]
    page = wptools.page(topic_normalized, lang=lang)
    page_parsed = page.get_parse(show=False)
    if "infobox" in page.data:
        infobox = page_parsed.data["infobox"]
        dumped_result = json.dumps(infobox, indent=4, ensure_ascii=False)
    else:
        dumped_result = r"{}"

    markdown_text += f"{INFOBOX_PREFIX}{dumped_result}"

    return markdown_text
