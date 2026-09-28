from datasets import load_dataset
from loguru import logger

from ..utils.constants import AUTHORIZED_LANG
from ..utils.typer_app import app
from .utils import Question


@app.command()
def get_mushroom_questions():
    """Load the mu-shroom dataset and store unique authorized-language questions to disk."""

    logger.info("Loading mu-shroom dataset")
    ds = load_dataset("Helsinki-NLP/mu-shroom", "all")

    total_written = 0
    total_rejected = 0

    logger.info("Processing validation and test splits")
    for key in ["validation", "test"]:
        for item in ds[key]:
            lang = item["lang"].lower()

            if lang not in AUTHORIZED_LANG:
                total_rejected += 1
                continue

            Question(
                lang=lang, prompt=item["model_input"], wiki_url=item["wikipedia_url"]
            ).write_disk()
            total_written += 1

    logger.info(f"Finished writing {total_written} accepted questions to disk.")
    logger.info(f"Rejected {total_rejected} questions due to unsupported language.")
