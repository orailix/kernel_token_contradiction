import base64
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from ..utils import paths

QUESTION_ROOT = paths.output / "questions"


class Question(BaseModel):
    """Represents a question with language, prompt text, and source URL."""

    lang: str
    prompt: str
    wiki_url: str

    @property
    def id(self) -> str:
        """Unique, stable hash identifier for the question."""
        return base64.urlsafe_b64encode(
            hashlib.md5(self.model_dump_json().encode("utf-8")).digest()
        ).decode()[:22]

    @property
    def output_file(self) -> Path:
        """Path to the JSONL file storing questions for this language."""
        QUESTION_ROOT.mkdir(exist_ok=True, parents=True)
        return QUESTION_ROOT / f"{self.lang}.jsonl"

    def __repr__(self) -> str:
        """Readable string representation of the question."""
        result = f"{self.__class__.__name__}[ {self.id} ](\n"
        for key in sorted(dict(self)):
            result += f"\t{key}: {getattr(self, key)}\n"
        result += ")"
        return result

    def write_disk(self) -> str:
        """Append the question to its corresponding JSONL file on disk."""

        # Skip duplicate
        existing_ids = set()
        if self.output_file.exists():
            with self.output_file.open("r") as f:
                for line in f:
                    if line.strip():
                        as_dict = json.loads(line)
                        existing_ids.add(as_dict["id"])

        if self.id in existing_ids:
            return

        is_nonempty = self.output_file.exists() and self.output_file.stat().st_size > 0
        with self.output_file.open("a") as f:
            if is_nonempty:
                f.write("\n")
            self_as_dict = dict(id=self.id, **dict(self))
            f.write(json.dumps(self_as_dict))


def get_all_questions() -> list[Question]:
    """Load all questions stored under QUESTION_ROOT from JSONL files."""
    result = []
    QUESTION_ROOT.mkdir(exist_ok=True, parents=True)
    for child in QUESTION_ROOT.iterdir():
        if child.is_file() and child.suffix == ".jsonl":
            with child.open("r") as f:
                for line in f:
                    if line.strip():
                        as_dict = json.loads(line)
                        result.append(
                            Question(
                                lang=as_dict["lang"],
                                prompt=as_dict["prompt"],
                                wiki_url=as_dict["wiki_url"],
                            )
                        )
    return result
