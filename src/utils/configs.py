import base64
import hashlib
import json
import os
import typing as t
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from decouple import config
from filelock import FileLock

from ..utils import paths


@dataclass
class BaseConfig(ABC):
    @classmethod
    @abstractmethod
    def key_ignored_in_hash(cls) -> t.List[str]:
        ...

    @classmethod
    @abstractmethod
    def class_output_name(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def ignore_in_hash_if_equal_to_default(cls) -> bool:
        ...

    # Output
    _base_output_dir: Path = Path(".")

    @property
    def base_output_dir(self) -> Path:
        if self._base_output_dir == Path("."):
            return paths.root / "output" / self.__class__.class_output_name()

        return self._base_output_dir / self.__class__.class_output_name()

    # Post init
    def __post_init__(self):
        _ = self.get_output_dir()

    @property
    def id(self) -> str:
        to_hash = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value != field.default or (
                not self.ignore_in_hash_if_equal_to_default()
            ):
                to_hash[field.name] = value

        # Hashing
        hashable = ""
        for key in sorted(to_hash):
            if key in self.__class__.key_ignored_in_hash():
                continue
            hashable += f"{key}={to_hash[key]}\n"

        # Persistent, replicable and URL-free hash
        return base64.urlsafe_b64encode(
            hashlib.md5(hashable.encode("utf-8")).digest()
        ).decode()[:22]

    def __repr__(self) -> str:
        result = f"{self.__class__.__name__}[ {self.id} ](\n"
        names = [f.name for f in fields(self)]
        for key in sorted(names):
            result += f"\t{key}: {getattr(self, key)}\n"
        result += ")"

        return result

    def get_as_dict(self) -> dict:
        self_as_dict = asdict(self)
        self_as_dict["_base_output_dir"] = str(self_as_dict["_base_output_dir"])

        return self_as_dict

    def get_output_dir(self) -> Path:
        result = self.base_output_dir / self.id
        result.mkdir(parents=True, exist_ok=True)

        # Dumping config
        self_as_dict = self.get_as_dict()
        with FileLock(result / "config.lock"):
            (result / "config.json").write_text(json.dumps(self_as_dict, indent=4))

        # Output
        return result

    def get_config_path(self) -> Path:
        return self.get_output_dir() / "config.json"

    @classmethod
    def get_all_configs(cls) -> t.List[t.Self]:
        result = []

        for dir_to_search in cls.get_dir_to_search_list():
            for child in dir_to_search.iterdir():
                result.append(cls.from_json(child / "config.json"))

        return result

    @classmethod
    def get_dir_to_search_list(cls) -> t.List[Path]:
        if "_BASE_OUTPUT_DIR" in os.environ:
            return [
                Path(os.environ["_BASE_OUTPUT_DIR"]) / cls.class_output_name(),
                paths.root / "output" / cls.class_output_name(),
            ]
        else:
            return [
                paths.root / "output" / cls.class_output_name(),
            ]

    @classmethod
    def from_env(cls) -> t.Self:
        attributes = {
            fld.name: config(fld.name.upper(), default=fld.default, cast=fld.type)
            for fld in fields(cls)
        }

        return cls(**attributes)

    @classmethod
    def from_json(cls, path: Path, reset_output_dir: bool = False) -> t.Self:

        # Converting
        path = Path(path)

        # Loading json - parsing special attributes
        with FileLock(path.with_suffix(".lock")):
            json_content = json.loads(path.read_text())

        if reset_output_dir:
            json_content["_base_output_dir"] = Path(".")
        else:
            json_content["_base_output_dir"] = Path(json_content["_base_output_dir"])

        # Output
        return cls(**json_content)

    @classmethod
    def autoconfig(cls, name: str, reset_output_dir: bool = False) -> t.Self:

        # Attempt 1: path?
        if Path(name).suffix == ".json" and Path(name).is_file():
            return cls.from_json(Path(name), reset_output_dir)

        # Attempt 2: known config?
        # We start with $_BASE_OUTPUT_DIR
        for dir_to_search in cls.get_dir_to_search_list():

            all_ids = []
            if dir_to_search.is_dir():
                for child in dir_to_search.iterdir():
                    if child.is_dir():
                        all_ids.append(child.name)

            # Found config?
            candidates = []
            for existing_id in all_ids:
                if len(name) >= 4 and existing_id[: len(name)] == name:
                    candidates.append(dir_to_search / existing_id / "config.json")

            # Return
            if len(candidates) == 1:
                return cls.from_json(candidates[0], reset_output_dir)

        raise ValueError(f"Autoconfig did not succeed: {name}")
