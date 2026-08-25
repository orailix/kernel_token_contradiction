import typing as t
from functools import cached_property

import numpy as np
from pydantic import BaseModel

from ..utils import paths, stopwords
from ..generation import get_tokenizer, get_generation_from_idx


class UQSignalValue(BaseModel):

    generation_id: str
    signal_name: str
    values: list

    @cached_property
    def output_path(self):
        (paths.output / "signals").mkdir(exist_ok=True, parents=True)
        return paths.output / "signals" / f"{self.signal_name}.jsonl"

    def renorm(
        self, new_name: t.Optional[str] = None, renorm_factor: t.Optional[float] = None
    ) -> t.Self:
        """Returns a new instance of UQSignalValue all values renormed.
        By default, the renormalization factor is the max of the current signal."""

        if renorm_factor is None:
            values_renormed: np.ndarray = self.values_np / np.max(self.values_np)
        else:
            values_renormed: np.ndarray = self.values_np / renorm_factor

        return UQSignalValue(
            generation_id=self.generation_id,
            signal_name=f"{self.signal_name}" if new_name is None else new_name,
            values=values_renormed.tolist(),
        )

    def softmax(
        self, new_name: t.Optional[str] = None, temparature: float = 1
    ) -> t.Self:
        """Returns a new instance of UQSignalValue with softmaxed values."""

        values_softmaxed: np.ndarray = np.exp(self.values_np / temparature)
        values_softmaxed /= np.sum(values_softmaxed)

        return UQSignalValue(
            generation_id=self.generation_id,
            signal_name=f"{self.signal_name}" if new_name is None else new_name,
            values=values_softmaxed.tolist(),
        )

    def revert(
        self,
        new_name: t.Optional[str] = None,
    ) -> t.Self:
        """Changes the values to 1 - values."""

        values_reverted: np.ndarray = 1 - self.values_np

        return UQSignalValue(
            generation_id=self.generation_id,
            signal_name=f"{self.signal_name}" if new_name is None else new_name,
            values=values_reverted.tolist(),
        )
    
    def nullify_stopwords(
        self,
        new_name: t.Optional[str] = None,
    ) -> t.Self:
        """Nullifies the signal value for stopwords."""

        values_filtered = self.values_np.copy()
        generation = get_generation_from_idx(self.generation_id)
        tokenizer = get_tokenizer(generation.generation_cfg.model_name)
        tokenized_output = [
            tokenizer.decode(item).lower().strip()
            for item in generation.output_tokens
        ]
        for idx, token_str in enumerate(tokenized_output):
            if token_str in stopwords.get_extended_stopwords_set():
                values_filtered[idx] = 0

        return UQSignalValue(
            generation_id=self.generation_id,
            signal_name=f"{self.signal_name}" if new_name is None else new_name,
            values=values_filtered.tolist(),
        )

    @property
    def values_np(self):
        """Values of the signal, of shape (n_tokens, dim)"""
        return np.array(self.values)

    def write_disk_no_check(self):
        """Writes on the disk without checking if it already exists."""
        is_nonempty = self.output_path.exists() and self.output_path.stat().st_size > 0

        with self.output_path.open("a") as f:
            if is_nonempty:
                f.write("\n")
            f.write(self.model_dump_json())

    def write_disk(self):
        """Writes on the disk and checks if it exists."""
        # Write to disk
        self.write_disk_no_check()

        # Remove olders
        lines = self.output_path.read_text().split("\n")
        lines_filtered = [
            l
            for idx, l in enumerate(lines)
            if (
                l[12 : 14 + len(self.generation_id)] != f'"{self.generation_id}"'
                or idx == len(lines) - 1
            )
        ]
        self.output_path.write_text("\n".join(lines_filtered))
