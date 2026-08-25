"""
Abstract Base Class for Uncertainty Quantification Signals.

This module defines the UQSignal abstract base class that all uncertainty quantification
signals must inherit from. It provides core functionality for computing signals on
generations, managing signal values, and interfacing with the evaluation framework.

Key methods to implement:
    - signal_name: Property returning the unique name for this signal
    - compute_signal_value: Method computing token-level signal values for a generation

The class also provides utilities for:
    - Computing signals on single or multiple generations
    - Caching and retrieving computed signal values
    - Searching for signal values by generation ID
"""

import json
import os
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..annotation import Annotation, get_all_annotations
from ..generation import Generation, get_tokenizer
from ..utils import paths
from ..utils.constants import CPU_ONLY, KTC_DEBUG, LLMS_INFOS
from .signal_value import UQSignalValue


class UQSignal(ABC):
    @classmethod
    def from_env(cls) -> t.Self:
        return cls()

    @property
    @abstractmethod
    def signal_name(self) -> str:
        ...

    def warm_up(self) -> None:
        logger.info(f"Warming up signal: {self.signal_name}")

        for model_name in LLMS_INFOS:
            get_tokenizer(model_name)

    def __init__(self):
        self._all_signal_values_cache = []

    @property
    def output_path(self) -> Path:
        (paths.output / "signals").mkdir(exist_ok=True, parents=True)
        result = paths.output / "signals" / f"{self.signal_name}.jsonl"
        if not result.is_file():
            result.write_text("")

        return paths.output / "signals" / f"{self.signal_name}.jsonl"

    @abstractmethod
    def compute_signal_value(self, generation: Generation) -> list:
        ...

    def compute_on_generation(
        self, generation: Generation, force_recompute: bool = False
    ) -> None:

        # Need recompute?
        if not force_recompute:
            if generation.id in self.get_all_computed_ids():
                return

        # Creating the SignalValue
        signal_value = UQSignalValue(
            generation_id=generation.id,
            signal_name=self.signal_name,
            values=self.compute_signal_value(generation),
        )

        # Saving
        signal_value.write_disk()

    def compute_on_multiple_generations(
        self, all_generations: t.List[Generation], force_recompute: bool = False
    ) -> None:

        # Filtering generations to recompute
        existing_generations = (
            set() if force_recompute else set(self.get_all_computed_ids())
        )
        generations_to_compute: t.List[Generation] = []
        generations_to_compute_ids = set()
        for generation in all_generations:
            if (generation.id not in existing_generations) and (
                generation.id not in generations_to_compute_ids
            ):
                generations_to_compute.append(generation)
                generations_to_compute_ids.add(generation.id)

        # Logging
        logger.debug(
            f"Computing signal {self.signal_name} on {len(generations_to_compute)} generations..."
        )

        # Creating the SignalValue
        for generation in tqdm(generations_to_compute):
            signal_value = UQSignalValue(
                generation_id=generation.id,
                signal_name=self.signal_name,
                values=self.compute_signal_value(generation),
            )

            # Saving
            signal_value.write_disk()

    def compute_on_all_generations(self, force_recompute: bool = False) -> None:

        # All generations
        rg = np.random.RandomState(seed=42)
        all_annotations = get_all_annotations()
        all_annotations = {
            key: all_annotations[key] for key in rg.permutation(list(all_annotations))
        }

        # Filtering
        annotations_filtered = {
            annot_id: annot
            for annot_id, annot in all_annotations.items()
            if (
                "gpt-4o" in annot.labels
                and "gpt-4.1" in annot.labels
                and annot.labels["gpt-4o"] == annot.labels["gpt-4.1"]
            )
            or ("mushroom" in annot.labels)
        }

        # Filtering - DEBUG mode
        if KTC_DEBUG:
            if CPU_ONLY and ("ccp_" in self.signal_name or "sar_" in self.signal_name):
                num_samples = 49
            else:
                num_samples = 488

            logger.warning(
                f"DEBUG MODE: Keeping only MUCH samples, sorting and clipping to {num_samples} samples. You should NOT use that option for regular signal compute."
            )
            annotations_filtered = {
                annot_id: annot
                for annot_id, annot in annotations_filtered.items()
                if "gpt-4o" in annot.labels
            }
            # Getting the generations
            generation_filtered = [
                annotations_filtered[annot_id].generation
                for annot_id in sorted(annotations_filtered)
            ][:num_samples]

        else:
            # Getting the generations
            generation_filtered = [
                annot.generation for annot in annotations_filtered.values()
            ]

        if os.getenv("COUNT_NN_FLOP", "0") == "1":
            logger.warning(
                "Clipping to 100 samples because of FLOP tracking - you should NOT use that option for regular signal compute as it slows down model calls"
            )
            generation_filtered = generation_filtered[:100]

        # Computing
        self.compute_on_multiple_generations(
            generation_filtered, force_recompute=force_recompute
        )

    def get_all_signal_values(self) -> t.List[UQSignalValue]:
        self._all_signal_values_cache = []
        with self.output_path.open("r") as f:
            for line in f:
                if line.strip():
                    self._all_signal_values_cache.append(
                        UQSignalValue(
                            **json.loads(line),
                        )
                    )

        return self._all_signal_values_cache

    def search_generation(
        self,
        generation: Generation | Annotation,
        all_signals: t.Optional[t.List[UQSignalValue]] = None,
        force_reload: bool = False,
    ) -> UQSignalValue:

        # Need convert?
        if isinstance(generation, Annotation):
            generation = generation.generation

        # Force reload?
        if force_reload or (
            all_signals is None and len(self._all_signal_values_cache) == 0
        ):
            all_signals = self.get_all_signal_values()
        elif all_signals is None:
            # In that case, we know len(self._all_signal_values_cache) > 0
            all_signals = self._all_signal_values_cache

        # Inspecting all signals
        for signal_value in all_signals:
            if signal_value.generation_id == generation.id:
                return signal_value

        # Trying with force reload
        if not force_reload:
            return self.search_generation(generation=generation, force_reload=True)

        raise ValueError(
            f"Did not found any UQSignalValue for this ID: {generation.id}"
        )

    def get_all_computed_ids(self) -> t.List[str]:
        """Returns the list of Anotation IDs that have been computed for that signal."""
        result = []
        for signal_value in self.get_all_signal_values():
            result.append(signal_value.generation_id)

        return result

    def get_mean(self) -> np.ndarray:
        """Returns the mean of the signal over all computed generations."""
        all_signal_values = self.get_all_signal_values()

        all_values = np.zeros((0,))
        for signal_value in all_signal_values:
            all_values = np.concatenate([all_values, signal_value.values_np], axis=0)

        return np.mean(all_values, axis=0)
