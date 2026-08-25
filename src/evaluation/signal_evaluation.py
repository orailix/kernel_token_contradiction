import json
import typing as t
from functools import cached_property

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from ..utils import paths


MANUAL_ADDON = {1: "_manual", 0: "_all", -1: "_auto"}

class SignalEvaluation:
    """
    Container for the evaluation of a Signal.

    Each entry is an array of shape (n_snippets, ).
    For example:
        self.mean_in[k] contains the mean of the signal for the k-th hallucination_in snippet.
    """

    _FIELDS = [
        "mean_in",
        "max_in",
        "geometric_in",
        "product_in",
        "mean_out",
        "max_out",
        "geometric_out",
        "product_out",
    ]

    def __init__(
        self,
        signal_name: str,
        lang: t.Optional[str] = None,
        manual: int = 0,
        include_much: bool = True,
        include_mushroom: bool = False,
        mean_in: t.Optional[np.ndarray] = None,
        max_in: t.Optional[np.ndarray] = None,
        geometric_in: t.Optional[np.ndarray] = None,
        product_in: t.Optional[np.ndarray] = None,
        mean_out: t.Optional[np.ndarray] = None,
        max_out: t.Optional[np.ndarray] = None,
        geometric_out: t.Optional[np.ndarray] = None,
        product_out: t.Optional[np.ndarray] = None,
    ) -> None:

        # Check definition status explicitly against None
        arrays = [
            mean_in,
            max_in,
            geometric_in,
            product_in,
            mean_out,
            max_out,
            geometric_out,
            product_out,
        ]
        defined = [a is not None for a in arrays]

        if any(defined) and not all(defined):
            raise ValueError("Either all arrays must be None or all must be defined.")
        elif not all(defined):
            arrays = [np.empty((0,)) for _ in self._FIELDS]

        # assign
        self.lang = lang
        self.manual = manual
        self.include_much = include_much
        self.include_mushroom = include_mushroom
        self.signal_name = signal_name
        for field, arr in zip(self._FIELDS, arrays):
            setattr(self, field, arr)

    @classmethod
    def from_dict(cls, d: dict) -> t.Self:
        return cls(
            signal_name=d["signal_name"],
            lang=None if "lang" not in d else d["lang"],
            manual=d["manual"],
            include_much=d["include_much"],
            include_mushroom=d["include_mushroom"],
            **{f: np.array(d[f]) for f in cls._FIELDS},
        )

    def as_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "lang": self.lang,
            "manual": self.manual,
            "include_much": self.include_much,
            "include_mushroom": self.include_mushroom,
            **{f: getattr(self, f).tolist() for f in self._FIELDS},
        }

    @cached_property
    def output_path(self):
        (paths.output / "evaluations").mkdir(exist_ok=True, parents=True)
        lang_addon = "" if self.lang is None else f"_{self.lang}"
        manual_addon = MANUAL_ADDON[self.manual]
        much_addon = "" if not self.include_much else "_much"
        mushroom_addon = "" if not self.include_mushroom else "_mushroom"
        return (
            paths.output
            / "evaluations"
            / f"{self.signal_name}{lang_addon}{much_addon}{manual_addon}{mushroom_addon}.json"
        )

    def write_disk(self):
        """Writes on the disk."""
        with self.output_path.open("w") as f:
            f.write(json.dumps(self.as_dict()))

    def _update_values(self, signal_chunk: np.ndarray, kind: str) -> None:
        """Generic update for 'in' or 'out' signals."""
        if kind not in ("in", "out"):
            raise ValueError("kind must be 'in' or 'out'.")

        # statistics
        try:
            mean_val = np.mean(signal_chunk)
        except Exception as e:
            raise ValueError(
                f"Incorrect signal chunk: <{type(signal_chunk)}>={signal_chunk}"
            )
        max_val = np.max(signal_chunk)
        geometric_val = 1 - (np.prod(1 - signal_chunk) + 1e-5) ** (
            1 / signal_chunk.shape[0]
        )
        product_val = 1 - np.prod(1 - signal_chunk)

        # append to the right attributes
        setattr(
            self,
            f"mean_{kind}",
            np.concatenate([getattr(self, f"mean_{kind}"), [mean_val]]),
        )
        setattr(
            self,
            f"max_{kind}",
            np.concatenate([getattr(self, f"max_{kind}"), [max_val]]),
        )
        setattr(
            self,
            f"geometric_{kind}",
            np.concatenate([getattr(self, f"geometric_{kind}"), [geometric_val]]),
        )
        setattr(
            self,
            f"product_{kind}",
            np.concatenate([getattr(self, f"product_{kind}"), [product_val]]),
        )

    def update_values_in(self, signal_chunk: np.ndarray) -> None:
        """Update values from a hallucination_in snippet."""
        self._update_values(signal_chunk, "in")

    def update_values_out(self, signal_chunk: np.ndarray) -> None:
        """Update values from a hallucination_out snippet."""
        self._update_values(signal_chunk, "out")

    def get_rocauc_prauc(self, aggregator: str) -> tuple:
        """
        Returns a tuple: (fpr, tpr, roc_auc, prc, rec, pr_auc, pr_ap).
        """
        if aggregator not in ("mean", "max", "product", "geometric"):
            raise ValueError(f"Unknown aggregator: {aggregator}")

        in_container = getattr(self, f"{aggregator}_in")
        out_container = getattr(self, f"{aggregator}_out")

        predictor = np.concatenate([in_container, out_container])
        labels = np.concatenate(
            [
                np.ones(in_container.shape[0], dtype=int),
                np.zeros(out_container.shape[0], dtype=int),
            ]
        )

        fpr, tpr, _ = roc_curve(labels, predictor)
        prc, rec, _ = precision_recall_curve(labels, predictor)
        return (
            fpr,
            tpr,
            auc(fpr, tpr),  # ROC
            prc,
            rec,
            auc(rec, prc),  # PR
            average_precision_score(labels, predictor),
        )
