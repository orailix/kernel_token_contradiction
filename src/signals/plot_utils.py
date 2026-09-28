import textwrap
import typing as t

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import entropy

from ..annotation import Annotation
from ..generation import get_tokenizer
from ..utils.constants import N_LOGITS_SAVED
from .signal_value import UQSignalValue

PLOT_COLORS = {
    1: "green",
    -1: "red",
}

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_on_ax(
    annotation: Annotation,
    ax: Axes,
    signal_values_list: t.List[UQSignalValue],
    pretty_names: t.Optional[t.List[str]] = None,
    title_char_len: int = 140,
) -> None:
    """Plots an annotation with several signal values."""

    # Tokenizer
    tokenizer = get_tokenizer(annotation.generation.generation_cfg.model_name)

    # Iterating over signals
    plot_count = 0
    num_tokens = len(annotation.generation.output_tokens)
    for signal_value in signal_values_list:

        # Plotting the signal itself
        label = (
            f"{signal_value.signal_name}"
            if pretty_names is None
            else pretty_names[plot_count]
        )
        ax.plot(
            signal_value.values_np,
            label=label,
            color=DEFAULT_COLORS[plot_count % len(DEFAULT_COLORS)],
        )

        # Plotting the chunk-level score
        chunk_signal = []
        for chunk in annotation.token_chunks:
            chunk_value = 1 - np.prod(1 - signal_value.values_np[chunk])
            for _ in chunk:
                chunk_signal.append(chunk_value.item())
        ax.plot(
            chunk_signal,
            color=DEFAULT_COLORS[plot_count % len(DEFAULT_COLORS)],
            linestyle=":",
            label="Agg. Claim UQ",
            alpha=1.0,
        )

        # Updating
        plot_count += 1

    # Plotting the text
    tokens = [tokenizer.decode(elt) for elt in annotation.generation.output_tokens]
    ax.xaxis.set_ticks_position("top")
    ax.set_xlim(-1, len(annotation.generation.output_tokens))
    ax.tick_params(axis="x", which="both", top=True, bottom=False, length=5)
    ax.set_xticks(
        ticks=range(num_tokens), labels=tokens, rotation=60, ha="left", fontsize=9
    )

    # Modify the color after setting the tick labels
    tick_labels = ax.get_xticklabels()
    tick_lines = ax.get_xticklines()

    current_chunk_idx = 0
    for idx in range(num_tokens):

        if idx not in annotation.token_chunks[current_chunk_idx]:
            current_chunk_idx += 1

        if "gpt-4o" in annotation.labels:
            color = PLOT_COLORS[annotation.labels["gpt-4o"][current_chunk_idx]]
        elif "mushroom" in annotation.labels:
            color = PLOT_COLORS[annotation.labels["mushroom"][current_chunk_idx]]
        else:
            raise RuntimeError(f"No label found in annotation: {annotation.id}")

        tick_labels[idx].set_color(color)
        tick_lines[2 * idx].set_markeredgecolor(color)  # top tick
        tick_lines[2 * idx + 1].set_markeredgecolor(color)  # bottom tick

    # Dealing with the labels, etc
    escaped_output = annotation.generation.output.replace("$", r"\$")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("UQ score")
    ax.set_title(
        textwrap.fill(escaped_output, width=title_char_len),
        fontdict=dict(fontsize=11),
    )
    ax.grid(axis="x", color="grey", linestyle="--", linewidth=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(ncol=2)


def get_plot(
    annotation: Annotation,
    signal_values_list: UQSignalValue | t.List[UQSignalValue],
    pretty_names: t.Optional[t.List[str]] = None,
    title_char_len: int = 140,
) -> Figure:
    """Plots an annotation with several signal values."""

    if not isinstance(signal_values_list, list):
        signal_values_list = [signal_values_list]

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_on_ax(
        annotation=annotation,
        ax=ax,
        signal_values_list=signal_values_list,
        pretty_names=pretty_names,
        title_char_len=title_char_len,
    )
    fig.tight_layout()

    return fig


def get_plot_debug_token_proba(
    annotation: Annotation,
    delta: int = N_LOGITS_SAVED,
    only_indices: list[int] | None = None,
) -> Figure:
    """Plots an annotation with several signal values."""

    # Init
    num_rows = (
        len(annotation.generation.output_tokens)
        if only_indices is None
        else len(only_indices)
    )
    num_tokens = len(annotation.generation.output_tokens)
    tokenizer = get_tokenizer(annotation.generation.generation_cfg.model_name)
    proba_of_sampled = [
        annotation.generation.topn_prob_dicts[tok_idx][
            annotation.generation.output_tokens[tok_idx]
        ]
        for tok_idx in range(num_tokens)
    ]
    rank_of_sampled = [
        sorted(
            annotation.generation.topn_prob_dicts[tok_idx].values(), reverse=True
        ).index(proba_of_sampled[tok_idx])
        for tok_idx in range(num_tokens)
    ]
    fig, axes = plt.subplots(num_rows, 1, figsize=(6, 3 * num_rows), sharex=False)

    # Getting sorted values and token indices
    sorted_values, sorted_token_indices = [], []
    for idx_token in range(len(annotation.generation.output_tokens)):
        sorted_values.append(
            sorted(
                annotation.generation.topn_prob_dicts[idx_token].values(), reverse=True
            )[:delta]
        )
        sorted_token_indices.append(
            sorted(
                annotation.generation.topn_prob_dicts[idx_token],
                key=annotation.generation.topn_prob_dicts[idx_token].get,
                reverse=True,
            )[:delta]
        )

    sorted_values = torch.Tensor(sorted_values).float()
    sorted_token_indices = torch.Tensor(sorted_token_indices).int()

    # In case there's only one row, axes is not a list, so we wrap it
    if num_rows == 1:
        axes = [axes]

    iterator = only_indices if only_indices is not None else list(range(num_tokens))
    for ax_idx, idx in enumerate(iterator):
        ax = axes[ax_idx]
        rank_for_idx = int(rank_of_sampled[idx])
        decoded_tokens = [
            tokenizer.decode(sorted_token_indices[idx, rank].cpu())
            for rank in range(delta)
        ]
        local_entropy = entropy(sorted_values[idx, :delta].cpu()) / np.log(24)
        ax.plot(
            range(delta), sorted_values[idx, :delta].cpu(), marker="x", label="proba"
        )
        ax.axvline(
            rank_for_idx,
            ymin=0,
            ymax=sorted_values[idx, rank_for_idx].cpu().item(),
            color="green",
            label="selected",
        )

        ax.set_title(
            f"Token index {idx}: {tokenizer.decode(annotation.generation.output_tokens[idx])} [entropy = {local_entropy:.4f}]"
        )
        ax.set_ylim(0, 1)
        ax.set_xticks(range(delta))
        ax.set_xticklabels(decoded_tokens, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    plt.show()

    return fig
