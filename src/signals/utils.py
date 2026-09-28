import json
import typing as t
from functools import lru_cache

import numpy as np
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    DebertaV2TokenizerFast,
)
from transformers.utils import cached_file

from ..generation import get_tokenizer, get_tokenizer_inverse_vocab
from ..replaceability import load_sorted_neighbors
from ..utils.constants import DEVICE, LLMS_INFOS

EPSILON = 1e-5


######################################################################
############################ GET ADJACENCY ###########################
######################################################################


def get_most_frequent_neighbors(
    tok: int, nu: int, model_name: str, lang: str = None, position: int = 0
) -> t.Set[int]:
    """
    Get the index of the top-nu most frequent neighbors of a token in Wikipedia (English).

    Parameters
    ----------
    tok :
        The index of the token in the vocabulary.
    nu :
        The number of neighbors to consider.
    model_name :
        The name of the model, used to select the correct tokenizer.
    lang :
        The language of the token
    position :
        The position of the neighbor for wiki adjacency.

    Returns
    -------
        The set of neighbors of the target token. It has a size of `nu`.
    """
    neighbors = load_sorted_neighbors(model_name, lang=lang, position=position)
    return set(neighbors[tok][:nu])


def get_wiki_adcacency_weights(
    set_neighbor_a: t.Set[int],
    set_neighbor_b: t.Set[int],
) -> float:
    """
    Computes the intersection of the neighbor token sets of two tokens.

    Parameters
    ----------
    set_neighbor_a :
        The set of neighbors of the first token.
    set_neighbor_b :
        The set of neighbors of the second token.

    Returns
    -------
        One minus the proportion of common neighbors between the two target tokens.
    """
    # Special case when one of the two tokens has no neighbors
    # In that case its likely <eos> so the set similarity is zero
    # And the non-contradiction is 1
    if len(set_neighbor_a) == 0 or len(set_neighbor_b) == 0:
        return 1

    # Intersection and output
    intersect = len(set_neighbor_a.intersection(set_neighbor_b))
    return 1 - intersect / min(len(set_neighbor_a), len(set_neighbor_b))


def get_wiki_adjacency(
    token_list: t.List[int],
    nu: int,
    model_name: str,
    position: int = 0,
    prefix_entail: bool = False,
    lang: str = None,
) -> torch.Tensor:
    """
    Computes the adjacency matrix of a list of tokens based on Wikipedia tokenization.

    Parameters
    ----------
    token_list :
        The list of tokens to compute the similarity matrix of.
    nu :
        The number of neighbors to consider.
    model_name :
        The name of the model, used to select the correct tokenizer.
    position :
        The position of the neighbor for wiki adjacency.
    prefix_entail :
        Whether or not prefix implies entailment.
    lang :
        If available, the language of the tokens.

    Returns
    -------
        The square matrix of shape (n_token, n_token) containing the wiki adjacency values.
        Its value is 1-(proportion of common token in the neighbor sets).
    """

    # Init the result
    # We set the diagonal to zero because token similarity is similar to a contradiction measure
    result = np.ones((len(token_list), len(token_list)))

    # Getting the neighbors
    neighbor_sets = []
    for tok in token_list:
        neighbor_sets.append(
            get_most_frequent_neighbors(
                tok, nu=nu, model_name=model_name, lang=lang, position=position
            )
        )

    # Pairwise comparison
    for idx_a in range(len(token_list)):
        for idx_b in range(idx_a + 1, len(token_list)):
            tok_a, tok_b = token_list[idx_a], token_list[idx_b]
            if prefix_entail and is_one_prefix_of_other(
                tok_a, tok_b, model_name=model_name
            ):
                sim = 1.0
            else:
                sim = get_wiki_adcacency_weights(
                    neighbor_sets[idx_a],
                    neighbor_sets[idx_b],
                )
            result[idx_a, idx_b] = sim
            result[idx_b, idx_a] = sim

    return torch.Tensor(result)


@lru_cache()
def get_nli_model(nli_model) -> DebertaV2ForSequenceClassification:
    """Gets the NLI model."""
    model: DebertaV2ForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(nli_model).to(DEVICE)
    )
    model.eval()
    return model


@lru_cache()
def get_nli_tokenizer(nli_model) -> DebertaV2TokenizerFast:
    """Gets the NLI tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    return tokenizer


def get_nli_adjacency(
    token_list: t.List[str],
    preceding_str: str,
    succeeding_str: str,
    do_gating: bool,
    nli_model: str,
) -> torch.Tensor:
    """Computes a NLI adjacency matrix.

    Parameters
    ----------
    token_list :
        The list of candidate tokens
    preceding_str :
        The string that preceeded in the sentence, that will be passed
        to the NLI model.
    succeeding_str :
        The string that succedes in the sentence, that will be passed
        to the NLI model.
    do_gating :
        If False, the NLI score is (1 - contradiction).
        If True, it is (1 - contradiction) * entailment
    nli_model :
        HuggingFace identifier of the NLI model to use.

    Returns
    -------
        The square matrix of shape (n_token, n_token) containing the NLI adjacency values."""

    n = len(token_list)
    premises = []
    hypotheses = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            premises.append(preceding_str + token_list[i] + succeeding_str)
            hypotheses.append(preceding_str + token_list[j] + succeeding_str)

    batch = get_nli_tokenizer(nli_model)(
        premises, hypotheses, truncation=False, padding=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = get_nli_model(nli_model)(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        contradictions = probs[:, 2]
        entailment = probs[:, 0]

    # Output
    if do_gating:
        result_linear = (1 - contradictions) * entailment
    else:
        result_linear = 1 - contradictions

    # Reshaping
    result = torch.ones((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result[i, j] = result_linear[idx]
            idx += 1

    # Symetrizing + output
    result = (result + result.T) / 2
    return result.cpu()


@lru_cache
def get_model_embedding(model_name: str) -> torch.Tensor:
    """Gets the weights of the emebdding block of the model."""
    # Resolve embedding key
    embed_key = LLMS_INFOS[model_name]["embed_key"]

    # Locate the safetensors index
    try:
        index_file = cached_file(
            model_name,
            "model.safetensors.index.json",
        )
    except OSError:
        index_file = cached_file(
            model_name,
            "pytorch_model.bin.index.json",
        )

    with open(index_file, "r") as f:
        index = json.load(f)

    # Checking embed_key
    if embed_key not in index["weight_map"]:
        raise KeyError(
            f"Embedding key {embed_key} not found. "
            f"Available keys include: {list(index['weight_map'])[:5]} ..."
        )

    # Find which shard contains the embedding tensor
    shard_file = index["weight_map"][embed_key]

    shard_path = cached_file(
        model_name,
        shard_file,
    )

    # Load only that shard
    try:
        state = load_file(shard_path, device="cpu")
    except SafetensorError:
        state = torch.load(shard_path, map_location="cpu")

    result = state[embed_key]

    # Normalizing
    result = result / result.norm(p=2, dim=1, keepdim=True)
    return result


def get_embedding_similarity(
    target_token: int, candidates: list[int], model_name: str
) -> torch.Tensor:
    """Similarity between token embedding at model's level."""
    embedding = get_model_embedding(model_name)
    result = (
        embedding[torch.Tensor(candidates).to(torch.long)] @ embedding[target_token]
    )
    result = torch.clip(result, min=0, max=None)
    return result.float().cpu()


def get_embedding_adjacency(candidates: list[int], model_name: str) -> torch.Tensor:
    """Similarity between token embedding at model's level."""
    embedding = get_model_embedding(model_name)
    result = (
        embedding[torch.Tensor(candidates).to(torch.long)]
        @ embedding[torch.Tensor(candidates).to(torch.long)].T
    )
    result = torch.clip(result, min=1e-8, max=None)
    return result.float().cpu()


def is_one_prefix_of_other(tok_a: int, tok_b: int, model_name: str) -> bool:
    """Indicates if one of the two tokens is prefix of the other."""
    tok_a_decoded = get_tokenizer_inverse_vocab(model_name)[tok_a]
    tok_b_decoded = get_tokenizer_inverse_vocab(model_name)[tok_b]
    min_len = min(len(tok_a_decoded), len(tok_b_decoded))
    return tok_a_decoded[:min_len] == tok_b_decoded[:min_len]


######################################################################
############################### KERNELS ##############################
######################################################################


def clip_neg_eigs(matrix: torch.Tensor) -> torch.Tensor:
    """
    Clips the negative eigenvalues of a symetric matrix.
    The result is also normalized to have a trace of 1.0.

    Parameters
    ----------
    matrix :
        The matrix. It should be real and symetric.

    Returns
    -------
        The metrix with clipped eigenvalues, in the same basis as the original one.
    """
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals_clipped = torch.clamp(eigvals, min=EPSILON)
    result = (eigvecs * eigvals_clipped) @ eigvecs.T
    return result / result.trace()


def augment_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """
    Augments the diagonal of a matrix to get a PSD matrix.
    The result is also normalized to have a trace of 1.0.

    Parameters
    ----------
    matrix :
        The matrix. It should be real and symetric.

    Returns
    -------
        The matrix with augmented diagonal value eigenvalues, in the same basis as the original one.
    """
    eigvals, _ = torch.linalg.eigh(matrix)
    shift_factor = torch.clamp(torch.amin(eigvals), max=-EPSILON)
    result = matrix - shift_factor * torch.eye(matrix.size(0))
    return result / result.trace()


def square_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the square of the matrix to ensure it's a PSD.
    The result is also normalized to have a trace of 1.0.

    Parameters
    ----------
    matrix :
        The matrix. It should be real and symetric.

    Returns
    -------
        The square matrix, which is PSD if `matrix` is symetric.
    """
    result = matrix @ matrix
    return result / result.trace()


def heat_kernel(adjacency: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    Compute the heat kernel of a graph.

    Parameters
    ----------
    adjacency :
        Input adjacency matrix. Should be symetric.

    tau :
        Heat diffusion time parameter. Higher t values diffuse the signal more widely across the graph.

    Returns
    ----------
        PSD heat kernel matrix.
    """

    degrees = adjacency.sum(dim=0)
    laplacian = torch.diag(degrees) - adjacency
    heat = torch.linalg.matrix_exp(-1 * tau * laplacian)
    heat = heat / heat.trace()

    return heat


######################################################################
############################ PRIOR + ENTROPY #########################
######################################################################


def von_neumann_entropy(psd_kernel: torch.Tensor, force_trace: bool = True) -> float:
    """
    Computes the Von Neumann entropy of a positive semi-definite matrix.

    Parameters
    ----------
    psd_kernel :
        A symmetric positive semi-definite matrix with trace 1.0.
    force_trace :
        If True, we will manually force the trave to be 1.0

    Returns
    -------
        The normalized Von Neumann entropy, scaled by log(dim), so that
        the result lies in [0, 1] when the input is PSD with trace 1.
    """
    if force_trace:
        psd_kernel = psd_kernel / psd_kernel.trace()

    eigvals = torch.linalg.eigh(psd_kernel.cpu()).eigenvalues
    eigvals = torch.clamp(eigvals, min=EPSILON)
    return (
        -1 * torch.sum(eigvals * torch.log(eigvals)).item() / np.log(psd_kernel.size(0))
    )
