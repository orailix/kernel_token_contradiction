# Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification

Jérémie Dentan, Alexi Canesse, Mahammed El Sharkawy, Sonia Vanier

*LIX (École Polytechnique, IP Paris, CNSR)*

Arxiv link: [https://arxiv.org/pdf/2608.22506](https://arxiv.org/pdf/2608.22506)

## Repository Overview

This repository provides the complete codebase to reproduce the experiments and results from our KTC paper. It contains:

- **`src/`**: Python source code implementing KTC and baseline signals
- **`scripts/`**: Bash scripts to reproduce all computations
- **`figures/`**: Jupyter notebooks to reproduce paper figures
- **`output/`**: Pre-computed data and signals (committed for reproducibility)

## Abstract

Claim-level Uncertainty Quantification (UQ) aims to mitigate the lack of reliability of Large Language Models (LLMs) by evaluating the factuality of each claim in their outputs. We introduce Kernel Token Contradiction (KTC), a lightweight approach to compute claim-level UQ under realistic white-box conditions. KTC represents the candidate tokens involved in LLM generation as a positive semi-definite kernel that integrates both the LLM’s conditional distribution and a token contradiction score. We then use the Von Neumann entropy to quantify the uncertainty of this kernel. To estimate token contradiction, we develop a new approach based on frequency statistics from the Wikipedia corpus. Although CPU-only, our approach achieves over an 8.2× speedup compared to state-of-the-art GPU-accelerated methods based on cross-encoders, and over a 65× speedup compared to CPU-only methods with comparable performance. Our evaluation spans two benchmarks across four European languages and 16 different models. KTC not only matches the average performance of existing methods but also outperforms them in high-precision regimes. This combination of computational efficiency and accuracy makes real-time monitoring of LLM outputs practical in production.

## Getting Started

### Prerequisites

- Python 3.12.0
- UV package manager (recommended) or pip
- GPU required for some baselines (CCP, SAR)

### Installation

```bash
git clone git@github.com:orailix/kernel_token_contradiction.git
cd kernel_token_contradiction
uv sync
```

Or with pip:
```bash
pip install -e .
```

## Reproducing the Results

### Step 1: Setup Data

```bash
bash scripts/1_data_setup.sh
```

Imports the MUCH benchmark and MU-SHroom dataset for all four languages (en, fr, de, es).

### Step 2: Compute Baseline Signals

**CPU Baselines (Max-L, Token Likelihood, Token Entropy):**
```bash
bash scripts/2a_baselines_cpu.sh
```

**GPU Baselines:**
```bash
# CCP - requires Nvidia GPU
sbatch scripts/2b_baselines_gpu_ccp.sh

# SAR - requires Nvidia GPU
sbatch scripts/2c_baselines_gpu_sar.sh
```

### Step 3: Compute KTC Signals

**Step 3a: Compute Replaceability (prerequisite for KTC):**
```bash
bash scripts/3a_ktc_replaceability.sh
```

Computes Wikipedia-based token neighbors for all 16 models and both position settings (0 and 1).
**Expected runtime:** ~6-11 hours (parallelized across models).

**Step 3b: Compute KTC Signals:**
```bash
bash scripts/3b_ktc_signals.sh [--force-recompute]
```

Explores the KTC hyperparameter space across multiple batches:
- **Batch 1:** Explores various values of (Nu, Tau, Token Position)
- **Batch 2:** Ablation study with different normalization methods (scale, mixture) and alpha values

Use `--force-recompute` to override existing computations.

## Reproducing Paper Figures

The `figures/` directory contains Jupyter notebooks to reproduce all paper visualizations:

- **`wikipedia_heuristic.ipynb`**: Analysis of Wikipedia-based token relationships
- **`motivating_example.ipynb`**: Visual demonstration of KTC on example claims
- **`benchmark.ipynb`**: Full benchmark results comparing KTC against baselines

## KTC Configuration

KTC has several configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adjacency_method` | `"wiki"` | Method for adjacency matrix (`wiki`, `nli_base`, `nli_gate`) |
| `kernel_method` | `"heat"` | Kernel method (`heat`, `clip`, `augm`, `square`) |
| `normalization_method` | `"scale"` | Normalization method (`scale`, `mixture`, `none`) |
| `delta` | `24` | Number of candidate tokens to consider |
| `nu` | `8` | Neighbor set size for Wikipedia adjacency |
| `tau` | `0.3` | Diffusion time for heat kernel |
| `token_position` | `0` | Position for Wikipedia adjacency (0 or 1) |
| `language_specific` | `False` | Use language-specific tokenizers |
| `prefix_entail` | `True` | Whether prefix implies entailment |

The script `3b_ktc_signals.sh` explores different configurations of these parameters across multiple batches.

## Computation Times

| Component | Time | Hardware |
|-----------|------|----------|
| Replaceability (3a) | 6-11 hours | CPU (parallelized) |
| KTC Signals (3b) | ~1 min | CPU |
| CCP Baseline | ~30-60 min | Nvidia A100 |
| SAR Baseline | ~10-20 min | Nvidia A100 |
| CPU Baselines | ~1 min | CPU |

## Acknowledgement

The `prompt`, `wiki_url`, and `lang` fields of the MUCH samples are extracted from the Mu-SHROOM [1], a dataset released under CC-BY-4.0 license.

This work received financial support from the research chair *Trustworthy and Responsible AI* at École Polytechnique.

This work was granted access to the HPC resources of IDRIS under the allocation **AD011014843R1**, made by GENCI.

[1] Raúl Vázquez, Timothee Mickus, Elaine Zosa, Teemu Vahtola, Jörg Tiedemann, Aman Sinha, Vincent Segonne, Fernando Sánchez-Vega, Alessandro Raganato, Jindřich Libovický, Jussi Karlgren, Shaoxiong Ji, Jindřich Helcl, Liane Guillou, Ona de Gibert, Jaione Bengoetxea, Joseph Attieh, Marianna Apidianaki. *SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared Task on Hallucinations and Related Observable Overgeneration Mistakes* ArXiv preprint. 2025. [https://arxiv.org/abs/2504.11975](https://arxiv.org/abs/2504.11975)

## Copyright and License

Copyright 2025–present Laboratoire d'Informatique de l'École Polytechnique.

This repository is released under the Apache-2.0 license.

Please cite our work as:

```bibtex
@misc{dentan_ktc_2026,
  title = {Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification},
  author = {Dentan, Jérémie and Canesse, Alexi and El Sharkawy, Mahammed and Vanier, Sonia},
  year = {2026},
  url = {https://arxiv.org/pdf/2608.22506},
}
```
