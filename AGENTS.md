# REMAG Agent Instructions

This document provides essential information for agentic coding agents operating in the REMAG repository.

## Project Overview
REMAG is a specialized metagenomic binning tool for recovering high-quality eukaryotic genomes. It uses a Siamese neural network with Barlow Twins loss, adaptive Leiden clustering, and a HyenaDNA-based eukaryotic filter.

## Development Environment & Commands

### Installation
- **Development Install:** `pip install -e ".[dev,plotting]"`
- **Dependencies:** Requires `miniprot` (usually via conda: `conda install -c bioconda miniprot`).

### Testing
- **Run all tests:** `pytest`
- **Run a single test file:** `pytest tests/test_clustering.py`
- **Run a specific test:** `pytest tests/test_clustering.py::test_cluster_contigs`
- **Run without slow tests:** `pytest -m "not slow"`

### Linting & Formatting
The project strictly follows `black` and `isort` conventions.
- **Format code:** `black .` (Line length: 88)
- **Sort imports:** `isort .`
- **Check styles:** `flake8 remag/`

## Code Style & Conventions

### Imports
- Follow `isort` (black profile):
    1. Standard library imports
    2. Third-party library imports (numpy, pandas, torch, etc.)
    3. Local `remag` imports
- Use absolute imports for local modules: `from .utils import setup_logging`.

### Naming Conventions
- **Modules/Packages:** `snake_case` (e.g., `remag/miniprot_utils.py`)
- **Functions & Variables:** `snake_case` (e.g., `generate_embeddings`)
- **Classes:** `PascalCase` (e.g., `PathManager`)
- **Constants:** `UPPER_SNAKE_CASE`

### Type Hinting
- Use Python 3.9+ type hints for all function signatures.
- Prefer `from typing import Dict, List, Union, Optional`.
- Define complex types in `remag/utils.py` (e.g., `FragmentDict`).

### Error Handling & Logging
- **Logging:** Use `loguru`. Avoid `print()`.
- **Exceptions:** Catch specific exceptions. Provide context in log messages.
- **CLI Exit:** In `core.py` or `cli.py`, use `sys.exit(1)` after logging an error.

### Data Handling
- **Pandas:** Used extensively for managing contig features, embeddings, and cluster assignments. Use vectorized operations where possible.
- **Torch:** Used for neural network training. Use `remag.utils.get_torch_device()` to ensure compatibility (CUDA/MPS/CPU).
- **Paths:** Use the `PathManager` class in `remag/utils.py` for all standard output file paths to maintain consistency.
- **Headers:** Use `extract_base_contig_name` from `remag.utils` when dealing with fragmented contig headers.

## Architecture Guidelines
- **Core Logic:** `remag/core.py` coordinates the pipeline stages.
- **Modularity:** Keep feature extraction (`features.py`), modeling (`models.py`), and clustering (`clustering.py`) decoupled.
- **Clustering:** REMAG uses an adaptive resolution approach. Be careful when modifying clustering logic as it impacts bin quality metrics.
- **Miniprot:** Used for quality assessment. Ensure temporary files are managed or cleaned up via `PathManager`.

## Committing Changes
- Do not add "Claude" or any agent name as co-author.
- Use concise, imperative commit messages (e.g., "fix: handle empty fasta in feature extraction").
