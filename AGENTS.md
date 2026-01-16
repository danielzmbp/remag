# REMAG Agent Guidelines

This document provides instructions and guidelines for AI agents operating within the REMAG codebase.
REMAG is a specialized metagenomic binning tool for recovering eukaryotic genomes.

## 1. Environment & Build

### Dependencies
- **Core**: Python 3.9+, PyTorch, NumPy, Pandas, Scikit-learn, Leidenalg.
- **External**: `miniprot` (via bioconda) is required for core gene analysis.
- **CLI**: `rich-click` is used for the command-line interface.
- **Logging**: `loguru` is used for all logging.

### Installation
```bash
# Development installation (editable mode with dev dependencies)
pip install -e ".[dev]"

# Install optional plotting dependencies
pip install ".[plotting]"
```

### Running Tests
The project uses `pytest` for testing.
```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_clustering.py

# Run a single test case
pytest tests/test_clustering.py::TestKNNGraph::test_construct_graph_minimal_case

# Run tests with markers (e.g., skip slow tests)
pytest -m "not slow"
```

### Linting & Formatting
Code quality is enforced using `black`, `isort`, and `flake8`.
```bash
# Format code (line length 88)
black .

# Sort imports
isort .

# Check for linting errors
flake8
```

---

## 2. Code Style & Conventions

### General
- **Language**: Python 3.9+
- **Paradigm**: Functional and Object-Oriented mix.
- **Path Handling**: Use `os.path.join` or `pathlib` for cross-platform compatibility.
- **Configuration**: `pyproject.toml` is the source of truth for build/tool config.

### Formatting
- **Style**: Follows `black` default style.
- **Line Length**: 88 characters.
- **Indentation**: 4 spaces.

### Imports
- **Sorting**: Handled by `isort` (profile "black").
- **Grouping**: Standard library -> Third party -> Local application.
- **Local Imports**: Use relative imports within the `remag` package (e.g., `from .utils import setup_logging`).

### Naming Conventions
- **Variables/Functions**: `snake_case` (e.g., `get_features`, `cluster_contigs`).
- **Classes**: `CamelCase` (e.g., `GraphManager`).
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_BATCH_SIZE`).
- **Files**: `snake_case.py`.

### Typing
- **Type Hints**: Use Python type hints (`typing` module) for new code to improve readability and tooling support.
- **Legacy Code**: Existing code may be loosely typed; maintain consistency with surrounding code.

### Logging
- **Library**: strictly use `loguru`. Do not use standard `logging`.
- **Usage**:
  ```python
  from loguru import logger
  
  logger.info("Starting analysis...")
  logger.debug(f"Processing {len(items)} items")
  logger.warning("Low coverage detected")
  logger.error(f"Failed to process: {error}")
  ```
- **Verbosity**: Controlled via CLI arguments (`-v/--verbose`).

### Error Handling
- **Exceptions**: Catch specific exceptions where possible.
- **CLI Errors**: In the CLI entry points (`main`), catch top-level exceptions, log them with `logger.error`, and exit with `sys.exit(1)`.
- **Graceful Failure**: Ensure intermediate files are cleaned up or preserved based on user flags (`--keep-intermediate`).

### CLI Development
- **Framework**: Use `rich-click` (drop-in replacement for `click` with better formatting).
- **Arguments**: Follow existing patterns in `cli.py`.
- **Help**: Provide helpful descriptions for all arguments.

### Documentation
- **Docstrings**: All public modules, classes, and functions must have docstrings.
- **Style**: Google or NumPy style docstrings are preferred.
- **Comments**: Comment complex logic, but prefer self-documenting code.

---

## 3. Architecture & Patterns

### Directory Structure
- `remag/`: Source code.
  - `core.py`: Main execution flow.
  - `cli.py`: Command-line interface definition.
  - `clustering.py`: Leiden clustering logic.
  - `models.py`: Neural network models (Siamese network).
  - `features.py`: Feature extraction (k-mers, coverage).
  - `hyenadna_classifier/`: Eukaryotic filtering model.
- `tests/`: Unit and integration tests.

### Key Workflows
1.  **Filtering**: HyenaDNA classifier filters non-eukaryotic contigs.
2.  **Feature Extraction**: Generates k-mer and coverage profiles; handles augmentation.
3.  **Embedding**: Siamese neural network (Barlow Twins loss) creates contig embeddings.
4.  **Clustering**: Graph-based Leiden clustering on embeddings.
5.  **Refinement**: Checks for core gene duplications (using `miniprot`) and refines bins.
6.  **Rescue**: Rescues small fragmented bins.

### Performance
- **Parallelism**: Use `multiprocessing` or `joblib` for CPU-intensive tasks (feature extraction). Respect `args.threads`.
- **Vectorization**: Use `numpy` and `torch` for numerical operations. Avoid explicit loops over large datasets.
- **Memory**: Be mindful of memory usage with large metagenomic datasets.

### External Tools
- **Miniprot**: Used for quality assessment. Ensure logic handles its absence or failure gracefully if possible, though it is a core requirement.
- **Paths**: Do not hardcode paths to external tools or data.

## 4. Testing Guidelines

- **Unit Tests**: Focus on individual components (e.g., graph construction, specific math functions).
- **Mocking**: Use `unittest.mock` to mock external calls (e.g., file I/O, `miniprot` execution).
- **Integration Tests**: `test_integration.py` runs the full pipeline. Use small synthetic datasets for these.
- **Markers**: Respect `slow` markers for long-running tests.

## 5. Version Control
- **Commits**: Write clear, descriptive commit messages.
- **Versioning**: Semantic versioning is used. Update `_version.py` and `pyproject.toml` when bumping versions.
