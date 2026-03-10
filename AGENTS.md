# AGENTS.md

This file is for agentic coding tools operating in this repository.
It captures the local build, lint, test, and coding conventions that matter most in `remag`.

## Repository overview

- Project type: Python package with CLI entrypoint `remag`
- Packaging: `setuptools` via `pyproject.toml`
- Main package: `remag/`
- Tests: `tests/`
- Helper scripts: `scripts/`
- Release/build automation: `.github/workflows/`

## Architecture map

- `remag/cli.py`: Click/rich-click CLI, argument normalization, user-facing validation
- `remag/core.py`: top-level pipeline orchestration
- `remag/features.py`: FASTA preprocessing, feature extraction, filtering, coverage prep
- `remag/models.py`: neural network training and embedding generation
- `remag/clustering.py`: graph construction, clustering, and bin quality logic
- `remag/miniprot_utils.py`: miniprot integration and core-gene mapping/duplication checks
- `remag/rescue.py`: fragmented-bin rescue logic
- `remag/output.py`: writing final outputs
- `remag/utils.py`: shared utilities and logging setup
- `remag/hyenadna_classifier/`: embedded classifier subsystem; keep changes isolated unless needed

## Setup

Recommended development install:

```bash
pip install -e ".[dev]"
```

Optional plotting extras:

```bash
pip install -e ".[dev,plotting]"
```

External dependency note:

- `miniprot` is required for core-gene analysis in real pipeline runs
- install separately, typically with `conda install -c bioconda miniprot`
- many unit tests do not require `miniprot`

## Build commands

Build the package exactly as CI/release does:

```bash
python -m build
```

Useful local install check:

```bash
python -m pip install -e .
```

CLI smoke check:

```bash
python -m remag --help
remag --help
```

## Lint and formatting commands

Format code:

```bash
black .
```

Sort imports:

```bash
isort .
```

Lint:

```bash
flake8 remag tests
```

When making Python edits, a safe local quality pass is:

```bash
isort . && black . && flake8 remag tests
```

## Test commands

Run the full suite:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_cli_defaults.py
```

Run a single test function:

```bash
pytest tests/test_cli_defaults.py::TestCliDefaults::test_default_values_no_coverage
```

Run tests by keyword:

```bash
pytest -k clustering
```

Run by marker:

```bash
pytest -m "not slow"
pytest -m integration
pytest -m benchmark
```

Collect tests without running them:

```bash
pytest --collect-only
```

The repo registers these markers in `pyproject.toml`:

- `slow`
- `integration`
- `benchmark`

Do not introduce new markers without updating `pyproject.toml`.

## Pytest notes

- Test discovery is rooted at `tests/`
- Shared fixtures live in `tests/conftest.py`
- Tests commonly use synthetic `numpy`/`pandas` data instead of full real datasets
- CLI tests typically use `click.testing.CliRunner`
- Integration boundaries are often patched with `unittest.mock.patch`
- It is acceptable here to test private helpers when the behavior is algorithmically important

## Code style

Follow existing repo style before introducing new patterns.

- Imports: group as standard library, third-party, then local `remag` imports
- Separate import groups with one blank line
- Use `isort` with the Black profile; do not hand-format imports into a conflicting style
- Formatting target is Black with line length 88
- Prefer clear temporary variables over compressed one-liners in pipeline code
- Add comments only when a step is non-obvious; avoid noisy narration
- Keep module docstrings short and direct when present

## Types

- Type hints are selective, not strict across the whole project
- Add types where they clarify helper APIs, data contracts, or return shapes
- Do not force comprehensive typing onto legacy pipeline code unless you are already refactoring it
- The `remag/hyenadna_classifier/` package uses more explicit typing; match local style there
- Existing aliases in `remag/utils.py` are a good model for lightweight clarity

## Naming conventions

- Functions, variables, modules, and test files: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Internal helpers: prefix with `_` when they are intentionally private
- Stateful coordinators commonly use `*Manager` names
- Path helpers commonly use `get_*_path`
- Test names should describe behavior or regression intent, not implementation trivia

Preserve domain-specific literals when they matter:

- cluster labels use `bin_{n}`
- unassigned/noise cluster uses `noise`

## CLI conventions

- Keep CLI-specific validation in `remag/cli.py`
- Keep business logic out of Click callbacks when possible
- The CLI currently normalizes inputs and passes an `argparse.Namespace` into `remag.core.main`
- Prefer Click exceptions such as `click.BadParameter` and `click.UsageError` for user input problems
- Preserve the quick-help/full-help split between `-h` and `--help`

## Logging and error handling

- Package code uses `loguru` logging configured via `remag.utils.setup_logging()`
- Prefer `logger.info` for stage transitions and major pipeline milestones
- Use `logger.debug` for cache hits, parameter details, and implementation detail
- Use `logger.warning` when execution can continue with degraded behavior
- Use `logger.error` when a stage fails or the program is about to exit
- At pipeline boundaries, broad exception handling is already used; if you keep that pattern, log enough context to diagnose failures
- In standalone scripts, stdlib `logging` is acceptable if that file already uses it

## Data and pipeline behavior

- This codebase is cache/output-directory driven; preserve reuse of existing outputs when present
- Do not casually bypass cache checks for embeddings, graph artifacts, or gene-mapping artifacts
- Be careful with expensive recomputation in feature generation, miniprot work, and clustering
- Avoid changing output filenames or directory layout without updating tests and docs

## Testing expectations for agents

- Prefer the smallest test scope that validates your change
- For CLI changes, start with a focused `tests/test_cli_defaults.py`-style invocation
- For clustering or math changes, add deterministic synthetic-data tests with fixed random seeds
- If you touch performance-sensitive code, keep or improve existing runtime thresholds
- If a change affects markers, fixtures, or CLI defaults, update tests in the same patch

## Things to avoid

- Do not add new heavyweight dependencies without strong justification
- Do not move substantial logic into scripts when it belongs in package modules
- Do not replace `loguru` with another logging style inside package code
- Do not silently change CLI defaults; tests cover several derived defaults
- Do not introduce unregistered pytest markers

## CI and release notes

- Build command in release workflow: `python -m build`
- Release workflow publishes to PyPI and triggers Docker publishing
- Docker release metadata is defined in `.github/workflows/docker-publish.yml`
- If packaging metadata changes, check `pyproject.toml`, release workflow, and Docker expectations together

## Editor/assistant rules present in repo

No repo-specific Cursor or Copilot instruction files were found at the time this file was generated:

- no `.cursor/rules/`
- no `.cursorrules`
- no `.github/copilot-instructions.md`

If any of those files are added later, update this document so agents inherit those rules.

## Recommended default workflow for agents

1. Read `pyproject.toml` and the touched module before editing.
2. Match local patterns in the specific subsystem you are changing.
3. Make the smallest safe change.
4. Run focused tests first, then broader checks if warranted.
5. Note any external-tool assumptions such as `miniprot` in your final report.
