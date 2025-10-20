# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `remag/`, with `cli.py` exposing the `remag` entry point, `core.py` orchestrating bin refinement, and `xgbclass/` bundling pretrained XGBoost models. Tests sit in `tests/`, following a mirror of the package layout for quick cross-referencing. Helper utilities reside in `scripts/`, while `build/` and `dist/` hold generated artifacts—do not edit these directly.

## Build, Test, and Development Commands
Install the project in editable mode with development extras via `pip install -e ".[dev]"`; run it inside a fresh Python 3.9+ virtual environment. Execute unit and integration checks with `pytest`, or use `pytest -m "not slow"` when iterating locally. After refactors, run `pytest --cov=remag` to confirm coverage stays aligned with existing modules.

## Coding Style & Naming Conventions
Code is formatted with Black (line length 88) and imports are sorted by isort’s Black profile; apply both using `black remag tests` and `isort remag tests`. Lint with `flake8 remag tests` to catch style and complexity issues. Prefer descriptive module-level functions, snake_case for variables, CapWords for classes, and keep docstrings focused on biological or algorithmic intent.

## Testing Guidelines
Place new tests under `tests/` using `test_<feature>.py` filenames and mirror the namespace of the code under test. Mark expensive workflows with `@pytest.mark.slow` or `@pytest.mark.integration` so they can be excluded in quick runs. When adding models or pipelines, include fixtures that cover both success paths and critical edge cases, and update coverage thresholds if substantial new logic is introduced.

## Commit & Pull Request Guidelines
Write commits in imperative mood (e.g., `Refine bin clustering heuristics`) and keep subjects under ~72 characters; when appropriate, adopt Conventional prefixes such as `feat:` or `fix:` as seen in recent history. Each pull request should summarize motivation, list key changes, link related issues, and paste the output of `pytest` (and coverage when relevant). Attach screenshots or CLI transcripts when modifying user-facing reporting so reviewers can verify formatting without rerunning full pipelines.

## Data & Model Artifact Notes
Prepackaged FASTA references live in `remag/db/`, and gradient boosting checkpoints reside under `remag/xgbclass/models/`; update them only when regenerating from trusted pipelines. Large binaries should not be committed directly—document download steps in `scripts/` or the README instead. When experimenting with new models, store interim outputs in ignored locations and provide reproducible scripts before promoting them into the package.
