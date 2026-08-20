# Changelog

All notable changes to REMAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.5] - 2026-08-20

### Added
- Support for interval COV/bedGraph coverage inputs, including compressed files
  and fragment-specific mean and standard deviation features.
- Continuous integration on Python 3.9 and 3.11 with tests, formatting, and lint
  checks.
- Automated Zenodo version publishing with current project metadata.

### Changed
- `--epochs` default: 400 to 100.
- `--barlow-lambda` default: 0.003 for all scenarios.
- CLI startup now uses lazy imports while retaining Rich help output.
- Package versions now come from installed distribution metadata and are checked
  against release tags.
- Simplified gene-mapping, feature-generation, rescue, and miniprot parsing paths;
  renamed the retained mapping directory to `temp_gene_mapping`.
- Updated the Docker build to Python 3.11, pinned miniprot 0.18, and current
  multi-platform publishing actions.
- Clarified core and intermediate output documentation.

### Fixed
- Use vectorized coverage transformation compatible with older pandas releases.
- Correct adjusted-k and final bin-count log messages.

### Removed
- Coverage-dependent auto-selection of the Barlow Twins lambda.
- `getattr` fallback defaults for `barlow_lambda` and `base_learning_rate` in
  `models.py`.
- Unused `--min-cluster-size` option.
- Obsolete standalone raw-coverage script.
- Redundant repository-local Bioconda release automation; upstream Bioconda
  updates remain unchanged.

## [0.4.4] - 2026-06-10

### Added
- Batched HyenaDNA filtering across contigs for faster GPU and MPS inference.
- Extra deterministic resampling for borderline two-window HyenaDNA calls.
- Regression tests for batched HyenaDNA prediction and filtering decisions.
- Regression coverage for `group_contigs_by_cluster`, contig-name suffix handling, `is_gzipped`, and `BarlowTwinsLoss`.

### Changed
- Use a recall-friendly HyenaDNA eukaryote filtering cutoff of 0.45 after resampling.
- Refactored miniprot command construction to remove duplicated command-list logic.

### Performance
- Optimized FASTA sequence writing and sequence formatting paths.
- Reduced unnecessary dictionary key-view and set allocations in utility and clustering paths.
- Hoisted repeated translation table creation out of k-mer feature mapping loops.
- Optimized core-gene occurrence statistics and duplicated-bin gene counting.
- Streamlined rescue and miniprot statistics loops by avoiding redundant key lookups.

### Fixed
- Corrected HyenaDNA window-count estimation when the final full window already reaches the sequence end.
- Ensure HyenaDNA filtering creates its output directory when called directly.
- Removed unused imports in rescue and HyenaDNA model code.

## [0.4.3] - 2026-05-09

### Fixed
- Preserved contig headers with decimal suffixes, such as SPAdes coverage values, when generating embeddings, clusters, bins, and plots.

## [0.4.2] - 2026-04-16

### Fixed
- Added `miniprot` to the Docker image so container runs include the required core gene annotation dependency.

### Changed
- Simplified the README structure, reduced redundant examples, and added an index for easier navigation.

## [0.4.1] - 2026-04-15

### Fixed
- Forward `--save-filtered-contigs` through the core pipeline so requested non-eukaryotic FASTA output is actually written.
- Accept shell-expanded coverage inputs reliably and align CLI defaults with the documented behavior.

### Changed
- Clarified installation guidance, CLI options, and output documentation.

## [0.4.0] - 2026-03-05

### Added
- Bin rescue strategy to merge fragmented bins under explicit duplication guardrails.
- `short-reads` / `sr` mode and CLI controls for rescue duplication limits.
- Reproducibility-oriented logging of the exact command line used for each run.
- Additional regression coverage for CLI defaults, rescue limits, and issue reproduction.

### Changed
- Replaced adaptive resolution selection with greedy Leiden clustering and removed the old refinement workflow.
- Updated clustering and rescue heuristics, including contamination-aware scoring, revised duplication thresholds, and adjusted default resolution sweeps.
- Refreshed the README to match the current pipeline and streamlined the Bioconda release flow to generate a recipe artifact for manual submission.

### Fixed
- `--filter-only` now exits after the filtering stage instead of being silently ignored.
- Duplicate alignment filenames are disambiguated using parent directories.
- Rescue-loop duplication handling and debug logging were cleaned up to avoid duplicate merges and excessive log spam.

## [0.3.4] - 2025-12-01

### Added
- Filter-only mode to run HyenaDNA screening and write the filtered FASTA without feature generation or binning.
- Mode presets for single-cell data that expand the k-NN graph, skip refinement, and bypass the eukaryotic filter to preserve low-input genomes.

### Changed
- Coassembly defaults tightened: auto Barlow Twins lambda now 0.005 for multi-sample runs, minimum contig length raised to 4096 bp when multiple coverage files are provided, and Leiden resolution sweeps trimmed with completeness-aware tie-breakers and early stopping to avoid over-splitting.
- Single-cell clustering tuned to prefer coarser Leiden resolutions (0.01 floor, capped at 1.0), with narrowed resolution sweeps to reduce fragmentation and clearer logging.
- Adaptive resolution selection now prioritizes completeness when duplication counts tie and halts sweeps when completeness collapses.
- Miniprot alignment thresholds raised for more reliable core gene detection.
- Training, HyenaDNA prediction, and clustering now seed all randomness for reproducible runs.

### Fixed
- Single-cell mode skips the eukaryotic filter by default to avoid discarding target contigs.
- Filter-only runs exit after writing the filtered FASTA instead of continuing into feature generation.

## [0.3.3] - 2025-11-04

### Added
- LRU cache for k-mer mapping to improve performance during refinement
- `--save-bins-before-refinement` option to preserve pre-refinement bin states
- Seeded random number generation for improved reproducibility

### Changed
- **Default minimum contig length increased from 1024bp to 4096bp**
- Improved resolution selection algorithm with better quality metrics
- Enhanced bin refinement algorithms for more accurate splitting
- Improved duplication detection thresholds to reduce false positives
- Removed redundant eukaryotic classification logging from clustering output

### Performance
- Feature tensor caching in SequenceDataset reduces redundant computations
- LRU cache significantly speeds up k-mer mapping lookups

## [0.3.2] - 2025-10-30

### Added
- Version number now included in `params.json` output for reproducibility tracking

### Changed
- **BREAKING**: Removed `transformers` library dependency in favor of standalone tokenizer
  - Significantly reduces installation size and dependency conflicts
  - Standalone character tokenizer with zero external dependencies
- Quality-aware resolution selection using SCG - 5*dups metric instead of pure contamination minimization
  - Balances completeness and contamination more effectively
  - Results in higher-quality bins overall
- Expanded refinement resolution testing from 14 to 32 fine-grained steps (0.07-3.0)
  - Improved precision in finding optimal split points for contaminated bins
  - Finer granularity around critical thresholds
- Reduced logging verbosity by converting many info/warning messages to debug level
  - Cleaner console output during normal operation
  - Use `--verbose` flag to see detailed debug information
- Lowered single-copy gene retention threshold from 80% to 75% in refinement validation

### Fixed
- Fragment generation now uses cryptographic hashing for deterministic, reproducible results
  - Replaced Python's randomized `hash()` with `hashlib.sha256()`
  - Ensures consistent results across different PYTHONHASHSEED values
- Suppressed pandas FutureWarning for fillna downcasting using `.infer_objects(copy=False)`
- Removed obsolete "Multiple bins detected" log message (redundant with adaptive resolution)

### Performance
- Deterministic fragment generation improves reproducibility across runs
- Quality-aware selection produces better bins with improved completeness/contamination balance

### Tuning
- Base learning rate: 0.005 → 0.0025 for more stable training
- Miniprot target coverage threshold: 0.50 → 0.45 for better gene detection sensitivity
- Max refinement rounds: 2 → 16 for more thorough bin refinement

## [0.3.0] - 2025-10-21

### Changed
- **BREAKING**: Replaced XGBoost classifier with HyenaDNA LLM-based model for eukaryotic filtering
  - Improved accuracy using pre-trained genomic foundation model
  - Probability-based classification with confidence scores
  - Adds window count and confidence metrics to classification output
- Reduced console output verbosity for cleaner logs
  - Training epoch prints reduced from every 5 to every 20 epochs
  - Resolution parameters formatted to 2 decimal places
  - Removed duplicate HyenaDNA filtering log message
- Adaptive resolution now enabled by default for automatic clustering optimization
  - Automatically determines optimal Leiden resolution based on core gene duplications
  - Tests multiple resolution values (0.7x, 1.0x, 1.4x) and selects best result
  - Can be disabled by explicitly providing `--leiden-resolution`

### Added
- Standalone HyenaDNA predictor for independent sequence classification
  - Self-contained script for eukaryotic vs prokaryotic classification
  - Minimal dependencies, can run without full REMAG pipeline
  - Includes example test data and conda environment specification
- Adaptive resolution determination system
  - Organism count estimation from core gene duplications
  - Multi-resolution testing to find optimal clustering parameters
- Gene mapping cache system for performance optimization
  - Caches miniprot PAF parsing results to avoid redundant runs
  - Significantly speeds up resolution testing and refinement (~10x faster)
- Batch processing controls for memory management
  - `--coverage-batch-size` for controlling coverage calculation memory usage
  - `--hyenadna-batch-size` for controlling HyenaDNA inference batch size
- Barlow Twins training diagnostics
  - Detailed cross-correlation matrix statistics tracking
  - Separate invariance and redundancy loss components
  - Helps identify training issues like collapsed embeddings
### Removed
- XGBoost classifier and all related dependencies (xgboost, scikit-learn)
- Legacy xgbclass module and pre-trained models
- Unused adaptive strategy parameters that provided no performance benefit

### Fixed
- XGBoost import errors and classification column name mismatches
- HyenaDNA memory usage through streaming sequence processing

### Performance
- Gene mapping cache reduces refinement time by ~10x
- HyenaDNA streaming reduces memory footprint for large datasets
- Batch size controls allow tuning for available system resources

## [0.2.5] - 2025-10-17

### Changed
- Overhauled the fusion layer architecture to improve multi-modal feature integration.
- Always write `embeddings.csv` as part of core outputs for downstream analysis.
- Raised the minimum supported Python version to 3.9 to match packaging requirements.

### Fixed
- Guard coverage normalization when contigs report zero reads to avoid invalid values.
- Restore the best Siamese checkpoint weights before exporting the trained model.

## [0.2.4] - 2025-09-14

### Fixed
- Fixed critical edge cases in refinement module that were causing failures
- Fixed security vulnerability and optimized performance
- Skip bins without duplication data during refinement to avoid unnecessary processing

### Changed
- Implemented conservative refinement strategy to preserve completeness and avoid over-fragmentation
- Eliminated code duplication in DataFrame column initialization

### Added
- Enhanced test infrastructure for better reliability

## [0.2.3] - 2025-08-20

### Fixed
- Fixed undefined variable errors in Leiden reclustering by using correct GraphManager attributes
- Fixed missing sklearn imports (cosine_similarity) in clustering module
- Fixed missing miniprot dependency in Bioconda recipe

### Changed
- Updated README examples to use v0.2.2 in Docker/Singularity commands
- Cleaned up unused imports in clustering and models modules

### Removed
- Removed unused warnings filter and List import

## [0.2.2] - 2025-08-14

### Changed
- Major codebase refactoring for improved maintainability and reduced complexity
- Removed k-means pre-filtering from clustering pipeline for simplification
- Extracted coverage calculation into dedicated classes (BAMCoverageCalculator, TSVCoverageCalculator)
- Consolidated fragment processing logic with FragmentProcessor class
- Split large clustering functions into focused GraphManager and ClusteringManager classes
- Extracted training components into EarlyStoppingManager, LearningRateScheduler, and TrainingManager classes
- Added centralized PathManager class for consistent path handling
- Enhanced error handling with @handle_errors decorator
- Increased miniprot quality thresholds to 0.55 coverage and 0.35 identity for better results
- Enabled compressed database support with miniprot -I flag

### Added
- Gene mapping cache system to avoid redundant miniprot runs during refinement
- Fast cached core gene duplication checking for refinement steps
- Comprehensive functionality testing framework

### Fixed
- Resolved undefined variable errors from k-means pre-filtering removal
- Fixed clustering pipeline variable reference issues
- Maintained full backward compatibility with existing function signatures

### Performance
- Refinement steps now reuse gene mappings instead of re-running miniprot (~10x faster)
- Reduced computational overhead during iterative bin splitting
- Code reduction of ~300 lines while improving modularity and readability
- Better quality filtering with optimized thresholds

## [0.2.1] - 2025-08-11

### Fixed
- Fixed CLI missing parameters error that prevented command execution
- Removed unused HDBSCAN parameters from CLI function signature

## [0.2.0] - 2025-08-10

### Changed
- Removed HDBSCAN dependency, now uses Leiden clustering exclusively
- Improved refinement process with better embedding name handling
- Cleaned up unused refinement functions for better maintainability

### Fixed
- Fixed embedding name mismatch in refinement module that was causing failures
- Improved error handling for miniprot dependency checks
- Better GPU import exception handling for environments without CUDA

### Performance
- Parallelized Leiden k-NN graph construction for faster clustering
- Optimized refinement workflow by removing redundant functions

### Documentation
- Restructured installation documentation for clarity
- Updated build configuration and documentation consistency

## [0.1.5] - 2025-08-07

### Fixed
- Fixed missing `python-igraph` dependency in Bioconda recipe causing installation failures
- Fixed missing `leidenalg` dependency in Bioconda recipe
- Updated Bioconda recipe template to include all required dependencies

### Changed
- Improved Bioconda workflow reliability

## [0.1.4] - 2025-08-07

### Added
- Leiden clustering algorithm as an alternative to HDBSCAN
- New clustering parameters: `--clustering-method`, `--leiden-resolution`, `--leiden-k-neighbors`, `--leiden-similarity-threshold`
- Graph-based community detection for improved binning performance
- Singularity container support documentation

### Changed
- Default clustering method changed from HDBSCAN to Leiden for better results
- Improved clustering performance for complex metagenomic samples
- Enhanced CLI option grouping for better user experience
- Updated README with expanded container usage instructions

### Fixed
- Improved handling of single-cluster results with automatic reclustering

## [0.1.3] - 2025-08-05

### Fixed
- Fix MPS empty tensor error when n_coverage_features = 0
- Fix Docker Hub description length issue

### Changed
- Update default min bin size for better performance
- Switch to dynamic versioning with setuptools-scm
- Clean up GitHub Actions workflows

### Added
- Conda installation option in README
- Docker installation option in README

## [0.1.2] - 2025-08-03

### Added
- Docker support with automated Docker Hub publishing
- Dockerfile for containerized deployment
- Docker workflow automation for GitHub Actions

### Changed
- Enhanced release workflow to include Docker image publishing

## [0.1.1] - 2025-07-31

### Added
- Support for CRAM files (automatically detected by extension)
- MinMax scaler for coverage normalization

### Fixed
- Improved coverage scaling for better clustering performance

## [0.1.0] - 2025-07-26

### Added
- Initial release of REMAG
- Bacterial filtering using 4CAC XGBoost classifier
- Contrastive learning with Siamese neural networks
- HDBSCAN clustering for genome binning
- Quality assessment using eukaryotic core genes
- Command-line interface with rich-click
- GPU acceleration support via RAPIDS
- Automated release workflow for PyPI, Bioconda, and Zenodo
- Release documentation and checklist

### Features
- Processes mixed prokaryotic-eukaryotic metagenomes
- Generates high-quality eukaryotic MAGs
- Iterative refinement for contamination removal
- Multi-modal feature fusion (k-mer + coverage)
- Comprehensive logging and progress tracking

### Changed
- Updated package metadata for better discoverability

### Fixed
- Various bug fixes and improvements

[Unreleased]: https://github.com/danielzmbp/remag/compare/v0.4.5...HEAD
[0.4.5]: https://github.com/danielzmbp/remag/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/danielzmbp/remag/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/danielzmbp/remag/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/danielzmbp/remag/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/danielzmbp/remag/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/danielzmbp/remag/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/danielzmbp/remag/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/danielzmbp/remag/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/danielzmbp/remag/compare/v0.3.0...v0.3.2
[0.3.0]: https://github.com/danielzmbp/remag/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/danielzmbp/remag/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/danielzmbp/remag/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/danielzmbp/remag/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/danielzmbp/remag/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/danielzmbp/remag/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/danielzmbp/remag/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/danielzmbp/remag/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/danielzmbp/remag/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/danielzmbp/remag/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/danielzmbp/remag/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/danielzmbp/remag/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/danielzmbp/remag/releases/tag/v0.1.0
