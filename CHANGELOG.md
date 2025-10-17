# Changelog

All notable changes to REMAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/danielzmbp/remag/compare/v0.2.5...HEAD
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
