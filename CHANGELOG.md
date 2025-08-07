# Changelog

All notable changes to REMAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/danielzmbp/remag/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/danielzmbp/remag/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/danielzmbp/remag/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/danielzmbp/remag/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/danielzmbp/remag/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/danielzmbp/remag/releases/tag/v0.1.0