# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

REMAG is a specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples. It uses contrastive learning with Siamese neural networks to generate meaningful contig embeddings for clustering.

## Development Commands

### Installation
```bash
# Development installation (preferred for contributors)
conda create -n remag python=3.9
conda activate remag
pip install -e ".[dev]"

# Production installation
pip install remag

# GPU-accelerated installation (optional)
pip install remag[gpu]

# From source
git clone https://github.com/danielzmbp/remag.git
cd remag
pip install .
```

### Testing
```bash
# Run tests (when implemented)
pytest

# Run tests with coverage
pytest --cov=remag

# Run slow tests (training/clustering workflows)
pytest -m slow

# Run integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"
```

### Code Quality
```bash
# Format code (line length: 88)
black .

# Sort imports (black-compatible profile)
isort .

# Lint code
flake8

# Run all code quality checks
black . && isort . && flake8
```

### Running REMAG
```bash
# Command line interface (preferred)
remag -f contigs.fasta -b alignments.bam -o output_directory

# Python module mode (if CLI has issues)
python -m remag -f contigs.fasta -b alignments.bam -o output_directory

# Python script mode (development)
python remag.py -f contigs.fasta -b alignments.bam -o output_directory

# With coverage from TSV file instead of BAM
remag -f contigs.fasta -t coverage.tsv -o output_directory

# Enable debug logging
remag -f contigs.fasta -b alignments.bam -o output_directory --verbose
```

## Architecture

### Core Pipeline (remag/core.py)
The main execution logic follows this flow:
1. **Bacterial Filtering**: Uses 4CAC XGBoost classifier to filter bacterial contigs
2. **Feature Extraction**: Combines k-mer composition (4-mers) with coverage profiles
3. **Contrastive Learning**: Trains Siamese network with Barlow Twins loss
4. **Clustering**: HDBSCAN clustering on embeddings
5. **Quality Assessment**: Uses miniprot against eukaryotic core genes
6. **Refinement**: Iterative bin splitting based on core gene duplications

### Key Modules

#### Neural Network (remag/models.py)
- **SiameseNetwork**: Dual-encoder architecture for k-mer and coverage features
- **FusionLayer**: Cross-attention mechanism between k-mer and coverage modalities
- **BarlowTwinsLoss**: Self-supervised learning loss function based on cross-correlation
- **SequenceDataset**: Handles positive pair generation from same-contig fragments

#### Feature Processing (remag/features.py)
- **4-mer composition**: Normalized k-mer frequencies with reverse complement handling
- **Coverage profiles**: Calculated from BAM files or TSV input
- **Fragment augmentation**: Multiple random fragments per contig for training

#### Clustering (remag/clustering.py)
- **HDBSCAN**: Main clustering algorithm for eukaryotic contigs
- **Chimera detection**: Analyzes large contigs for chimeric sequences

#### 4CAC Classifier (remag/xgbclass/)
- **XGBoost models**: Pre-trained classifiers for bacterial sequence detection
- **Multi-scale classification**: Different models for different contig lengths
- **K-mer feature extraction**: 3-7mer frequencies for classification

### Data Flow

1. **Input**: FASTA contigs + BAM/TSV coverage data
2. **Filtering**: 4CAC classifier removes bacterial contigs
3. **Fragmentation**: Large contigs split into overlapping fragments
4. **Feature Matrix**: Combined k-mer + coverage features
5. **Training**: Siamese network learns contig representations
6. **Embeddings**: Generated for original contigs only
7. **Clustering**: HDBSCAN on embeddings
8. **Refinement**: Core gene analysis and contamination removal
9. **Output**: Binned FASTA files and cluster assignments

### Configuration

The tool uses rich-click for CLI with organized option groups:
- **Input/Output**: FASTA, BAM/TSV, output directory
- **Contrastive Learning**: Epochs, batch size, embedding dimensions
- **Clustering**: HDBSCAN parameters
- **Filtering**: Bacterial filter, refinement options

### Important Implementation Details

- **Dual-encoder architecture**: Separate encoders for k-mer (136 features) and coverage features
- **Fragment naming**: Uses `.original`, `.hN.M` patterns for augmented fragments
- **Positive pairs**: Generated from fragments of the same original contig
- **Device handling**: Supports CUDA, MPS, and CPU with appropriate optimizations
- **Memory management**: Batch processing and caching for large datasets
- **Cross-attention fusion**: Multi-head attention between k-mer and coverage modalities

### Pre-trained Models

- **4CAC classifiers**: Located in `remag/xgbclass/models/` for different scales
- **Eukaryotic database**: Core genes in `remag/db/` for quality assessment
- **Model persistence**: Trained Siamese models saved to `siamese_model.pt`

### Testing Strategy

- **Unit tests**: Individual module functionality (not yet implemented)
- **Integration tests**: End-to-end pipeline validation (marked with `@pytest.mark.integration`)
- **Slow tests**: Full training/clustering workflows (marked with `@pytest.mark.slow`)

### Project Structure

- **Entry points**: CLI via `remag.cli:main`, script via `remag.py`, module via `__main__.py`
- **Package data**: XGBoost models in `xgbclass/models/`, eukaryotic databases in `db/`
- **Dependencies**: Core scientific stack (numpy, pandas, torch, hdbscan, xgboost)
- **Optional GPU**: RAPIDS cuML/cuDF for accelerated clustering (install with `[gpu]` extra)