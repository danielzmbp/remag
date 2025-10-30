# REMAG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16443991.svg)](https://doi.org/10.5281/zenodo.16443991)

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Quick Start

### Option 1: Using Conda (Recommended - handles all dependencies)
```bash
# Create environment and install everything
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag

# Run REMAG (output directory optional - defaults to remag_output)
remag contigs.fasta -c alignments.bam
```

### Option 2: Using Docker (No local installation needed)
```bash
docker run --rm -v $(pwd):/data danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

### Option 3: Using pip
```bash
# Create environment first
conda create -n remag python=3.9
conda activate remag

# Install dependencies and REMAG
conda install -c bioconda miniprot
pip install remag

# Run REMAG
remag contigs.fasta -c alignments.bam
```

## Installation

### Recommended: Conda Installation

This is the easiest method as conda handles all dependencies automatically:

```bash
# Create a new environment with all dependencies
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag

# Verify installation
remag --help
```

Note: `miniprot` is pulled in automatically as a dependency of the conda package; no separate installation is required when installing `remag` via conda.

### Alternative: PyPI Installation

If you prefer pip, you'll need to install the external dependency separately:

```bash
# Step 1: Create and activate environment
conda create -n remag python=3.9
conda activate remag

# Step 2: Install external dependency
conda install -c bioconda miniprot

# Step 3: Install REMAG from PyPI
pip install remag
```

### Advanced Conda Setup

For additional features:

```bash
# Basic installation
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag

# Add optional plotting capabilities
conda install -c conda-forge matplotlib umap-learn
```

### Using Docker

```bash
# Pull and run the latest version (output directory defaults to remag_output)
docker run --rm -v $(pwd):/data danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam

# Or specify output directory
docker run --rm -v $(pwd):/data danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output

# For interactive use
docker run -it --rm -v $(pwd):/data danielzmbp/remag:latest /bin/bash
```

### Using Singularity

```bash
# Pull and run the latest version directly
singularity run docker://danielzmbp/remag:latest \
  contigs.fasta -c alignments.bam

# Build Singularity image from Docker Hub
singularity build remag_v0.3.2.sif docker://danielzmbp/remag:v0.3.2

# Or build latest version
singularity build remag_latest.sif docker://danielzmbp/remag:latest

# Run with Singularity
singularity run --bind $(pwd):/data remag_v0.3.2.sif \
  /data/contigs.fasta -c /data/alignments.bam

# Or use exec for direct command execution
singularity exec --bind $(pwd):/data remag_v0.3.2.sif \
  remag /data/contigs.fasta -c /data/alignments.bam -o /data/output

# For interactive shell
singularity shell --bind $(pwd):/data remag_v0.3.2.sif

# Build a local Singularity image file (optional)
singularity build remag.sif docker://danielzmbp/remag:latest
singularity run remag.sif contigs.fasta -c alignments.bam
```

### From source

```bash
# Create and activate conda environment
conda create -n remag python=3.9
conda activate remag

# Clone and install
git clone https://github.com/danielzmbp/remag.git
cd remag
pip install .
```

### Development installation

For contributors and developers:

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Optional Features Installation

For visualization capabilities:

```bash
# Install with plotting dependencies
pip install "remag[plotting]"
```


## Usage

### Command line interface

After installation, you can use REMAG via the command line:

```bash
# Basic usage (output defaults to remag_output in FASTA directory)
remag contigs.fasta -c alignments.bam

# With explicit output directory
remag contigs.fasta -c alignments.bam -o output_directory

# Multiple samples using glob patterns
remag contigs.fasta -c "samples/*.bam"

# Using explicit -f flag (both styles work)
remag -f contigs.fasta -c alignments.bam

# Keep intermediate files with -k shorthand
remag contigs.fasta -c alignments.bam -k
```

### Python module mode

```bash
python -m remag contigs.fasta -c alignments.bam
```

### Getting help

```bash
# Quick reference (basic options)
remag -h

# Full documentation (all advanced options)
remag --help
```

## How REMAG Works

REMAG uses a sophisticated multi-stage pipeline specifically designed for eukaryotic genome recovery:

1. **Eukaryotic Filtering**: By default, REMAG automatically filters for eukaryotic contigs using the integrated HyenaDNA LLM-based classifier (can be disabled with `--skip-bacterial-filter`)
2. **Feature Extraction**: Combines k-mer composition (4-mers) with coverage profiles across multiple samples. Large contigs are split into overlapping fragments for augmentation during training
3. **Contrastive Learning**: Trains a Siamese neural network using the Barlow Twins self-supervised loss function. This creates embeddings where fragments from the same contig are close together
4. **Adaptive Resolution**: Automatically determines optimal Leiden clustering resolution by estimating organism count from core gene analysis, then testing multiple resolutions and selecting the one that maximizes bin quality (completeness - 5 × contamination)
5. **Clustering**: Graph-based Leiden clustering on the learned contig embeddings to form bins
6. **Quality Assessment**: Uses miniprot to align bins against a database of eukaryotic core genes to detect contamination
7. **Iterative Refinement**: Automatically splits contaminated bins based on core gene duplications, then tests lower resolutions to find the most conservative solution

## Key Features

- **Automatic Eukaryotic Filtering**: The HyenaDNA classifier uses a pre-trained genomic foundation model to identify and retain eukaryotic sequences
- **Multi-Sample Support**: Can process coverage information from multiple samples (BAM/CRAM files) simultaneously
- **Adaptive Resolution**: Automatically determines optimal clustering resolution based on bin completeness and contamination
- **Barlow Twins Loss**: Uses a self-supervised contrastive learning approach that doesn't require negative pairs
- **Fragment Augmentation**: Large contigs are split into multiple overlapping fragments during training to improve representation learning
- **Conservative Refinement**: After successful bin refinement, tests lower resolutions to find the most consolidated solution that maintains quality

## Options

Use `remag -h` for quick reference or `remag --help` for full documentation.

### Essential Options

```
  FASTA_ARG                       Input FASTA file (positional argument). Can also use -f/--fasta
  -f, --fasta PATH                Input FASTA file with contigs to bin. Can be gzipped.
  -c, --coverage PATH             Coverage files for calculation. Supports BAM, CRAM (indexed), and TSV formats.
                                  Auto-detects format by extension. Supports space-separated paths and glob patterns
                                  (e.g., "*.bam", "*.cram", "*.tsv"). Use quotes around glob patterns.
  -o, --output PATH               Output directory for results. [default: remag_output in FASTA directory]
  -t, --threads INTEGER           Number of CPU cores to use for parallel processing.  [default: 8]
  -v, --verbose                   Enable verbose logging.
  -k, --keep-intermediate         Keep intermediate files (embeddings, features, model, etc.).
  -h, --help                      Show quick reference or full help.
```

### Advanced Options

For complete list of advanced options (neural network parameters, clustering settings, refinement options, etc.), run:
```bash
remag --help
```

## Output

REMAG produces several output files:

### Core output files (always created):
- `bins/`: Directory containing FASTA files for each bin
- `bins.csv`: Final contig-to-bin assignments
- `embeddings.csv`: Contig embeddings from the neural network
- `remag.log`: Detailed log file
- `*_eukaryotic_filtered.fasta`: Filtered FASTA file with only eukaryotic contigs retained (when eukaryotic filtering is enabled)

### Additional files (with `-k` / `--keep-intermediate` option):
- `siamese_model.pt`: Trained Siamese neural network model
- `kmer_embeddings.csv`: K-mer encoder embeddings (before fusion)
- `coverage_embeddings.csv`: Coverage encoder embeddings (before fusion)
- `params.json`: Complete run parameters for reproducibility
- `features.csv`: Extracted k-mer and coverage features
- `fragments.pkl`: Fragment information used during training
- `hyenadna_classification_results.csv`: HyenaDNA eukaryotic classification results
- `organism_estimation_gene_counts.json`: Gene counts used for adaptive resolution determination
- `refinement_summary.json`: Summary of the bin refinement process
- `gene_contig_mappings.json`: Cached gene-to-contig mappings for faster refinement
- `core_gene_duplication_results.json`: Core gene duplication analysis from refinement
- `temp_miniprot/`: Temporary directory for miniprot alignments (removed unless --keep-intermediate)

### Visualization (optional, requires plotting dependencies):
To generate UMAP visualization plots:

```bash
# Install plotting dependencies if not already installed
pip install remag[plotting]

# Generate UMAP visualization from embeddings
python scripts/plot_features.py --features output_directory/embeddings.csv --clusters output_directory/bins.csv --output output_directory
```

This creates:
- `umap_coordinates.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot with cluster assignments


## Requirements

### Core dependencies (always installed):
- Python 3.9+
- PyTorch (≥1.11.0)
- einops (≥0.6.0) - for HyenaDNA model operations
- scikit-learn (≥1.0.0)
- leidenalg (≥0.9.0) - for graph-based clustering
- igraph (≥0.10.0) - for graph construction in Leiden clustering
- pandas (≥1.3.0)
- numpy (≥1.21.0)
- pysam (≥0.18.0)
- loguru (≥0.6.0)
- tqdm (≥4.62.0)
- rich-click (≥1.5.0)

### External dependencies (must be installed separately):
- **miniprot** - Required for core gene analysis and quality assessment
  - Install with: `conda install -c bioconda miniprot`

### Optional dependencies:
- **For visualization**: matplotlib (≥3.5.0), umap-learn (≥0.5.0)
  - Install with: `pip install remag[plotting]`

The package includes a pre-trained HyenaDNA classifier model for eukaryotic contig filtering. The HyenaDNA model is a genomic foundation model based on the Hyena operator architecture.

## Acknowledgments

The integrated HyenaDNA classifier uses a pre-trained genomic foundation model:

- **Repository**: [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)
- **Paper**: Nguyen E, Poli M, Faizi M, et al. HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. NeurIPS 2023.
   

## License

MIT License - see LICENSE file for details.

## Citation

If you use REMAG in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16443991.svg)](https://doi.org/10.5281/zenodo.16443991)

```bibtex
@software{gomez_perez_2025_remag,
  author       = {Gómez-Pérez, Daniel},
  title        = {REMAG: Recovering high-quality Eukaryotic genomes from complex metagenomes},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16443991},
  url          = {https://doi.org/10.5281/zenodo.16443991}
}
```

Note: The DOI 10.5281/zenodo.16443991 represents all versions and will always resolve to the latest release. A manuscript describing REMAG is in preparation.
