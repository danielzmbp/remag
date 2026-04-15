# REMAG

[![Preprint DOI](https://img.shields.io/badge/Preprint%20DOI-10.64898%2F2026.03.05.709928-blue)](https://doi.org/10.64898/2026.03.05.709928)

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Quick Start

### Option 1: Using Conda (recommended)
```bash
# Create environment and install REMAG with its external dependency
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag

# Run REMAG
remag contigs.fasta -c alignments.bam
```

### Option 2: Using Docker
```bash
docker run --rm -v $(pwd):/data danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

### Option 3: Using pip
```bash
# Create environment first
conda create -n remag python=3.9
conda activate remag

# Install the external dependency, then REMAG
conda install -c bioconda miniprot
pip install remag

remag contigs.fasta -c alignments.bam
```

## Installation

### Conda

This is the easiest installation path because the conda package pulls in `miniprot` automatically.

```bash
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag
remag --help
```

### PyPI

If you install from PyPI, install `miniprot` separately first:

```bash
conda create -n remag python=3.9
conda activate remag
conda install -c bioconda miniprot
pip install remag
```

### Optional plotting dependencies

```bash
conda install -c conda-forge matplotlib umap-learn
```

### GPU acceleration

REMAG uses PyTorch and will use GPU acceleration automatically when a supported backend is available. No extra REMAG flag is required.

#### Conda with NVIDIA CUDA

If you want a CUDA-enabled PyTorch build, install REMAG first and then replace the CPU PyTorch package with the CUDA-enabled one that matches your system:

```bash
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1
```

Adjust the CUDA version to match your driver and platform.

#### Apple Silicon

On Apple Silicon, PyTorch can use Metal (`mps`) automatically when available. In most cases no extra REMAG-specific setup is needed beyond installing a current PyTorch build.

#### PyPI installs

If you install REMAG with `pip`, install the PyTorch build you want first, then install REMAG:

```bash
conda create -n remag python=3.9
conda activate remag
conda install -c bioconda miniprot

# Install the desired PyTorch build first
pip install torch

# Then install REMAG
pip install remag
```

For NVIDIA systems, use the PyTorch install command from the official PyTorch selector so the wheel matches your CUDA runtime.

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
singularity build remag_v0.3.4.sif docker://danielzmbp/remag:v0.3.4

# Or build latest version
singularity build remag_latest.sif docker://danielzmbp/remag:latest

# Run with Singularity
singularity run --bind $(pwd):/data remag_v0.3.4.sif \
  /data/contigs.fasta -c /data/alignments.bam

# Or use exec for direct command execution
singularity exec --bind $(pwd):/data remag_v0.3.4.sif \
  remag /data/contigs.fasta -c /data/alignments.bam -o /data/output

# For interactive shell
singularity shell --bind $(pwd):/data remag_v0.3.4.sif

# Build a local Singularity image file (optional)
singularity build remag.sif docker://danielzmbp/remag:latest
singularity run remag.sif contigs.fasta -c alignments.bam
```

### From source

```bash
conda create -n remag python=3.9
conda activate remag

git clone https://github.com/danielzmbp/remag.git
cd remag
conda install -c bioconda miniprot
pip install .
```

### Development installation

```bash
pip install -e ".[dev]"
```


## Usage

### Command line interface

After installation, you can use REMAG via the command line:

```bash
# Basic usage (output defaults to remag_output in FASTA directory)
remag contigs.fasta -c alignments.bam

# With explicit output directory
remag contigs.fasta -c alignments.bam -o output_directory

# Multiple samples using repeated flags
remag contigs.fasta -c sample1.bam -c sample2.bam

# Multiple samples using shell-expanded globs
remag contigs.fasta -c samples/*.bam

# Using explicit -f flag (both styles work)
remag -f contigs.fasta -c alignments.bam

# Using precomputed coverage tables (one TSV per sample)
remag contigs.fasta -c sample1.tsv -c sample2.tsv

# Keep intermediate files with -k shorthand
remag contigs.fasta -c alignments.bam -k

# Only run eukaryotic filtering (skip binning)
remag contigs.fasta -c alignments.bam --filter-only

# Use single-cell mode (adjusts k-NN and clustering defaults)
remag contigs.fasta -c alignments.bam -m single-cell
```

### Python module mode

```bash
python -m remag contigs.fasta -c alignments.bam
```

### Coverage TSV format

Precomputed coverage TSVs are supported as an alternative to BAM/CRAM. Use one TSV per sample.

- Column 1: contig ID
- Last column: coverage value for that contig
- No header row

Example:

```tsv
contig_1	12.4
contig_2	3.8
contig_3	0.0
```

TSV input provides contig-level coverage only. REMAG cannot infer fragment-specific coverage for augmented fragments from a TSV, so every fragment from the same contig gets the same coverage value. Use BAM/CRAM if you want fragment-level augmented coverage features. Do not mix TSV inputs with BAM/CRAM inputs in the same run.

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
4. **Eukaryotic Gene Marker Annotation**: Uses miniprot to annotate contigs with eukaryotic single-copy core genes, providing the quality metrics needed for clustering decisions
5. **Greedy Clustering**: Iteratively extracts bins using a greedy Leiden approach -- at each step, tests multiple Leiden resolutions on the remaining contigs, selects the single best-quality cluster (by F1 score of completeness vs. contamination), removes it from the graph, and repeats
6. **Bin Rescue**: Merges fragmented bins into larger bins based on embedding similarity and single-copy gene safety, and rescues unbinned contigs into matching bins

## Key Features

- **Automatic Eukaryotic Filtering**: The HyenaDNA classifier uses a pre-trained genomic foundation model to identify and retain eukaryotic sequences
- **Multi-Sample Support**: Can process coverage information from multiple samples (BAM/CRAM files) simultaneously
- **Greedy Multi-Resolution Clustering**: Iteratively extracts bins by testing multiple Leiden resolutions at each step, allowing different bins to use different resolutions for optimal quality
- **Barlow Twins Loss**: Uses a self-supervised contrastive learning approach that doesn't require negative pairs
- **Fragment Augmentation**: Large contigs are split into multiple overlapping fragments during training to improve representation learning
- **Bin Rescue**: Merges fragmented bins and rescues unbinned contigs into existing bins based on embedding similarity and single-copy gene safety

## Options

Use `remag -h` for a quick reference or `remag --help` for the full CLI documentation.

Commonly used options:

- `-c, --coverage`: one or more BAM, CRAM, or TSV coverage inputs
- `-o, --output`: output directory; defaults to `remag_output` next to the input FASTA
- `-k, --keep-intermediate`: retain embeddings, features, model weights, and other intermediate files
- `--filter-only`: stop after eukaryotic filtering and write filtered FASTA output
- `-m, --mode`: select presets such as `metagenomics` or `single-cell`
- `--save-filtered-contigs`: also write the contigs removed by the eukaryotic filter

For the complete list of neural-network, clustering, filtering, and rescue options, run:

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
- `*_eukaryotic_filtered.fasta`: Filtered FASTA file with only eukaryotic contigs retained when eukaryotic filtering is enabled

### Additional files (with `-k` / `--keep-intermediate` option):
- `siamese_model.pt`: Trained Siamese neural network model
- `kmer_embeddings.csv`: K-mer encoder embeddings (before fusion)
- `coverage_embeddings.csv`: Coverage encoder embeddings (before fusion)
- `params.json`: Complete run parameters for reproducibility
- `features.csv`: Extracted k-mer and coverage features
- `fragments.pkl`: Fragment information used during training
- `*_hyenadna_classification.tsv`: HyenaDNA eukaryotic classification results (tab-separated)
- `gene_contig_mappings.json`: Cached gene-to-contig mappings for faster processing
- `core_gene_duplication_results.json`: Core gene duplication analysis
- `knn_graph_edges.csv`: k-NN graph edge list used for Leiden clustering
- `knn_graph_stats.json`: k-NN graph construction statistics
- `temp_miniprot/`: Temporary directory for miniprot alignments (removed unless --keep-intermediate)

### Additional filtering output:
- `*_non_eukaryotic.fasta`: Contigs removed by the HyenaDNA filter when `--save-filtered-contigs` is used

### Visualization (optional, requires plotting dependencies):
To generate UMAP visualization plots:

```bash
# Install plotting dependencies if not already installed
pip install "remag[plotting]"

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
- scipy (≥1.6.0)
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

```bibtex
@article {G{\'o}mez-P{\'e}rez2026.03.05.709928,
	author = {G{\'o}mez-P{\'e}rez, Daniel and Raguideau, S{\'e}bastien and Warring, Sally and James, Robert and Hildebrand, Falk and Quince, Christopher},
	title = {REMAG: recovery of eukaryotic genomes from metagenomic data using contrastive learning},
	elocation-id = {2026.03.05.709928},
	year = {2026},
	doi = {10.64898/2026.03.05.709928},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/03/08/2026.03.05.709928},
	eprint = {https://www.biorxiv.org/content/early/2026/03/08/2026.03.05.709928.full.pdf},
	journal = {bioRxiv}
}
```
