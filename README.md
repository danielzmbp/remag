# REMAG

[![Preprint DOI](https://img.shields.io/badge/Preprint%20DOI-10.64898%2F2026.03.05.709928-blue)](https://doi.org/10.64898/2026.03.05.709928)

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Installation

### Conda (recommended)

```bash
conda create -n remag -c bioconda -c conda-forge remag
conda activate remag
```

### PyPI

Install `miniprot` separately first:

```bash
conda create -n remag python=3.9
conda activate remag
conda install -c bioconda miniprot
pip install remag
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

### Docker

```bash
docker pull danielzmbp/remag:latest
```

### Optional plotting dependencies

```bash
pip install remag[plotting]
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

## Quick Start

### Conda

```bash
remag contigs.fasta -c alignments.bam
```

### Docker

```bash
docker run --rm -v $(pwd):/data danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

### Singularity

```bash
singularity build remag.sif docker://danielzmbp/remag:latest
singularity run --bind $(pwd):/data remag.sif \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```


## Usage

### Command line interface

After installation, you can use REMAG via the command line:

```bash
# Basic usage
remag contigs.fasta -c alignments.bam

# With explicit output directory
remag contigs.fasta -c alignments.bam -o output_directory

# Multiple samples
remag contigs.fasta -c sample1.bam -c sample2.bam

# Multiple samples using shell-expanded globs
remag contigs.fasta -c samples/*.bam

# Using precomputed coverage tables (one TSV per sample)
remag contigs.fasta -c sample1.tsv -c sample2.tsv

# Only run eukaryotic filtering (skip binning)
remag contigs.fasta -c alignments.bam --filter-only

# Use single-cell mode (adjusts k-NN defaults and skips eukaryotic filtering)
remag contigs.fasta -c alignments.bam -m single-cell

# Keep intermediate files
remag contigs.fasta -c alignments.bam -k
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

## Common Options

- `-c, --coverage`: one or more BAM, CRAM, or TSV coverage inputs
- `-o, --output`: output directory; defaults to `remag_output` next to the input FASTA
- `-k, --keep-intermediate`: retain embeddings, features, model weights, and other intermediate files
- `--filter-only`: stop after eukaryotic filtering and write filtered FASTA output
- `-m, --mode`: select presets such as `metagenomics` or `single-cell`
- `--save-filtered-contigs`: also write the contigs removed by the eukaryotic filter

Use `remag -h` for a quick reference and `remag --help` for the full CLI, including training, clustering, filtering, and rescue options.

## How It Works

REMAG recovers eukaryotic bins with a multi-stage pipeline:

1. **Eukaryotic filtering**: By default, REMAG filters contigs with the integrated HyenaDNA classifier. This step can be disabled with `--skip-bacterial-filter`.
2. **Feature extraction**: REMAG combines 4-mer composition with optional multi-sample coverage data. Large contigs are augmented into multiple fragments for training.
3. **Contrastive learning**: A Siamese network trained with Barlow Twins learns embeddings that place fragments from the same contig close together.
4. **Core gene annotation**: `miniprot` maps eukaryotic single-copy core genes to support clustering and quality checks.
5. **Greedy clustering and rescue**: REMAG applies greedy Leiden clustering across multiple resolutions, then merges or rescues bins when single-copy gene checks support it.

## Output

### Core outputs

- `bins/`: Directory containing FASTA files for each bin
- `bins.csv`: Final contig-to-bin assignments
- `embeddings.csv`: Contig embeddings from the neural network
- `remag.log`: Detailed log file
- `*_eukaryotic_filtered.fasta`: Filtered FASTA written when eukaryotic filtering is enabled

### Additional outputs with `-k` / `--keep-intermediate`

- `siamese_model.pt`: Trained Siamese neural network model
- `kmer_embeddings.csv`: K-mer encoder embeddings (before fusion)
- `coverage_embeddings.csv`: Coverage encoder embeddings (before fusion)
- `params.json`: Run parameters for reproducibility
- `features.csv`: Extracted k-mer and coverage features
- `fragments.pkl`: Fragment information used during training
- `*_hyenadna_classification.tsv`: HyenaDNA eukaryotic classification results (tab-separated)
- `gene_contig_mappings.json`: Cached gene-to-contig mappings
- `core_gene_duplication_results.json`: Core gene duplication analysis
- `knn_graph_edges.csv`: k-NN graph edge list used for Leiden clustering
- `knn_graph_stats.json`: k-NN graph construction statistics
- `temp_miniprot/`: Temporary directory for miniprot alignments

### Filtering output

- `*_non_eukaryotic.fasta`: Contigs removed by the HyenaDNA filter when `--save-filtered-contigs` is used

### Visualization

With plotting dependencies installed, you can generate UMAP plots from `embeddings.csv` and `bins.csv`:

```bash
pip install "remag[plotting]"
python scripts/plot_features.py --features output_directory/embeddings.csv --clusters output_directory/bins.csv --output output_directory
```

- `umap_coordinates.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot with cluster assignments


## Requirements

- Python 3.9+
- `miniprot` is required for core gene analysis when installing outside conda packages or the project Docker image
- Plotting extras are optional: `pip install remag[plotting]`

The package includes a pre-trained HyenaDNA classifier model for eukaryotic contig filtering.

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
