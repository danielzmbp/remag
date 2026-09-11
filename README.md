# REMAG

[![Preprint DOI](https://img.shields.io/badge/Preprint%20DOI-10.64898%2F2026.03.05.709928-blue)](https://doi.org/10.64898/2026.03.05.709928)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16762340.svg)](https://doi.org/10.5281/zenodo.16762340)

**R**ecovery of **E**ukaryotic **M**etagenome-**A**ssembled **G**enomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Index

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Common Options](#common-options)
- [How It Works](#how-it-works)
- [Output](#output)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citation](#citation)

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
pip install "remag[plotting]"
```

### GPU acceleration

REMAG automatically uses NVIDIA CUDA or Apple Silicon Metal (`mps`) when available, otherwise CPU. No extra flag is required.

#### NVIDIA GPUs (Linux)

Install CUDA-enabled PyTorch before REMAG. This example uses CUDA 12.8; use the [PyTorch selector](https://pytorch.org/get-started/locally/) to choose a build matching your GPU and driver.

```bash
conda create -n remag-gpu -c conda-forge -c bioconda python=3.11 pip miniprot
conda activate remag-gpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install remag
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### Apple Silicon (macOS)

In your native Apple Silicon REMAG environment:

```bash
python -m pip install torch remag
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

The relevant GPU check should report `True`.

## Quick Start

### Conda

```bash
remag contigs.fasta -c alignments.bam
```

### Docker

```bash
docker run --rm -v "$(pwd):/data" danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

### Singularity

```bash
singularity build remag.sif docker://danielzmbp/remag:latest
singularity run --bind "$(pwd):/data" remag.sif \
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

# Using interval coverage files (one COV/bedGraph per sample)
remag contigs.fasta -c sample1.bam.cov.gz -c sample2.bam.cov.gz

# Only run eukaryotic filtering (skip binning)
remag contigs.fasta --filter-only

# Keep intermediate files
remag contigs.fasta -c alignments.bam -k

# Recompute results in an existing output directory
remag contigs.fasta -c alignments.bam -o output_directory --force
```

During feature generation, REMAG rejects repeated identifiers among retained contigs, using the first whitespace-separated token in each FASTA header.

Coverage is optional. To use sequence composition alone:

```bash
remag contigs.fasta -o output_directory
```

### Reusing or replacing results

REMAG reuses available results in the output directory. When existing outputs are found, it prints:

```text
Existing REMAG results found. Reusing available outputs. Use --force to recompute.
```

Use `--force` to remove recognized REMAG outputs and rerun the requested workflow. This clears cached features, model weights, embeddings, gene mappings, previous bins, filtering outputs, temporary miniprot directories, and logs. Other files in the directory are left in place. If a supplied input would be removed, REMAG stops before deleting anything.

Use `--force` or a different output directory when changing inputs or analysis settings. `--force --filter-only` clears previous results and runs filtering only. Saved `umap_coordinates.csv` and `umap_plot.pdf` are also cleared; regenerate plots after the run.

### Python module mode

```bash
python -m remag contigs.fasta -c alignments.bam
```

### Precomputed Coverage Formats

Precomputed coverage files are supported as an alternative to BAM/CRAM. Use one file per sample. REMAG auto-detects the precomputed coverage layout from the file contents.

#### Contig-level TSV/TXT

Use this format when you have one average coverage value per contig.

- Column 1: contig ID
- Last column: coverage value for that contig
- No header row

Example:

```tsv
contig_1	12.4
contig_2	3.8
contig_3	0.0
```

Contig-level TSV/TXT input provides contig-level coverage only. REMAG cannot infer fragment-specific coverage for augmented fragments from this layout, so every fragment from the same contig gets the same coverage value.

#### Interval COV/bedGraph

Use this format when you have per-interval coverage, such as output with run-length encoded coverage blocks.

- Column 1: contig ID
- Column 2: interval start, 0-based inclusive
- Column 3: interval end, 0-based exclusive
- Column 4: coverage value for that interval
- No header row

Example:

```tsv
ctg0	0	256	1
ctg0	256	859	2
ctg0	859	861	1
ctg1	0	73	6
```

Supported extensions include `.cov`, `.cov.gz`, `.bedgraph`, `.bedgraph.gz`, `.bg`, and `.bg.gz`. Missing intervals are treated as zero coverage. Interval coverage is used to compute fragment-specific mean and standard deviation coverage, so it supports REMAG's random augmentations.

Do not mix BAM/CRAM inputs with precomputed coverage inputs in the same run.

## Common Options

- `-c, --coverage`: one or more BAM, CRAM, contig-level TSV/TXT, or interval COV/bedGraph coverage inputs
- `-o, --output`: output directory; defaults to `remag_output` next to the input FASTA
- `-k, --keep-intermediate`: retain features, model weights, encoder embeddings, and other optional intermediate files; `embeddings.csv` is saved without this flag
- `--force`: remove existing REMAG outputs and recompute results
- `--filter-only`: stop after eukaryotic filtering and write filtered FASTA output
- `--save-filtered-contigs`: also write classified contigs rejected by the eukaryotic filter
- `--skip-bacterial-filter`: disable the HyenaDNA filter
- `--min-bin-size`: minimum bin size written to FASTA; defaults to 500,000 bp

Use `remag -h` for a quick reference and `remag --help` for the full CLI, including training, clustering, filtering, and rescue options.

The default minimum contig length is 1,000 bp with zero or one coverage file and 4,096 bp with multiple files. The k-NN graph uses 15 neighbors, and HyenaDNA filtering is enabled. Use `--min-contig-length`, `--leiden-k-neighbors`, and `--skip-bacterial-filter` to adjust these settings explicitly. Multiple coverage files lower the default base learning rate from `0.005` to `0.0005`; an explicitly supplied learning rate is preserved.

## How It Works

REMAG recovers eukaryotic bins with a multi-stage pipeline:

1. **Eukaryotic filtering**: By default, REMAG filters contigs with the integrated HyenaDNA classifier. This step can be disabled with `--skip-bacterial-filter`.
2. **Feature extraction**: REMAG combines 4-mer composition with optional multi-sample coverage data. Contigs are augmented into fragments for training when their lengths permit; contigs longer than 50 kb receive augmentations from each half.
3. **Contrastive learning**: A Siamese network trained with Barlow Twins learns embeddings that place fragments from the same contig close together.
4. **Core gene annotation**: `miniprot` maps eukaryotic single-copy core genes to support clustering and quality checks.
5. **Greedy clustering and rescue**: REMAG applies greedy Leiden clustering across multiple resolutions, then merges or rescues bins when single-copy gene checks support it.

## Output

### Binning outputs

- `bins/`: FASTA files for bins meeting `--min-bin-size`
- `bins.csv`: Final assignments for contigs in saved bins; excludes noise and bins below the minimum size
- `embeddings.csv`: Embeddings for original contigs, including those that do not enter a saved bin
- `fragments.pkl`: Fragment sequences and coordinates used by the pipeline; currently written even without `-k`
- `remag.log`: Detailed log file
- `gene_contig_mappings.json`: Cached gene mappings when miniprot finds accepted matches
- `core_gene_duplication_results.json`: Core gene duplication analysis for the final saved bins, after rescue and minimum-size filtering

### Additional outputs with `-k` / `--keep-intermediate`

- `siamese_model.pt`: Trained Siamese neural network model
- `kmer_embeddings.csv`: K-mer encoder embeddings (before fusion)
- `coverage_embeddings.csv`: Coverage encoder embeddings, when coverage features are present (before fusion)
- `params.json`: Run parameters for reproducibility
- `features.csv`: Extracted k-mer and coverage features
- `knn_graph_edges.csv`: k-NN graph edge list used for Leiden clustering
- `knn_graph_stats.json`: k-NN graph construction statistics
- `temp_gene_mapping/`: Miniprot files used to generate gene mappings
- `temp_miniprot/`: Per-bin miniprot files, when the fallback duplication-check path runs

### Filtering outputs

- `*_hyenadna_classification.tsv`: Predictions for contigs meeting `--min-contig-length`
- `*_eukaryotic_filtered.fasta`: Contigs retained by the HyenaDNA filter
- `*_non_eukaryotic.fasta`: Rejected contigs, when `--save-filtered-contigs` is used and at least one contig is rejected

`--filter-only` stops after this stage and does not produce binning outputs. Use it with filtering enabled; `--skip-bacterial-filter` disables that stage. Contigs below the minimum length are excluded from classification and the rejected-contig FASTA.

If no contigs pass the classifier, the current implementation falls back to the original input FASTA instead of writing a filtered FASTA. Classifier initialization failures also fall back to the input; consult `remag.log` for details.

### Visualization

The plotting helper currently runs from a REMAG source checkout. From the repository root, install the plotting dependencies and generate UMAP plots from `embeddings.csv` and `bins.csv`:

```bash
pip install -e ".[plotting]"
python scripts/plot_features.py --features output_directory/embeddings.csv --clusters output_directory/bins.csv --output output_directory
```

- `umap_coordinates.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot with cluster assignments

## Requirements

- Python 3.9+
- `miniprot` on `PATH` for core gene analysis during binning; `--filter-only` does not require it. The Dockerfile includes miniprot; install it separately for PyPI and source installations.
- Plotting extras are optional: `pip install "remag[plotting]"`

The package includes two pre-trained HyenaDNA classifiers (1,024- and 4,096-base contexts) and the protein reference database used for core gene mapping.

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
