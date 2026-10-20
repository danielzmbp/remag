# REMAG

[![Preprint DOI](https://img.shields.io/badge/Preprint%20DOI-10.64898%2F2026.03.05.709928-blue)](https://doi.org/10.64898/2026.03.05.709928)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16762340.svg)](https://doi.org/10.5281/zenodo.16762340)

**R**ecovery of **E**ukaryotic **M**etagenome-**A**ssembled **G**enomes using contrastive learning. REMAG bins eukaryotic genomes from mixed prokaryotic–eukaryotic metagenomes using sequence composition and optional coverage data.

[Installation](#installation) · [Quick Start](#quick-start) · [Usage](#usage) · [Common Options](#common-options) · [How It Works](#how-it-works) · [Output](#output) · [Troubleshooting](#troubleshooting) · [Citation](#citation)

## Installation

Requires Python 3.9+ and `miniprot` on `PATH` for binning. `--filter-only` does not require miniprot. The package includes the July 2026 HyenaDNA classifier and the protein reference database.

### Conda (recommended)

```bash
conda create -n remag -c conda-forge -c bioconda remag
conda activate remag
```

### PyPI

Install miniprot separately; for example, use a Conda environment:

```bash
conda create -n remag -c conda-forge -c bioconda python=3.11 pip miniprot
conda activate remag
python -m pip install remag
```

### From source

Create an environment with miniprot, then install from the repository:

```bash
conda create -n remag -c conda-forge -c bioconda python=3.11 pip miniprot
conda activate remag
git clone https://github.com/danielzmbp/remag.git
cd remag
python -m pip install .
```

Container commands are in [Quick Start](#quick-start); plotting extras are in [Visualization](#visualization).

### GPU acceleration

Training and embedding generation automatically select NVIDIA CUDA, then Apple Silicon MPS, then CPU. **HyenaDNA filtering uses CUDA or CPU only.** The supplied Dockerfile installs CPU-only PyTorch.

For NVIDIA GPUs on Linux, install CUDA-enabled PyTorch before REMAG. This example uses CUDA 12.8; choose a compatible build with the [PyTorch selector](https://pytorch.org/get-started/locally/).

```bash
conda create -n remag-gpu -c conda-forge -c bioconda python=3.11 pip miniprot
conda activate remag-gpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install remag
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

For Apple Silicon, use a native ARM environment with miniprot installed:

```bash
python -m pip install torch remag
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

The relevant GPU check should report `True`.

## Quick Start

Provide a contig FASTA and coordinate-sorted BAM/CRAM files containing reads mapped to that assembly. Use one alignment file per sample.

```bash
remag contigs.fasta -c alignments.bam -o output_directory
```

Recovered genomes are saved in `output_directory/bins/`, with contig assignments in `output_directory/bins.csv`. Check `output_directory/remag.log` for progress and warnings.

Without `-o`, results go to `remag_output` next to the input FASTA. You can also run REMAG with `python -m remag`.

### Docker

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd):/data" danielzmbp/remag:latest \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

### Singularity

```bash
singularity build remag.sif docker://danielzmbp/remag:latest
singularity run --bind "$(pwd):/data" remag.sif \
  /data/contigs.fasta -c /data/alignments.bam -o /data/output
```

For reproducible runs, record the installed version with `remag --version` and use a versioned container image tag rather than `latest`.

## Usage

```bash
# Multiple samples (or use -c samples/*.bam)
remag contigs.fasta -c sample1.bam -c sample2.bam

# Precomputed coverage: one file per sample
remag contigs.fasta -c sample1.tsv -c sample2.tsv
remag contigs.fasta -c sample1.bam.cov.gz -c sample2.bam.cov.gz

# Only run eukaryotic filtering
remag contigs.fasta --filter-only

# Keep features, model weights, and encoder embeddings
remag contigs.fasta -c alignments.bam -k

# Replace existing results
remag contigs.fasta -c alignments.bam -o output_directory --force
```

Quote paths containing spaces, e.g. `-c "sample 1.bam" -c "sample 2.bam"`. FASTA input may be gzipped. Retained contigs must have unique IDs (the first whitespace-separated token in each header).

### Coverage inputs

Use one file per sample. Do not mix BAM/CRAM with precomputed coverage in the same run.

**BAM/CRAM:** provide coordinate-sorted alignments against the input assembly. REMAG attempts to create missing indexes, so pre-index files if their directory is read-only. Alignment files are processed one at a time with readers controlled by `--threads`. If memory is exhausted, reduce `--threads`; `--coverage-batch-size` can further limit queued work. Memory and worker failures stop the run.

**Contig-level TSV/TXT:** provide whitespace-separated rows without a header: contig ID first, coverage value last. A two-column table avoids ambiguity with interval detection:

```tsv
contig_1	12.4
contig_2	3.8
contig_3	0.0
```

Every augmented fragment inherits its contig's coverage value; this format cannot provide fragment-specific coverage.

**Interval COV/bedGraph:** provide non-overlapping intervals with four columns and no header: contig ID, start (0-based inclusive), end (exclusive), and coverage:

```tsv
ctg0	0	256	1
ctg0	256	859	2
ctg0	859	861	1
ctg1	0	73	6
```

Missing intervals count as zero coverage. REMAG computes fragment-specific means and standard deviations from these intervals.

Accepted precomputed extensions are `.tsv`, `.txt`, `.cov`, `.bedgraph`, and `.bg`, optionally followed by `.gz`. The layout is detected from the contents.

### Reusing or replacing results

REMAG validates saved training and clustering results against recorded settings and input identities. Compatible embeddings can be reused without saved model weights; rescue restarts from pre-rescue assignments. Incompatible or incomplete provenance requires `--force` or a new output directory.

`--force` removes recognized REMAG outputs, including bins, caches, filtering results, logs, temporary miniprot directories, and UMAP plots. Other files are left in place. Cleanup stops before deletion if an input would be removed, `bins/` is a symlink, or a temporary directory contains symlinks. `--force --filter-only` also clears prior binning results.

Cached filtered FASTA files are reused without reclassification, even when the recorded classifier policy differs or is unknown (a warning is logged). Use `--force` or a new output directory when changing inputs, analysis settings, or filtering policy.

## Common Options

Use `remag -h` for a quick reference and `remag --help` for all options.

| Option | Purpose / default |
| --- | --- |
| `-c, --coverage` | One or more coverage files; optional |
| `-o, --output` | Output directory; `remag_output` next to the input FASTA |
| `-t, --threads` | CPU workers; 8 |
| `-k, --keep-intermediate` | Retain features, model weights, and encoder embeddings |
| `--force` | Remove recognized outputs and recompute |
| `--filter-only` | Stop after eukaryotic filtering |
| `--save-filtered-contigs` | Also save classified contigs rejected by the filter |
| `--skip-bacterial-filter` | Disable HyenaDNA filtering |
| `--min-contig-length` | Minimum admitted contig and training-fragment length; 1,000 bp |
| `--graph-min-contig-length` | Minimum graph contig length; 1,000 bp with one coverage file, otherwise 3,000 bp |
| `--min-bin-size` | Minimum bin size exported after rescue; 500,000 bp |
| `--skip-rescue` | Skip both bin merging and unassigned-contig recruitment |

The graph cutoff must be at least the admission cutoff; its default is raised if necessary. Admitted contigs below the graph cutoff share the embedding space and remain eligible for rescue. The graph cutoff does not change filtering or training-fragment lengths.

Other defaults: 100 epochs, 256 embedding dimensions, 15 k-NN neighbors, and enabled HyenaDNA filtering. The base learning rate is `0.005`, or `0.0005` with multiple coverage files; explicit values take precedence. Effective settings are saved in `params.json`.

## How It Works

1. **Filter:** HyenaDNA identifies eukaryotic contigs (unless `--skip-bacterial-filter` is set).
2. **Extract features:** combine 4-mer composition with coverage. Generate training fragments where lengths permit; contigs longer than 50 kb receive augmentations from each half.
3. **Learn embeddings:** train a Siamese network with Barlow Twins on fragments from the same contig.
4. **Annotate markers:** use miniprot to map eukaryotic single-copy core genes for clustering and quality checks.
5. **Cluster and rescue:** run weighted Leiden clustering across multiple resolutions, merge fragmented bins, recruit unassigned contigs, then apply the minimum bin size.

Composition and coverage encoders feed cross-attention and multiscale fusion (`p` is the number of coverage features):

![Composition (136 → 128 → 64) and coverage (p → 64 → 32) feed cross-attention and fusion, producing a D-dimensional embedding (default 256). The embedding feeds graph clustering and a D → 128 → 512 training projection followed by Barlow Twins loss.](https://raw.githubusercontent.com/danielzmbp/remag/main/docs/images/architecture.png)

See the [methods guide](https://github.com/danielzmbp/remag/blob/main/docs/methods.md) for model architecture, clustering, and rescue details.

## Output

### Main results

| File | Contents |
| --- | --- |
| `bins/bin_*.fa` | Bins meeting `--min-bin-size` |
| `bins.csv` | Assignments for contigs in exported bins; excludes noise and smaller bins |
| `remag.log` | Run progress, warnings, and errors |

<details>
<summary>Additional outputs</summary>

All runs retain `params.json`. Matching reruns preserve the original parameter record.

| File | Contents |
| --- | --- |
| `params.json` | Effective run settings |
| `pre_rescue.csv` | Initial assignments for all admitted contigs; those outside the graph start as noise |
| `rescue_assignments.csv` | Assignments before the final size filter, including noise and small bins, with an `exported` flag |
| `embeddings.csv` | Embeddings for all admitted original contigs |
| `training_provenance.json` | Model dimensions, training/filter settings, and identities used to validate reuse |
| `clustering_provenance.json` | Clustering/rescue settings and input fingerprints |
| `knn_graph_edges.csv` | Original graph edges used for clustering and merge validation |
| `knn_graph_contigs.csv` | Ordered graph contig IDs; edge indices refer to these rows, not necessarily embedding rows |
| `knn_graph_stats.json` | Graph statistics and cache fingerprints |
| `gene_contig_mappings.json` | Cached marker mappings and positions |
| `core_gene_duplication_results.json` | Marker statistics for final exported bins, including `within_contig_duplications` |
| `fragments.pkl` | Fragment sequences and coordinates; retained even without `-k` |

Graph files and embeddings are retained without `-k`. Annotation failures stop binning; a successful search with no accepted matches reports zero detected genes. Multiple non-overlapping copies of a marker within one contig appear in the quality report, while clustering and rescue count that family once per contig.

With `-k`, REMAG also retains `features.csv`, `siamese_model.pt` (a PyTorch state dictionary), `kmer_embeddings.csv`, and (when coverage is present) `coverage_embeddings.csv`. Miniprot intermediates are kept in `temp_gene_mapping/` and, when the fallback per-bin check runs, `temp_miniprot/`.

</details>

### Filtering outputs

- `*_hyenadna_classification.tsv`: predictions for successfully classified contigs meeting `--min-contig-length`
- `*_eukaryotic_filtered.fasta`: retained contigs
- `*_non_eukaryotic.fasta`: rejected contigs, when `--save-filtered-contigs` is set and at least one contig is rejected

`--filter-only` stops here. Keep filtering enabled: combining it with `--skip-bacterial-filter` produces no filtered FASTA. Contigs below the minimum length are excluded from classification and the rejected-contig file.

The July 2026 model uses one window for contigs up to 4,096 bp and up to 16 evenly spaced windows for longer contigs. The whole contig is retained when the mean eukaryote probability is **≥0.30**. Tokenization leaves room for up to 4,094 biological bases per window. Original FASTA headers, sequence content, and order are preserved. The `confidence` column is diagnostic; it does not control filtering. See [filter_policy.py](https://github.com/danielzmbp/remag/blob/main/remag/filter_policy.py) for the checkpoint identity and exact settings.

Filtering uses fixed batches of 64 contigs and up to 256 windows per inference call; nondefault `--hyenadna-batch-size` values are ignored with a warning. If no contigs pass, REMAG saves an empty filtered FASTA and finishes without training or binning.

Checkpoint/prediction validation errors stop the run. Other classifier initialization failures fall back to the original FASTA; other batch-classification failures retain the affected contigs without prediction rows. Check `remag.log` for these fallbacks.

### Visualization

From a source checkout, install the plotting extras and generate UMAP plots:

```bash
python -m pip install -e ".[plotting]"
python scripts/plot_features.py \
  --features output_directory/embeddings.csv \
  --clusters output_directory/bins.csv \
  --output output_directory
```

This writes `umap_coordinates.csv` and `umap_plot.pdf`. The helper script is available in the repository, not installed as a CLI command.

## Troubleshooting

Check `remag.log` for the stage reached and any warnings or errors.

| Problem | What to check |
| --- | --- |
| No bins produced | Check whether filtering retained contigs, whether contigs remained unassigned, and whether any bins reached the default 500 kb export threshold (`--min-bin-size`). |
| Too few training pairs | Training requires at least two positive fragment pairs; very small inputs may not generate enough usable fragments. |
| Coverage calculation runs out of memory | Reduce `--threads`; use `--coverage-batch-size` to further limit queued work. |

## Acknowledgments

The classifier builds on [HyenaDNA](https://github.com/HazyResearch/hyena-dna): Nguyen E, Poli M, Faizi M, et al. *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution.* NeurIPS 2023.

## License

[MIT](https://github.com/danielzmbp/remag/blob/main/LICENSE).

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
