# REMAG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16443991.svg)](https://doi.org/10.5281/zenodo.16443991)

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Quick Start

```bash
# Install from PyPI
pip install remag

# Run REMAG
remag -f contigs.fasta -b alignments.bam -o output_directory
```

## Installation

### From PyPI (recommended)

```bash
# Create conda environment (optional but recommended)
conda create -n remag python=3.9
conda activate remag

# Install from PyPI
pip install remag
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

### GPU-accelerated installation

For GPU-accelerated clustering (requires NVIDIA GPU):

```bash
# Install with RAPIDS support
pip install "remag[gpu]"
```

## Usage

### Command line interface

After installation, you can use REMAG via the command line:

```bash
remag -f contigs.fasta -b alignments.bam -o output_directory
```

### Python module mode

```bash
python -m remag -f contigs.fasta -b alignments.bam -o output_directory
```

## How REMAG Works

REMAG uses a sophisticated multi-stage pipeline specifically designed for eukaryotic genome recovery:

1. **Bacterial Pre-filtering**: By default, REMAG automatically filters out bacterial contigs using the integrated 4CAC classifier (can be disabled with `--skip-bacterial-filter`)
2. **Feature Extraction**: Combines k-mer composition (4-mers) with coverage profiles across multiple samples. Large contigs are split into overlapping fragments for augmentation during training
3. **Contrastive Learning**: Trains a Siamese neural network using the Barlow Twins self-supervised loss function. This creates embeddings where fragments from the same contig are close together
4. **HDBSCAN Clustering**: Density-based clustering on the learned contig embeddings to form bins
5. **Quality Assessment**: Uses miniprot to align bins against a database of eukaryotic core genes to detect contamination
6. **Iterative Refinement**: Automatically splits contaminated bins based on core gene duplications to improve bin quality

## Key Features

- **Automatic Bacterial Filtering**: The 4CAC classifier automatically identifies and removes bacterial sequences before binning
- **Multi-Sample Support**: Can process coverage information from multiple samples (BAM files) simultaneously
- **Barlow Twins Loss**: Uses a self-supervised contrastive learning approach that doesn't require negative pairs
- **Fragment Augmentation**: Large contigs are split into multiple overlapping fragments during training to improve representation learning

## Options

```
  -f, --fasta PATH                Input FASTA file with contigs to bin. Can be gzipped.  [required]
  -b, --bam PATH                  Input BAM file(s) for coverage calculation. Must be indexed. Each BAM represents a sample. Supports space-separated files or glob patterns (e.g., "*.bam", "sample_*.bam"). Use quotes around glob patterns.
  -t, --tsv PATH                  Input TSV file(s) with coverage information.
  -o, --output PATH               Output directory for results.  [required]
  --epochs INTEGER RANGE          Training epochs for neural network.  [default: 400; 50<=x<=2000]
  --batch-size INTEGER RANGE      Batch size for training.  [default: 2048; 64<=x<=8192]
  --embedding-dim INTEGER RANGE   Embedding dimension for contrastive learning.  [default: 256; 64<=x<=512]
  --base-learning-rate FLOAT RANGE
                                  Base learning rate for optimizer.  [default: 0.008; 0.00001<=x<=0.1]
  --min-cluster-size INTEGER RANGE
                                  Minimum fragments per cluster.  [default: 2; 2<=x<=100]
  --min-samples INTEGER RANGE     Minimum samples for HDBSCAN core points.  [default: None; 1<=x<=100]
  --cluster-selection-epsilon FLOAT RANGE
                                  Epsilon for HDBSCAN cluster selection.  [default: 0.0; 0.0<=x<=1.0]
  --min-contig-length INTEGER RANGE
                                  Minimum contig length in bp.  [default: 1000; 500<=x<=10000]
  --max-positive-pairs INTEGER RANGE
                                  Maximum positive pairs for contrastive learning.  [default: 5000000; 100000<=x<=10000000]
  -c, --cores INTEGER RANGE       Number of CPU cores.  [default: 8; 1<=x<=64]
  --min-bin-size INTEGER RANGE    Minimum bin size in bp.  [default: 100000; 50000<=x<=10000000]
  -v, --verbose                   Enable verbose logging.
  --skip-bacterial-filter         Skip bacterial contig filtering (4CAC classifier + contrastive learning).
  --skip-refinement               Skip bin refinement.
  --skip-kmeans-filtering         Skip K-means filtering on embeddings.
  --max-refinement-rounds INTEGER RANGE
                                  Maximum refinement rounds.  [default: 2; 1<=x<=10]
  --num-augmentations INTEGER RANGE
                                  Number of random fragments per contig.  [default: 8; 1<=x<=32]
  --keep-intermediate             Keep intermediate files (training fragments, etc.).
  -h, --help                      Show this message and exit.
```

## Output

REMAG produces several output files:

### Core output files (always created):
- `bins/`: Directory containing FASTA files for each bin
- `bins.csv`: Final contig-to-bin assignments
- `remag.log`: Detailed log file
- `*_non_bacterial_filtered.fasta`: Filtered FASTA file with bacterial contigs removed (when bacterial filtering is enabled)

### Additional files (with `--keep-intermediate` option):
- `embeddings.csv`: Contig embeddings from the neural network
- `umap_embeddings.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot with cluster assignments
- `siamese_model.pt`: Trained Siamese neural network model
- `params.json`: Complete run parameters for reproducibility
- `features.csv`: Extracted k-mer and coverage features
- `fragments.pkl`: Fragment information used during training
- `classification_results.csv`: 4CAC bacterial classification results
- `refinement_summary.json`: Summary of the bin refinement process
- `kmeans_filtering_stats.json`: Statistics from k-means pre-filtering (if enabled)
- `core_gene_duplication_results.json`: Core gene duplication analysis from refinement
- `temp_miniprot/`: Temporary directory for miniprot alignments (removed unless --keep-intermediate)


## Requirements

- Python 3.8+
- PyTorch (≥1.11.0)
- scikit-learn (≥1.0.0)
- XGBoost (≥1.6.0) - for 4CAC classifier
- HDBSCAN (≥0.8.28)
- UMAP (≥0.5.0)
- pandas (≥1.3.0)
- numpy (≥1.21.0)
- matplotlib (≥3.5.0)
- pysam (≥0.18.0)
- loguru (≥0.6.0)
- tqdm (≥4.62.0)
- rich-click (≥1.5.0)
- joblib (≥1.1.0)

The package includes a pre-trained 4CAC classifier model for bacterial contig filtering. The 4CAC classifier code and models are adapted from the [Shamir-Lab/4CAC repository](https://github.com/Shamir-Lab/4CAC).

## Acknowledgments

The integrated 4CAC classifier (`xgbclass` module) is adapted from the work by Shamir Lab:

- **Repository**: [Shamir-Lab/4CAC](https://github.com/Shamir-Lab/4CAC)
- **Paper**: Pu L, Shamir R. 4CAC: 4-class classifier of metagenome contigs using machine learning and assembly graphs. Nucleic Acids Res. 2024;52(19):e94–e94.
   

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
