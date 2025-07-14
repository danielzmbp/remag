# REMAG

Metagenomic binning using neural networks and contrastive learning.

## Quick Start

```bash
# Create conda environment and install
conda create -n remag python=3.9
conda activate remag
pip install remag

# Run REMAG
remag -f contigs.fasta -b alignments.bam -o output_directory
```

## Installation

### From PyPI (recommended)

First, create and activate a conda environment:

```bash
conda create -n remag python=3.9
conda activate remag
pip install remag
```

### From source

```bash
# Create and activate conda environment
conda create -n remag python=3.9
conda activate remag

# Clone and install
git clone https://github.com/yourusername/remag.git
cd remag
pip install .
```

### Development installation

```bash
# Create and activate conda environment
conda create -n remag python=3.9
conda activate remag

# Clone and install in development mode
git clone https://github.com/yourusername/remag.git
cd remag
pip install -e ".[dev]"
```

### Troubleshooting Installation

If you encounter `ModuleNotFoundError: No module named 'remag.cli'` when running `remag`:

1. **Activate your conda environment**: Make sure you're in the correct environment with `conda activate remag`
2. **Check installation**: Verify REMAG is installed with `pip show remag`
3. **Use module mode**: Try running with explicit Python: `python -m remag` instead of `remag`
4. **Reinstall if needed**: If issues persist, try `pip uninstall remag && pip install remag`

**Note**: Always use a dedicated conda environment to avoid conflicts with other packages.

## Usage

### Command line interface

After installation, you can use REMAG via the command line:

```bash
remag -f contigs.fasta -b alignments.bam -o output_directory
```

### Python script mode

You can also run REMAG as a Python script:

```bash
python remag.py -f contigs.fasta -b alignments.bam -o output_directory
```

### Python module mode

```bash
python -m remag -f contigs.fasta -b alignments.bam -o output_directory
```

## Features

- **Neural Networks**: Utilizes deep learning with contrastive learning for contig representation
- **Multi-modal Feature Extraction**: Combines k-mer frequencies, coverage profiles, and fragment features
- **Bacterial Filtering**: Includes integrated 4CAC classifier (xgbclass) for sequence classification
- **Advanced Clustering**: HDBSCAN clustering with optional K-means pre-clustering
- **Quality Control**: Built-in bin refinement and quality assessment
- **Flexible Input**: Supports both BAM files and TSV coverage data
- **Rich Visualization**: UMAP projections and interactive plots

## Options

```
  -f, --fasta PATH                Input FASTA file with contigs to bin. Can be gzipped.  [required]
  -b, --bam PATH                  Input BAM file(s) for coverage calculation. Must be indexed. Each BAM represents a sample. Supports glob patterns (e.g., '*.bam', 'sample_*.bam').
  -t, --tsv PATH                  Input TSV file(s) with coverage information.
  -o, --output PATH               Output directory for results.  [required]
  --epochs INTEGER RANGE          Training epochs for neural network.  [default: 400; 1<=x<=10000]
  --batch-size INTEGER RANGE      Batch size for training.  [default: 512; 32<=x<=8192]
  --embedding-dim INTEGER RANGE   Embedding dimension for contrastive learning.  [default: 256; 32<=x<=512]
  --nce-temperature FLOAT RANGE   Temperature for InfoNCE loss.  [default: 0.07; 0.01<=x<=1.0]
  --min-cluster-size INTEGER RANGE
                                  Minimum fragments per cluster.  [default: 2; 1<=x<=1000]
  --min-samples INTEGER RANGE     Minimum samples for HDBSCAN core points.  [default: 1; 1<=x<=1000]
  --min-contig-length INTEGER RANGE
                                  Minimum contig length in bp.  [default: 1000; 500<=x<=50000]
  --max-positive-pairs INTEGER RANGE
                                  Maximum positive pairs for contrastive learning.  [default: 5000000; 10000<=x<=50000000]
  -c, --cores INTEGER RANGE       Number of CPU cores.  [default: 8; 1<=x<=128]
  --min-bin-size INTEGER RANGE    Minimum bin size in bp.  [default: 50000; 50000<=x<=10000000]
  -v, --verbose                   Enable verbose logging.
  --enable-preclustering / --disable-preclustering
                                  Enable K-means pre-clustering to remove bacterial contigs before HDBSCAN.  [default: enable-preclustering]
  --skip-bacterial-filter         Skip bacterial contig filtering (4CAC classifier + contrastive learning).
  --skip-refinement               Skip bin refinement.
  --max-refinement-rounds INTEGER RANGE
                                  Maximum refinement rounds.  [default: 2; 1<=x<=5]
  --num-augmentations INTEGER RANGE
                                  Number of random fragments per contig.  [default: 8; 0<=x<=64]
  -h, --help                      Show this message and exit.
```

## Output

REMAG produces several output files:

- `bins/`: Directory containing FASTA files for each bin
- `final_clusters_contigs.csv`: Final contig-to-bin assignments
- `embeddings.csv`: Contig embeddings from the neural network
- `umap_embeddings.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot
- `siamese_model.pt`: Trained neural network model
- `remag.log`: Detailed log file

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- XGBoost (for 4CAC classifier)
- HDBSCAN
- UMAP
- pandas
- numpy
- matplotlib
- pysam
- loguru
- tqdm
- rich-click
- joblib

The package includes the pre-trained 4CAC classifier models for bacterial sequence filtering. The 4CAC classifier code is adapted from the [Shamir-Lab/4CAC repository](https://github.com/Shamir-Lab/4CAC).

## Usage Examples

For detailed usage examples, see the `examples/` directory:

- `examples/xgbclass_example.py`: Demonstrates how to use the integrated 4CAC classifier for sequence classification

## Acknowledgments

The integrated 4CAC classifier (`xgbclass` module) is adapted from the work by Shamir Lab:

- **Repository**: [Shamir-Lab/4CAC](https://github.com/Shamir-Lab/4CAC)
- **Paper**: 4CAC - 4-class contamination assessment classifier for prokaryotic genome assemblies

## License

MIT License - see LICENSE file for details.

## Citation

If you use REMAG in your research, please cite:

```
[Citation information will be added when available]
```
