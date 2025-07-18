# REMAG

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

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
git clone https://github.com/danielzmbp/remag.git
cd remag
pip install .
```

### Development installation

```bash
# Create and activate conda environment
conda create -n remag python=3.9
conda activate remag

# Clone and install in development mode
git clone https://github.com/danielzmbp/remag.git
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

## How REMAG Works

REMAG uses a sophisticated multi-stage pipeline specifically designed for eukaryotic genome recovery:

1. **Bacterial Pre-filtering**: Uses the integrated 4CAC classifier to identify and optionally remove bacterial contigs
2. **Feature Extraction**: Combines k-mer composition (4-mers) with coverage profiles, using fragment-based augmentation
3. **Representation Learning**: Trains a Siamese neural network with contrastive learning to generate meaningful contig embeddings
4. **HDBSCAN Clustering**: 
   - HDBSCAN clustering on contig embeddings
   - Noise point recovery using soft clustering
5. **Chimera Detection**: Analyzes large contigs for chimeric sequences using embedding similarity
6. **Quality Assessment**: Uses miniprot against eukaryotic core genes to detect contamination
7. **Iterative Refinement**: Splits contaminated bins based on core gene duplications

## Features

- **Eukaryotic-Centric Design**: Specifically optimized for recovering eukaryotic genomes from mixed samples
- **Contrastive Learning**: Uses Siamese neural networks with InfoNCE loss for contig representation learning
- **Multi-Modal Features**: Combines k-mer composition (4-mers) with coverage profiles using dual encoders
- **Bacterial Pre-filtering**: Integrated 4CAC classifier removes bacterial contigs before main clustering
- **Advanced Clustering**: HDBSCAN clustering with soft noise recovery
- **Chimera Detection**: Specialized detection of chimeric contigs using embedding similarity analysis
- **Quality-Driven Refinement**: Iterative bin splitting based on core gene duplications (miniprot + eukaryotic database)
- **Noise Recovery**: Advanced noise point recovery using approximate HDBSCAN prediction
- **Flexible Input**: Supports multiple BAM files (each representing a sample) and TSV coverage data
- **Rich Visualization**: UMAP projections with clustering results

## Options

```
  -f, --fasta PATH                Input FASTA file with contigs to bin. Can be gzipped.  [required]
  -b, --bam PATH                  Input BAM file(s) for coverage calculation. Must be indexed. Each BAM represents a sample. Supports space-separated files or glob patterns (e.g., "*.bam", "sample_*.bam"). Use quotes around glob patterns.
  -t, --tsv PATH                  Input TSV file(s) with coverage information.
  -o, --output PATH               Output directory for results.  [required]
  --epochs INTEGER RANGE          Training epochs for neural network.  [default: 400; 1<=x<=10000]
  --batch-size INTEGER RANGE      Batch size for training.  [default: 512; 32<=x<=8192]
  --embedding-dim INTEGER RANGE   Embedding dimension for contrastive learning.  [default: 256; 32<=x<=512]
  --nce-temperature FLOAT RANGE   Temperature for InfoNCE loss.  [default: 0.07; 0.01<=x<=1.0]
  --min-cluster-size INTEGER RANGE
                                  Minimum fragments per cluster.  [default: 5; 1<=x<=1000]
  --min-samples INTEGER RANGE     Minimum samples for HDBSCAN core points.  [default: 3; 1<=x<=1000]
  --min-contig-length INTEGER RANGE
                                  Minimum contig length in bp.  [default: 1000; 500<=x<=50000]
  --max-positive-pairs INTEGER RANGE
                                  Maximum positive pairs for contrastive learning.  [default: 5000000; 10000<=x<=50000000]
  -c, --cores INTEGER RANGE       Number of CPU cores.  [default: 8; 1<=x<=128]
  --min-bin-size INTEGER RANGE    Minimum bin size in bp.  [default: 100000; 50000<=x<=10000000]
  -v, --verbose                   Enable verbose logging.
  --skip-bacterial-filter         Skip bacterial contig filtering (4CAC classifier + contrastive learning).
  --skip-refinement               Skip bin refinement.
  --max-refinement-rounds INTEGER RANGE
                                  Maximum refinement rounds.  [default: 2; 1<=x<=5]
  --num-augmentations INTEGER RANGE
                                  Number of random fragments per contig.  [default: 8; 0<=x<=64]
  --skip-chimera-detection        Skip chimera detection for large contigs.
  --noise-recovery-threshold FLOAT RANGE
                                  Threshold for noise point recovery (lower = more permissive).  [default: 0.3; 0.1<=x<=0.9]
  -h, --help                      Show this message and exit.
```

## Output

REMAG produces several output files:

- `bins/`: Directory containing FASTA files for each bin
- `bins.csv`: Final contig-to-bin assignments (excludes noise contigs)
- `embeddings.csv`: Contig embeddings from the neural network
- `umap_embeddings.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot
- `siamese_model.pt`: Trained neural network model
- `remag.log`: Detailed log file

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
