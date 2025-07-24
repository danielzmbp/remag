# REMAG

**RE**covery of eukaryotic genomes using contrastive learning. A specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/danielzmbp/remag.git
cd remag

# Create conda environment and install
conda create -n remag python=3.9
conda activate remag
pip install .

# Run REMAG
remag -f contigs.fasta -b alignments.bam -o output_directory
```

## Installation

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

1. **Bacterial Pre-filtering**: Uses the integrated 4CAC classifier to identify and optionally remove bacterial contigs
2. **Feature Extraction**: Combines k-mer composition (4-mers) with coverage profiles, using fragment-based augmentation
3. **Representation Learning**: Trains a Siamese neural network with Barlow Twins contrastive learning to generate meaningful contig embeddings
4. **HDBSCAN Clustering**: HDBSCAN clustering on contig embeddings
5. **Quality Assessment**: Uses miniprot against eukaryotic core genes to detect contamination
6. **Iterative Refinement**: Splits contaminated bins based on core gene duplications

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
                                  Base learning rate for optimizer.  [default: 0.0003; 0.00001<=x<=0.01]
  --min-cluster-size INTEGER RANGE
                                  Minimum fragments per cluster.  [default: 2; 1<=x<=1000]
  --min-samples INTEGER RANGE     Minimum samples for HDBSCAN core points.  [default: None; 1<=x<=1000]
  --cluster-selection-epsilon FLOAT RANGE
                                  Epsilon for HDBSCAN cluster selection.  [default: 0.0; 0.0<=x<=5.0]
  --min-contig-length INTEGER RANGE
                                  Minimum contig length in bp.  [default: 1000; 500<=x<=50000]
  --max-positive-pairs INTEGER RANGE
                                  Maximum positive pairs for contrastive learning.  [default: 5000000; 10000<=x<=50000000]
  -c, --cores INTEGER RANGE       Number of CPU cores.  [default: 8; 1<=x<=128]
  --min-bin-size INTEGER RANGE    Minimum bin size in bp.  [default: 100000; 50000<=x<=10000000]
  -v, --verbose                   Enable verbose logging.
  --skip-bacterial-filter         Skip bacterial contig filtering (4CAC classifier + contrastive learning).
  --skip-refinement               Skip bin refinement.
  --skip-kmeans-filtering         Skip K-means filtering on embeddings.
  --max-refinement-rounds INTEGER RANGE
                                  Maximum refinement rounds.  [default: 2; 1<=x<=10]
  --num-augmentations INTEGER RANGE
                                  Number of random fragments per contig.  [default: 8; 0<=x<=64]
  --keep-intermediate             Keep intermediate files (training fragments, etc.).
  -h, --help                      Show this message and exit.
```

## Output

REMAG produces several output files:

- `bins/`: Directory containing FASTA files for each bin
- `bins.csv`: Final contig-to-bin assignments
- `remag.log`: Detailed log file

If using `--keep-intermediate` option also produces:

- `embeddings.csv`: Contig embeddings from the neural network
- `umap_embeddings.csv`: UMAP projections for visualization
- `umap_plot.pdf`: UMAP visualization plot
- `siamese_model.pt`: Trained neural network model


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

```
[Citation information will be added when available]
```
