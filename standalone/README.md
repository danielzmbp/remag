# HyenaDNA Eukaryote Predictor

A standalone predictor for identifying eukaryotic contigs using the HyenaDNA model.

## Features

- **Fast prediction** using sliding window with majority voting
- **Minimal dependencies** (~5 packages)
- **GPU/CPU/MPS support** with automatic device detection
- **Outputs**:
  - Filtered FASTA file with eukaryotic sequences
  - TSV table with predictions and confidence scores

## Installation

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate hyenadna-predictor
```

This installs the minimal dependencies:
- PyTorch (GPU support)
- BioPython (FASTA I/O)
- einops (tensor operations)
- tqdm (progress bars)
- numpy

### 2. Verify setup

The package includes:
- `predict_eukaryotes.py` - Main prediction script
- `standalone_hyenadna.py` - HyenaDNA model architecture
- `pytorch_model.bin` - Pre-trained model weights (1.7MB)

## Usage

### Basic usage

```bash
python predict_eukaryotes.py \
  --input contigs.fasta \
  --output-fasta eukaryotes.fasta \
  --output-table predictions.tsv
```

### Options

- `--input, -i` (required): Input FASTA file
- `--output-fasta, -f` (required): Output FASTA with eukaryotic sequences
- `--output-table, -t` (required): Output TSV with predictions
- `--model, -m`: Path to model file (default: `./pytorch_model.bin`)
- `--threshold`: Confidence threshold for filtering (default: 0.5, range: 0.0-1.0)
- `--min-length`: Minimum contig length to predict (default: 1024 bp)
- `--device`: Device to use: `auto` (default), `cpu`, `cuda`, or `mps`
- `--batch-size`: Batch size for prediction (default: 64)

### Examples

**Conservative filtering (high confidence only)**
```bash
python predict_eukaryotes.py \
  --input contigs.fasta \
  --output-fasta eukaryotes_confident.fasta \
  --output-table predictions.tsv \
  --threshold 0.9
```

**Use CPU only**
```bash
python predict_eukaryotes.py \
  --input contigs.fasta \
  --output-fasta eukaryotes.fasta \
  --output-table predictions.tsv \
  --device cpu
```

**Process smaller contigs**
```bash
python predict_eukaryotes.py \
  --input contigs.fasta \
  --output-fasta eukaryotes.fasta \
  --output-table predictions.tsv \
  --min-length 500
```

## Output Format

### Filtered FASTA (`--output-fasta`)
Contains only contigs predicted as eukaryotic with confidence >= threshold.

```fasta
>contig_123 length=5000 prediction=eukaryote
ACGTACGTACGT...
>contig_456 length=8000 prediction=eukaryote
ACGTACGTACGT...
```

### Predictions Table (`--output-table`)
Tab-separated values with all predictions.

```
contig_id       length  num_windows     prediction      eukaryote_prob  confidence
contig_1        5000    5               eukaryote       0.8923          0.8000
contig_2        3000    3               non_eukaryote   0.2145          0.6667
contig_3        8000    8               eukaryote       0.9234          1.0000
```

**Columns:**
- `contig_id`: Sequence identifier from FASTA header
- `length`: Contig length in base pairs
- `num_windows`: Number of 1024bp windows evaluated
- `prediction`: Classification (eukaryote or non_eukaryote)
- `eukaryote_prob`: Average eukaryotic probability across windows
- `confidence`: Proportion of windows supporting the majority vote

## Performance

Tested on diverse genomic datasets:
- **Accuracy**: ~83%
- **Precision**: ~86.5%
- **Recall**: ~89.5%

On typical hardware:
- **CPU**: ~2-5 contigs/sec
- **GPU**: ~30-60 contigs/sec
- **M1/M2 (MPS)**: ~40-80 contigs/sec

## Technical Details

### Prediction Algorithm

1. Splits contigs into 1024bp overlapping windows (50% overlap)
2. Runs model inference on each window
3. Uses majority voting across windows
4. Returns confidence as proportion supporting prediction
5. Filters by confidence threshold (default: 0.5)

### Adaptive Stride

For faster inference on large contigs:
- Contigs < 2kb: 512bp stride (50% overlap)
- Contigs 2-10kb: 2048bp stride (2x window)
- Contigs > 10kb: 8192bp stride (8x window)

Early stopping activates once confidence reaches 90%.

## Troubleshooting

### Model not found
Ensure `pytorch_model.bin` is in the same directory as the script.

### Out of memory
- Reduce `--batch-size` (try 32 or 16)
- Use `--device cpu` instead of GPU
- Process in smaller batches

### Slow prediction
- Use GPU: `--device cuda` or `--device mps` (Apple Silicon)
- Increase `--batch-size` for better GPU utilization
- Use default adaptive stride (enabled by default)

## Citation

Based on HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution
https://arxiv.org/abs/2306.15794

## License

MIT
