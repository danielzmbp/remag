#!/usr/bin/env python
"""
Standalone HyenaDNA Eukaryote Predictor

Predicts whether contigs are eukaryotic and outputs:
1. Filtered FASTA file (only eukaryotic sequences)
2. Predictions table (TSV format)

Usage:
    python predict_eukaryotes.py --input contigs.fasta --output-fasta eukaryotes.fasta --output-table predictions.tsv
"""

import sys
import argparse
from pathlib import Path
from itertools import islice
from typing import Iterator, Tuple, Dict, List

import torch
from Bio import SeqIO

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from standalone_hyenadna import CharacterTokenizer
except ImportError:
    print("Error: Could not import standalone_hyenadna.py", file=sys.stderr)
    print("Make sure standalone_hyenadna.py is in the same directory as this script", file=sys.stderr)
    sys.exit(1)


def load_model(model_path: Path, device: str = 'cpu'):
    """Load trained HyenaDNA model and tokenizer."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    hyena_model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )
    hyena_model = hyena_model.to(device)
    hyena_model.eval()

    # Create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=1024,
        padding_side='left',
    )

    return hyena_model, tokenizer


def sliding_window(sequence: str, window_size: int = 1024, stride: int = 512) -> Iterator[str]:
    """Yield sliding windows from a sequence."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    sequence = sequence.upper()
    seq_len = len(sequence)

    if seq_len <= window_size:
        yield sequence
        return

    start = 0
    while True:
        end = min(start + window_size, seq_len)
        window = sequence[start:end]

        if len(window) >= window_size // 2:
            yield window

        if end == seq_len:
            break

        start += stride
        if start >= seq_len:
            break


def estimate_window_count(seq_len: int, window_size: int = 1024, stride: int = 512) -> int:
    """Estimate how many windows will be generated."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if seq_len <= window_size:
        return 1

    full_windows = ((seq_len - window_size) // stride) + 1
    next_start = full_windows * stride
    has_partial = next_start < seq_len and (seq_len - next_start) >= window_size // 2

    return full_windows + (1 if has_partial else 0)


def _build_window_encoder(tokenizer):
    """Create a fast window encoder for the tokenizer."""
    def slow_encoder(windows: List[str]) -> torch.Tensor:
        encoded = tokenizer(
            windows,
            padding='max_length',
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 1024),
            return_tensors='pt'
        )
        return encoded['input_ids']

    required_attrs = [
        "model_max_length",
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "unk_token_id",
        "padding_side",
    ]
    if not all(hasattr(tokenizer, attr) for attr in required_attrs):
        return slow_encoder

    max_length = int(tokenizer.model_max_length)
    pad_id = int(tokenizer.pad_token_id)
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    unk_id = int(tokenizer.unk_token_id)
    max_tokens = max_length - 2

    if getattr(tokenizer, "padding_side", "right") != 'left' or max_tokens <= 0:
        return slow_encoder

    char_to_id: Dict[str, int] = {}
    if hasattr(tokenizer, "characters"):
        for ch in tokenizer.characters:
            idx = tokenizer.convert_tokens_to_ids(ch)
            char_to_id[ch] = idx
            char_to_id[ch.upper()] = idx
            char_to_id[ch.lower()] = idx

    def encode(windows: List[str]) -> torch.Tensor:
        batch_size = len(windows)
        if batch_size == 0:
            return torch.empty((0, max_length), dtype=torch.long)

        input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
        lookup = char_to_id.get

        for row, window in enumerate(windows):
            seq = window.upper()[:max_tokens]
            tokens = [lookup(ch, unk_id) for ch in seq]

            length = len(tokens) + 2
            length = min(length, max_length)
            start = max_length - length

            end_idx = start
            input_ids[row, end_idx] = cls_id
            end_idx += 1

            if tokens:
                available = max_length - end_idx - 1
                if available > 0:
                    use_tokens = tokens[:available]
                    input_ids[row, end_idx:end_idx + len(use_tokens)] = torch.tensor(
                        use_tokens, dtype=torch.long
                    )
                    end_idx += len(use_tokens)

            sep_position = min(end_idx, max_length - 1)
            input_ids[row, sep_position] = sep_id

        return input_ids

    return encode


def _get_window_encoder(tokenizer):
    """Get or create cached window encoder."""
    encoder = getattr(tokenizer, "_cached_window_encoder", None)
    if encoder is None:
        encoder = _build_window_encoder(tokenizer)
        tokenizer._cached_window_encoder = encoder
    return encoder


@torch.inference_mode()
def predict_window_stats(
    windows: List[str],
    model,
    tokenizer,
    device: str,
) -> Tuple[int, float, int]:
    """Run inference on windows and return statistics."""
    if not windows:
        return 0, 0.0, 0

    encoder = _get_window_encoder(tokenizer)
    input_ids = encoder(windows).to(device)

    outputs = model(input_ids)

    if isinstance(outputs, tuple):
        logits = outputs[0]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs

    if logits.shape[-1] == 2:
        diff = logits[:, 1] - logits[:, 0]
        preds = diff >= 0
        euk_probs = torch.sigmoid(diff)
    else:
        probs = torch.softmax(logits, dim=-1)
        euk_probs = probs[:, 1]
        preds = torch.argmax(probs, dim=-1) == 1

    batch_windows = preds.shape[0]
    euk_count = int(preds.sum().item())
    prob_sum = float(euk_probs.sum().item())

    return euk_count, prob_sum, batch_windows


def predict_contig(
    sequence: str,
    model,
    tokenizer,
    device: str,
    window_size: int = 1024,
    stride: int = 512,
    batch_size: int = 32,
    adaptive_stride: bool = True,
    stride_short: int = 512,
    stride_medium: int = 2048,
    stride_long: int = 8192,
    early_stopping: bool = True,
    early_stop_threshold: float = 0.90,
    early_stop_min_windows: int = 3,
    early_stop_check_interval: int = 16,
) -> Dict:
    """
    Predict whether a contig is eukaryotic using sliding window + majority voting.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if adaptive_stride:
        seq_len = len(sequence)
        if seq_len < 2000:
            stride = stride_short
        elif seq_len < 10000:
            stride = stride_medium
        else:
            stride = stride_long

    window_iter = sliding_window(sequence, window_size, stride)
    total_windows = estimate_window_count(len(sequence), window_size, stride) if early_stopping else None

    max_batch_size = batch_size
    chunk_size = batch_size
    if early_stopping:
        chunk_size = max(1, min(early_stop_check_interval, batch_size))

    num_windows = 0
    num_eukaryote = 0
    prob_sum = 0.0

    while True:
        batch = list(islice(window_iter, chunk_size))
        if not batch:
            break

        euk_count, prob_total, batch_windows = predict_window_stats(batch, model, tokenizer, device)

        if batch_windows == 0:
            continue

        num_windows += batch_windows
        num_eukaryote += euk_count
        prob_sum += prob_total

        if early_stopping:
            if (
                total_windows is not None
                and (
                    2 * num_eukaryote > total_windows
                    or 2 * (num_windows - num_eukaryote) > total_windows
                )
            ):
                break

            if num_windows >= early_stop_min_windows:
                majority_pred = 1 if num_eukaryote * 2 > num_windows else 0
                if majority_pred == 1:
                    confidence = num_eukaryote / num_windows
                else:
                    confidence = (num_windows - num_eukaryote) / num_windows

                if confidence >= early_stop_threshold:
                    break

        if early_stopping and chunk_size < max_batch_size:
            chunk_size = min(chunk_size * 2, max_batch_size)

    if num_windows == 0:
        return {
            'prediction': 'non_eukaryote',
            'eukaryote_prob': 0.0,
            'confidence': 0.0,
            'num_windows': 0,
            'length': len(sequence)
        }

    majority_pred = 1 if num_eukaryote > num_windows / 2 else 0

    if majority_pred == 1:
        confidence = num_eukaryote / num_windows
    else:
        confidence = (num_windows - num_eukaryote) / num_windows

    avg_prob = prob_sum / num_windows if num_windows else 0.0

    return {
        'prediction': 'eukaryote' if majority_pred == 1 else 'non_eukaryote',
        'eukaryote_prob': float(avg_prob),
        'confidence': float(confidence),
        'num_windows': num_windows,
        'length': len(sequence)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Predict eukaryotic contigs using HyenaDNA'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input FASTA file with contigs'
    )
    parser.add_argument(
        '--output-fasta', '-f',
        type=Path,
        required=True,
        help='Output FASTA file with eukaryotic contigs'
    )
    parser.add_argument(
        '--output-table', '-t',
        type=Path,
        required=True,
        help='Output TSV file with predictions'
    )
    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=Path(__file__).parent / 'pytorch_model.bin',
        help='Path to model file (default: ./pytorch_model.bin)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for filtering as eukaryotic (default: 0.5)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=1024,
        help='Minimum contig length to predict (default: 1024)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for prediction (default: 64)'
    )

    args = parser.parse_args()

    # Check input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check model file
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}", file=sys.stderr)
    print(f"Loading model from {args.model}...", file=sys.stderr)

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Count total contigs
    total_contigs = sum(1 for _ in SeqIO.parse(args.input, "fasta"))

    results = []
    eukaryotic_records = []
    num_skipped = 0

    # Process sequences
    print(f"Predicting on {total_contigs} contigs...", file=sys.stderr)
    iterator = SeqIO.parse(args.input, "fasta")
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=total_contigs, desc="Predicting", unit="contig")

    for record in iterator:
        sequence = str(record.seq)

        # Skip short contigs
        if len(sequence) < args.min_length:
            num_skipped += 1
            continue

        # Predict
        result = predict_contig(
            sequence,
            model,
            tokenizer,
            device,
            window_size=1024,
            stride=512,
            batch_size=args.batch_size,
            adaptive_stride=True,
        )

        result['contig_id'] = record.id
        results.append(result)

        # Collect eukaryotic records
        if result['prediction'] == 'eukaryote' and result['confidence'] >= args.threshold:
            eukaryotic_records.append(record)

    # Write eukaryotic FASTA
    print(f"Writing {len(eukaryotic_records)} eukaryotic sequences to {args.output_fasta}...", file=sys.stderr)
    SeqIO.write(eukaryotic_records, args.output_fasta, "fasta")

    # Write predictions table
    print(f"Writing predictions to {args.output_table}...", file=sys.stderr)
    with open(args.output_table, 'w') as f:
        # Header
        f.write("contig_id\tlength\tnum_windows\tprediction\teukaryote_prob\tconfidence\n")

        # Results
        for result in results:
            f.write(
                f"{result['contig_id']}\t"
                f"{result['length']}\t"
                f"{result['num_windows']}\t"
                f"{result['prediction']}\t"
                f"{result['eukaryote_prob']:.4f}\t"
                f"{result['confidence']:.4f}\n"
            )

    # Summary statistics
    num_eukaryote = sum(1 for r in results if r['prediction'] == 'eukaryote')
    num_non_eukaryote = len(results) - num_eukaryote
    num_filtered = sum(1 for r in results if r['prediction'] == 'eukaryote' and r['confidence'] >= args.threshold)

    print(f"\nResults:", file=sys.stderr)
    print(f"  Predicted: {len(results):,} contigs", file=sys.stderr)
    print(f"    Eukaryotic: {num_eukaryote:,} ({100*num_eukaryote/len(results):.1f}%)", file=sys.stderr)
    print(f"    Non-eukaryotic: {num_non_eukaryote:,} ({100*num_non_eukaryote/len(results):.1f}%)", file=sys.stderr)
    print(f"  Filtered (confidence >= {args.threshold}): {num_filtered:,} sequences", file=sys.stderr)
    if num_skipped > 0:
        print(f"  Skipped: {num_skipped:,} contigs <{args.min_length}bp", file=sys.stderr)


if __name__ == '__main__':
    main()
