"""
HyenaDNA-based eukaryotic sequence classifier for REMAG.

This module provides a classifier interface compatible with REMAG's
filtering pipeline, using a pre-trained HyenaDNA model for eukaryotic
sequence detection.
"""

import os
import sys
import random
from typing import List, Dict, Tuple
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import numpy as np
from loguru import logger

# Add compatibility layer for models trained with different module paths
# This allows loading models that expect 'hyenadna_model' module
if 'hyenadna_model' not in sys.modules:
    try:
        from . import hyenadna_model
        sys.modules['hyenadna_model'] = hyenadna_model
    except ImportError:
        pass  # Will fall back to full module path

try:
    from .standalone_tokenizer import StandaloneCharacterTokenizer as CharacterTokenizer
except ImportError:
    from standalone_tokenizer import StandaloneCharacterTokenizer as CharacterTokenizer


def sliding_window(sequence: str, window_size: int = 4096, stride: int = 2048):
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


def estimate_window_count(seq_len: int, window_size: int = 4096, stride: int = 2048) -> int:
    """Estimate how many windows will be generated."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if seq_len <= window_size:
        return 1

    full_windows = ((seq_len - window_size) // stride) + 1
    next_start = full_windows * stride
    has_partial = next_start < seq_len and (seq_len - next_start) >= window_size // 2

    return full_windows + (1 if has_partial else 0)


def generate_random_windows(sequence: str, window_size: int, num_windows: int, seed: int = 42) -> List[str]:
    """Generate random windows from a sequence for additional sampling.

    Args:
        sequence: DNA sequence to sample from
        window_size: Size of each window
        num_windows: Number of random windows to generate
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of random window sequences
    """
    seq_len = len(sequence)
    if seq_len <= window_size:
        return [sequence]

    windows = []
    max_start = seq_len - window_size

    # Use local seeded RNG for deterministic behavior
    rng = random.Random(seed)
    for _ in range(num_windows):
        start = rng.randint(0, max_start)
        window = sequence[start:start + window_size]
        windows.append(window)

    return windows


def _build_window_encoder(tokenizer):
    """Create a fast window encoder for the tokenizer."""
    def slow_encoder(windows: List[str]) -> torch.Tensor:
        encoded = tokenizer(
            windows,
            padding='max_length',
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 4096),
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


class HyenaDNAClassifier:
    """
    Dual-model HyenaDNA-based classifier for eukaryotic sequence detection.

    Uses adaptive model selection:
    - Small model (1024 context) for contigs < 4096bp with aggressive windowing
    - Large model (4096 context) for contigs >= 4096bp for better long-range detection

    This class provides an interface similar to the previous xgbClass,
    compatible with REMAG's filtering pipeline.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'auto',
        window_size: int = None,
        stride: int = None,
        batch_size: int = 64,
        min_contig_length: int = 1024,
        use_dual_models: bool = True,
        length_threshold: int = 4096,
    ):
        """
        Initialize the HyenaDNA classifier with dual-model support.

        Args:
            model_path: Path to specific model file. If None, uses dual-model approach.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            window_size: Override window size (None = auto-select based on model)
            stride: Override stride (None = auto-select based on model)
            batch_size: Batch size for inference
            min_contig_length: Minimum contig length to classify
            use_dual_models: Use both 1024 and 4096 models adaptively (default: True)
            length_threshold: Sequence length threshold for model selection (default: 4096)
        """
        # Set random seeds for deterministic behavior
        random.seed(42)
        np.random.seed(42)

        self.batch_size = batch_size
        self.min_contig_length = min_contig_length
        self.use_dual_models = use_dual_models
        self.length_threshold = length_threshold

        # Determine device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        curr_path = os.path.dirname(os.path.abspath(__file__))

        # Initialize dual models or single model
        if use_dual_models and model_path is None:
            logger.info("Initializing dual-model classifier (1024 + 4096 context)")

            # Load small model (1024 context)
            small_model_path = os.path.join(curr_path, "models", "pytorch_model.bin.20251024")
            if not os.path.exists(small_model_path):
                raise FileNotFoundError(f"Small model not found: {small_model_path}")

            logger.debug(f"Loading small model (1024) from {small_model_path}")
            self.small_model = torch.load(small_model_path, map_location=self.device, weights_only=False)
            self.small_model = self.small_model.to(self.device)
            self.small_model.eval()

            # Load large model (4096 context)
            large_model_path = os.path.join(curr_path, "models", "pytorch_model.bin.20251116.4k")
            if not os.path.exists(large_model_path):
                raise FileNotFoundError(f"Large model not found: {large_model_path}")

            logger.debug(f"Loading large model (4096) from {large_model_path}")
            self.large_model = torch.load(large_model_path, map_location=self.device, weights_only=False)
            self.large_model = self.large_model.to(self.device)
            self.large_model.eval()

            # Create tokenizers for both models
            self.small_tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=1024,
                padding_side='left',
            )

            self.large_tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=4096,
                padding_side='left',
            )

            # Set to None - will be selected dynamically
            self.model = None
            self.tokenizer = None
            self.window_size = None
            self.stride = None

        else:
            # Single model mode (backward compatibility)
            if model_path is None:
                model_path = os.path.join(curr_path, "models", "pytorch_model.bin.20251116.4k")
                default_window_size = 4096
                default_stride = 2048
                default_max_length = 4096
            else:
                # Infer parameters from model name
                if '4k' in model_path or '4096' in model_path:
                    default_window_size = 4096
                    default_stride = 2048
                    default_max_length = 4096
                else:
                    default_window_size = 1024
                    default_stride = 512
                    default_max_length = 1024

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.debug(f"Loading single model from {model_path}")
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=default_max_length,
                padding_side='left',
            )

            self.window_size = window_size if window_size is not None else default_window_size
            self.stride = stride if stride is not None else default_stride
            self.small_model = None
            self.large_model = None
            self.small_tokenizer = None
            self.large_tokenizer = None

        logger.debug("HyenaDNA classifier initialized successfully")

    @torch.inference_mode()
    def _predict_window_stats(
        self,
        windows: List[str],
        model=None,
        tokenizer=None,
    ) -> Tuple[int, float, int]:
        """Run inference on windows and return statistics.

        Args:
            windows: List of window sequences to predict
            model: Model to use (if None, uses self.model)
            tokenizer: Tokenizer to use (if None, uses self.tokenizer)
        """
        if not windows:
            return 0, 0.0, 0

        # Use provided model/tokenizer or fall back to instance defaults
        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        encoder = _get_window_encoder(tokenizer)
        input_ids = encoder(windows).to(self.device)

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
        self,
        sequence: str,
        adaptive_stride: bool = True,
        early_stopping: bool = True,
        low_confidence_threshold: float = 0.75,
        additional_random_windows: int = 8,
    ) -> Dict:
        """
        Predict whether a contig is eukaryotic using adaptive dual-model approach.

        Strategy:
        - Small contigs (<4096bp): Use 1024-context model with aggressive windowing (256 stride)
        - Large contigs (≥4096bp): Use 4096-context model with standard windowing
        - Special handling: Very low confidence triggers additional resampling

        Args:
            sequence: DNA sequence to classify
            adaptive_stride: Use adaptive stride based on sequence length
            early_stopping: Stop early when confidence is high
            low_confidence_threshold: Confidence threshold below which to add random windows (default: 0.75)
            additional_random_windows: Number of random windows to add when confidence is low (default: 8)

        Returns:
            Dictionary with prediction results including:
                - prediction: 'eukaryote' or 'non_eukaryote' (based on avg_prob >= 0.5)
                - eukaryote_prob: Average eukaryotic probability across all windows
                - confidence: Fraction of windows agreeing with final prediction
                - num_windows: Number of windows analyzed
                - length: Sequence length
                - resampled: Boolean indicating if random sampling was triggered
                - model_used: Which model was used ('small_1024', 'large_4096', or 'single')
        """
        seq_len = len(sequence)

        # Select model, tokenizer, window_size, and stride based on sequence length
        if self.use_dual_models:
            if seq_len < self.length_threshold:
                # Small contigs: use 1024 model with aggressive windowing
                model = self.small_model
                tokenizer = self.small_tokenizer
                window_size = 1024
                # More aggressive stride for better coverage on small contigs
                if adaptive_stride:
                    if seq_len < 2000:
                        stride = 256  # 75% overlap for very small contigs
                    elif seq_len < 3000:
                        stride = 384  # ~62% overlap
                    else:
                        stride = 512  # 50% overlap
                else:
                    stride = 512
                model_name = 'small_1024'
            else:
                # Large contigs: use 4096 model with standard windowing
                model = self.large_model
                tokenizer = self.large_tokenizer
                window_size = 4096
                if adaptive_stride:
                    if seq_len < 8000:
                        stride = 2048  # 50% overlap
                    elif seq_len < 20000:
                        stride = 4096  # No overlap
                    else:
                        stride = 8192  # Large jumps for very long contigs
                else:
                    stride = 2048
                model_name = 'large_4096'
        else:
            # Single model mode
            model = self.model
            tokenizer = self.tokenizer
            window_size = self.window_size
            stride = self.stride
            if adaptive_stride:
                if seq_len < 4000:
                    stride = window_size // 2
                elif seq_len < 20000:
                    stride = window_size
                else:
                    stride = window_size * 2
            model_name = 'single'

        window_iter = sliding_window(sequence, window_size, stride)
        total_windows = estimate_window_count(seq_len, window_size, stride) if early_stopping else None

        num_windows = 0
        num_eukaryote = 0
        prob_sum = 0.0
        resampled = False
        uncertainty_detected = False

        while True:
            batch = list(islice(window_iter, self.batch_size))
            if not batch:
                break

            euk_count, prob_total, batch_windows = self._predict_window_stats(batch, model, tokenizer)

            if batch_windows == 0:
                continue

            num_windows += batch_windows
            num_eukaryote += euk_count
            prob_sum += prob_total

            if early_stopping and num_windows >= 3:
                # Calculate current confidence
                avg_prob = prob_sum / num_windows
                is_eukaryote = avg_prob >= 0.5

                if is_eukaryote:
                    confidence = num_eukaryote / num_windows
                else:
                    confidence = (num_windows - num_eukaryote) / num_windows

                # High confidence - stop early
                if confidence >= 0.90:
                    break

                # Low confidence detected - switch to random sampling immediately
                if confidence < low_confidence_threshold and not uncertainty_detected and seq_len > window_size:
                    uncertainty_detected = True
                    resampled = True

                    # Generate random windows and add them to current batch processing
                    # For small contigs, use more random windows for better coverage
                    num_random = additional_random_windows * 2 if seq_len < self.length_threshold else additional_random_windows
                    random_windows = generate_random_windows(sequence, window_size, num_random)
                    euk_count_extra, prob_total_extra, batch_windows_extra = self._predict_window_stats(
                        random_windows, model, tokenizer
                    )

                    if batch_windows_extra > 0:
                        num_windows += batch_windows_extra
                        num_eukaryote += euk_count_extra
                        prob_sum += prob_total_extra

                        # Recalculate confidence with random windows added
                        avg_prob = prob_sum / num_windows
                        is_eukaryote = avg_prob >= 0.5

                        if is_eukaryote:
                            confidence = num_eukaryote / num_windows
                        else:
                            confidence = (num_windows - num_eukaryote) / num_windows

                        # Check if random sampling resolved uncertainty
                        if confidence >= 0.85:
                            break

                # Check impossible outcomes for early stopping
                if total_windows is not None and (
                    2 * num_eukaryote > total_windows
                    or 2 * (num_windows - num_eukaryote) > total_windows
                ):
                    break

        # Special case: Very low confidence with only 2 windows - force more sampling
        if num_windows == 2 and prob_sum / num_windows in [0.5]:
            # Ambiguous case - add more random windows
            num_additional = 10 if seq_len < self.length_threshold else 6
            random_windows = generate_random_windows(sequence, window_size, num_additional)
            euk_count_extra, prob_total_extra, batch_windows_extra = self._predict_window_stats(
                random_windows, model, tokenizer
            )
            if batch_windows_extra > 0:
                num_windows += batch_windows_extra
                num_eukaryote += euk_count_extra
                prob_sum += prob_total_extra
                resampled = True

        if num_windows == 0:
            return {
                'prediction': 'non_eukaryote',
                'eukaryote_prob': 0.0,
                'confidence': 0.0,
                'num_windows': 0,
                'length': seq_len,
                'resampled': False,
                'model_used': model_name
            }

        # Calculate final results
        avg_prob = prob_sum / num_windows if num_windows else 0.0
        is_eukaryote = avg_prob >= 0.5

        if is_eukaryote:
            confidence = num_eukaryote / num_windows
        else:
            confidence = (num_windows - num_eukaryote) / num_windows

        return {
            'prediction': 'eukaryote' if is_eukaryote else 'non_eukaryote',
            'eukaryote_prob': float(avg_prob),
            'confidence': float(confidence),
            'num_windows': num_windows,
            'length': seq_len,
            'resampled': resampled,
            'model_used': model_name
        }

    def _classify_single(self, seq):
        """Helper method for parallel classification."""
        if len(seq) < self.min_contig_length:
            return [1.0, 0.0]  # Too short

        result = self.predict_contig(seq)
        euk_prob = result['eukaryote_prob']
        non_euk_prob = 1.0 - euk_prob
        return [non_euk_prob, euk_prob]

    def classify(self, sequences, parallel=True, max_workers=4):
        """
        Classify sequences as eukaryotic or non-eukaryotic.

        Compatible interface with xgbClass.classify().

        Args:
            sequences: Either a single string or a list of strings
            parallel: Use parallel processing for batch classification (default: True)
            max_workers: Maximum number of parallel workers (default: 4)

        Returns:
            For single sequence: numpy array of shape (1, 2) with [non_euk_prob, euk_prob]
            For list of sequences: numpy array of shape (n, 2) with probabilities for each
        """
        if isinstance(sequences, str):
            # Single sequence
            if len(sequences) < self.min_contig_length:
                return np.array([[1.0, 0.0]])  # Too short, classify as non-eukaryotic

            result = self.predict_contig(sequences)
            euk_prob = result['eukaryote_prob']
            non_euk_prob = 1.0 - euk_prob

            return np.array([[non_euk_prob, euk_prob]])

        elif isinstance(sequences, list):
            # List of sequences
            logger.info(f"{len(sequences)} sequences to classify")

            if parallel and len(sequences) > 10:
                # Use parallel processing for larger batches
                # Note: GPU models need to be used carefully in parallel - ThreadPoolExecutor
                # shares memory space so the CUDA context is shared
                logger.info(f"Using parallel processing with {max_workers} workers")

                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, result in enumerate(executor.map(self._classify_single, sequences)):
                        results.append(result)
                        if (i + 1) % 1000 == 0:
                            logger.info(f"Classified {i + 1}/{len(sequences)} sequences")

                logger.info(f"Classification complete: {len(sequences)} sequences")
                return np.array(results)
            else:
                # Sequential processing for small batches or when parallel is disabled
                results = []
                for i, seq in enumerate(sequences):
                    if i > 0 and i % 1000 == 0:
                        logger.info(f"Classified {i}/{len(sequences)} sequences")

                    if len(seq) < self.min_contig_length:
                        results.append([1.0, 0.0])  # Too short
                        continue

                    result = self.predict_contig(seq)
                    euk_prob = result['eukaryote_prob']
                    non_euk_prob = 1.0 - euk_prob
                    results.append([non_euk_prob, euk_prob])

                logger.info(f"Classification complete: {len(sequences)} sequences")
                return np.array(results)

        else:
            raise TypeError("Can only classify strings or lists of strings")
