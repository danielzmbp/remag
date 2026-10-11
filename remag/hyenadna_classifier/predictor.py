"""
HyenaDNA-based eukaryotic sequence classifier for REMAG.

This module provides a classifier interface compatible with REMAG's
filtering pipeline, using a pre-trained HyenaDNA model for eukaryotic
sequence detection.
"""

import hashlib
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
from loguru import logger

from ..filter_policy import (
    EUKARYOTE_FILTER_THRESHOLD,
    INFERENCE_BATCH_SIZE,
    MODEL_FILENAME,
    MODEL_SHA256,
    RECORD_BATCH_SIZE,
    WINDOW_BUDGET,
    WINDOW_SIZE,
    ClassifierPolicyError,
)

# Add compatibility layer for models trained with different module paths
# This allows loading models that expect 'hyenadna_model' module
if "hyenadna_model" not in sys.modules:
    try:
        from . import hyenadna_model

        sys.modules["hyenadna_model"] = hyenadna_model
    except ImportError:
        pass  # Will fall back to full module path

try:
    from .standalone_tokenizer import StandaloneCharacterTokenizer as CharacterTokenizer
except ImportError:
    from standalone_tokenizer import StandaloneCharacterTokenizer as CharacterTokenizer


def uniform_positions(length: int) -> List[int]:
    """Spread up to 16 nominal 4 kb windows over the entire contig."""
    if length <= WINDOW_SIZE:
        return [0]
    n = min(WINDOW_BUDGET, 1 + math.ceil((length - WINDOW_SIZE) / 2048))
    return sorted({round(j * (length - WINDOW_SIZE) / (n - 1)) for j in range(n)})


def _build_window_encoder(tokenizer):
    """Create a fast window encoder for the tokenizer."""

    def slow_encoder(windows: List[str]) -> torch.Tensor:
        encoded = tokenizer(
            windows,
            padding="max_length",
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 4096),
            return_tensors="pt",
        )
        return encoded["input_ids"]

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

    if getattr(tokenizer, "padding_side", "right") != "left" or max_tokens <= 0:
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
                    input_ids[row, end_idx : end_idx + len(use_tokens)] = torch.tensor(
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
    """Single 4 kb HyenaDNA model with deterministic, bounded window sampling."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        window_size: int = None,
        stride: int = None,
        batch_size: int = INFERENCE_BATCH_SIZE,
        min_contig_length: int = 1000,
        use_dual_models: bool = False,
        length_threshold: int = 4096,
    ):
        """Load the verified July 2026 checkpoint with its 4,096-token context.

        Legacy sampling/model-selection arguments are accepted for compatibility
        but cannot alter the selected policy. A custom path must contain the same
        verified checkpoint. Auto device selection uses CUDA when available,
        otherwise CPU; MPS has not been validated for this policy.
        """
        if batch_size != INFERENCE_BATCH_SIZE:
            logger.warning(
                "HyenaDNA uses the validated batch size of 256 windows; "
                f"ignoring requested batch size {batch_size}."
            )
        if (
            use_dual_models
            or window_size not in (None, WINDOW_SIZE)
            or stride not in (None, 2048)
            or length_threshold != 4096
        ):
            logger.warning("Legacy HyenaDNA model/stride settings are ignored.")
        self.batch_size = INFERENCE_BATCH_SIZE
        self.min_contig_length = min_contig_length
        self.use_dual_models = False
        self.window_size = WINDOW_SIZE
        self.small_model = self.large_model = None
        self.small_tokenizer = self.large_tokenizer = None

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(torch.device(device))
        if torch.device(self.device).type not in ("cpu", "cuda"):
            raise ClassifierPolicyError(
                "This HyenaDNA policy supports CPU and CUDA; MPS is not validated."
            )
        logger.info(f"HyenaDNA inference device: {self.device}")
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "models", MODEL_FILENAME
            )
        with open(model_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != MODEL_SHA256:
            raise ClassifierPolicyError(
                "HyenaDNA checkpoint SHA256 does not match the selected policy."
            )
        self.model = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],
            model_max_length=WINDOW_SIZE,
            padding_side="left",
        )

    @torch.inference_mode()
    def _predict_window_predictions(
        self,
        windows: List[str],
        model=None,
        tokenizer=None,
    ) -> List[Tuple[bool, float]]:
        """Run inference on windows and return per-window predictions/probabilities."""
        if not windows:
            return []

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

        combined = torch.stack((preds.to(euk_probs.dtype), euk_probs), dim=1)
        return [(bool(pred), float(prob)) for pred, prob in combined.cpu().tolist()]

    def _predict_batch(self, sequences):
        """Infer a bounded record batch in contig/start order, using Python sums."""
        requests = [
            (i, start)
            for i, sequence in enumerate(sequences)
            for start in uniform_positions(len(sequence))
        ]
        scores = [[] for _ in sequences]
        for begin in range(0, len(requests), INFERENCE_BATCH_SIZE):
            batch = requests[begin : begin + INFERENCE_BATCH_SIZE]
            windows = [sequences[i][start : start + WINDOW_SIZE] for i, start in batch]
            predictions = self._predict_window_predictions(windows)
            if len(predictions) != len(batch):
                raise ClassifierPolicyError(
                    "HyenaDNA returned an unexpected window count."
                )
            for (i, _), (vote, probability) in zip(batch, predictions):
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ClassifierPolicyError(
                        "HyenaDNA returned an invalid probability."
                    )
                scores[i].append((vote, probability))
        results = []
        for sequence, values in zip(sequences, scores):
            probability = sum(p for _, p in values) / len(values)
            # Preserve the diagnostic native 0.5 window-agreement statistic.
            # It does not participate in sampling or the 0.30 admission decision.
            native_class = probability >= 0.5
            confidence = sum(vote == native_class for vote, _ in values) / len(values)
            results.append(
                {
                    "prediction": (
                        "eukaryote"
                        if probability >= EUKARYOTE_FILTER_THRESHOLD
                        else "non_eukaryote"
                    ),
                    "eukaryote_prob": probability,
                    "confidence": confidence,
                    "num_windows": len(values),
                    "length": len(sequence),
                    "resampled": False,
                    "model_used": "single_4096_20260724",
                }
            )
        return results

    def _predict_full_precision_batch(self, sequences):
        cuda = torch.device(self.device).type == "cuda"
        # Keep native full-precision inference without changing later training.
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type="cuda" if cuda else "cpu", enabled=False):
                return self._predict_batch(sequences)
        finally:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def predict_contigs(
        self,
        sequences: List[str],
        adaptive_stride: bool = True,
        early_stopping: bool = False,
        low_confidence_threshold: float = 0.75,
        additional_random_windows: int = 8,
    ) -> List[Dict]:
        """Score all eligible contigs; legacy sampling arguments are ignored.

        Groups of 64 eligible contigs share inference calls of at most 256 windows.
        Every planned window contributes to the arithmetic mean. No RNG is used.
        """
        results = [
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.0,
                "confidence": 0.0,
                "num_windows": 0,
                "length": len(sequence),
                "resampled": False,
                "model_used": "none",
            }
            for sequence in sequences
        ]
        eligible = [
            i for i, seq in enumerate(sequences) if len(seq) >= self.min_contig_length
        ]
        for start in range(0, len(eligible), RECORD_BATCH_SIZE):
            indices = eligible[start : start + RECORD_BATCH_SIZE]
            batch = [sequences[i].upper() for i in indices]
            for i, result in zip(indices, self._predict_full_precision_batch(batch)):
                results[i] = result
        return results

    def predict_contig(
        self,
        sequence: str,
        adaptive_stride: bool = True,
        early_stopping: bool = False,
        low_confidence_threshold: float = 0.75,
        additional_random_windows: int = 8,
    ) -> Dict:
        """Predict one contig with the same fixed policy as batched filtering."""
        return self.predict_contigs([sequence])[0]
