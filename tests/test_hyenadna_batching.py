"""Regressions for the validated deterministic single-model filtering policy."""

import hashlib
import math
import random
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from remag.filter_policy import MODEL_FILENAME, MODEL_SHA256, ClassifierPolicyError
from remag.hyenadna_classifier import predictor
from remag.hyenadna_classifier.predictor import (
    CharacterTokenizer,
    HyenaDNAClassifier,
    _build_window_encoder,
    uniform_positions,
)


@pytest.mark.parametrize(
    "length,expected",
    [
        (999, [0]),
        (1000, [0]),
        (4094, [0]),
        (4095, [0]),
        (4096, [0]),
        (4097, [0, 1]),
        (6144, [0, 2048]),
        (6145, [0, 1024, 2049]),
        (8192, [0, 2048, 4096]),
        (10000, [0, 1968, 3936, 5904]),
        (34816, list(range(0, 30721, 2048))),
        (
            34817,
            [
                0,
                2048,
                4096,
                6144,
                8192,
                10240,
                12288,
                14336,
                16385,
                18433,
                20481,
                22529,
                24577,
                26625,
                28673,
                30721,
            ],
        ),
        (
            100000,
            [
                0,
                6394,
                12787,
                19181,
                25574,
                31968,
                38362,
                44755,
                51149,
                57542,
                63936,
                70330,
                76723,
                83117,
                89510,
                95904,
            ],
        ),
    ],
)
def test_verified_window_examples(length, expected):
    assert uniform_positions(length) == expected


def test_windows_cover_endpoints_without_partial_tails():
    for length in list(range(4097, 35000, 37)) + [10**6, 10**8]:
        starts = uniform_positions(length)
        assert 2 <= len(starts) <= 16
        assert starts == sorted(set(starts))
        assert starts[0] == 0 and starts[-1] == length - 4096
        assert all(0 <= start <= length - 4096 for start in starts)


def make_classifier(probability=None):
    clf = HyenaDNAClassifier.__new__(HyenaDNAClassifier)
    clf.device = "cpu"
    clf.min_contig_length = 1000
    clf.model = torch.nn.Linear(1, 1)
    clf.calls = []

    def infer(self, windows, model=None, tokenizer=None):
        self.calls.append(list(windows))
        return [
            (p >= 0.5, p)
            for p in (
                (
                    probability(w)
                    if callable(probability)
                    else (
                        probability
                        if probability is not None
                        else 0.8 if w.startswith("A") else 0.2
                    )
                )
                for w in windows
            )
        ]

    clf._predict_window_predictions = MethodType(infer, clf)
    return clf


def test_reference_batch_order_and_window_limit():
    sequences = [chr(65 + i % 26) * 100000 for i in range(65)]
    sequences.insert(1, "too short")
    clf = make_classifier()
    result = clf.predict_contigs(sequences)
    assert [len(call) for call in clf.calls] == [256, 256, 256, 256, 16]
    expected = [
        seq.upper()[p : p + 4096]
        for seq in sequences
        if len(seq) >= 1000
        for p in uniform_positions(len(seq))
    ]
    assert [w for call in clf.calls for w in call] == expected
    assert result[1]["num_windows"] == 0
    for seq, row in zip(sequences, result):
        if len(seq) >= 1000:
            assert row["num_windows"] == 16
            assert (
                row["eukaryote_prob"] == sum([0.8 if seq[0] == "A" else 0.2] * 16) / 16
            )
        assert row["length"] == len(seq)
        assert row["resampled"] is False


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "p,retained",
    [
        (math.nextafter(0.3, 0.0), False),
        (0.3, True),
        (math.nextafter(0.3, 1.0), True),
    ],
)
def test_inclusive_probability_threshold(p, retained, device):
    clf = make_classifier(p)
    clf.device = device
    result = clf.predict_contig("a" * 1000)
    assert result["eukaryote_prob"] == p
    assert (result["prediction"] == "eukaryote") is retained
    assert result["num_windows"] == 1


def test_arithmetic_probability_mean_not_voting_or_logits():
    clf = make_classifier()
    probs = [0.01, 0.01, 0.95]
    clf._predict_window_predictions = lambda windows: [(p >= 0.5, p) for p in probs]
    row = clf.predict_contig("A" * 8192)
    assert row["eukaryote_prob"] == sum(probs) / len(probs)
    assert row["prediction"] == "eukaryote"  # Two of three native votes are negative.
    assert row["num_windows"] == 3


def test_legacy_options_and_rng_cannot_change_sampling():
    clf = make_classifier(0.5)
    py_state, np_state = random.getstate(), np.random.get_state()
    sequence = "acgt" * 20000
    baseline = clf.predict_contig(sequence)
    other = clf.predict_contigs(
        [sequence],
        adaptive_stride=False,
        early_stopping=True,
        low_confidence_threshold=1,
        additional_random_windows=100,
    )[0]
    assert other == baseline
    assert other["num_windows"] == 16 and other["resampled"] is False
    assert random.getstate() == py_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state))
    assert all(w == w.upper() for call in clf.calls for w in call)


def test_minimum_length_and_empty_input_never_infer():
    clf = make_classifier()
    assert clf.predict_contigs([]) == []
    rows = clf.predict_contigs(["", "A" * 999])
    assert clf.calls == []
    assert all(
        r["num_windows"] == 0 and r["prediction"] == "non_eukaryote" for r in rows
    )
    clf.min_contig_length = 2000
    assert clf.predict_contig("A" * 1000)["num_windows"] == 0


def test_native_tokenizer_padding_truncation_and_unknowns():
    tok = CharacterTokenizer(
        characters=list("ACGTN"), model_max_length=4096, padding_side="left"
    )
    windows = ["acgtnRY", "A" * 4094 + "CG", "N" * 1000]
    encoded = _build_window_encoder(tok)(windows)
    native = tok(
        [w.upper() for w in windows],
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )["input_ids"]
    assert torch.equal(encoded, native)
    assert encoded.shape == (3, 4096)
    assert encoded[0, -9:].tolist() == [0, 7, 8, 9, 10, 11, 6, 6, 1]
    assert torch.all(encoded[0, :-9] == 4)
    assert encoded[1, 0] == 0 and encoded[1, -1] == 1
    assert torch.all(encoded[1, 1:-1] == 7)


@pytest.mark.parametrize("output_style", ["tensor", "tuple", "object"])
def test_native_binary_probability_operation(output_style):
    clf = make_classifier()
    clf.tokenizer = CharacterTokenizer(list("ACGTN"), 4096, padding_side="left")
    logits = torch.tensor([[1.0, 0.25], [0.1, 0.9]], dtype=torch.float32)

    def model(ids):
        assert not torch.is_grad_enabled()
        assert ids.shape == (2, 4096)
        return (
            (logits,)
            if output_style == "tuple"
            else SimpleNamespace(logits=logits) if output_style == "object" else logits
        )

    clf.model = model
    values = HyenaDNAClassifier._predict_window_predictions(
        clf, ["A" * 1000, "C" * 4096]
    )
    expected = torch.sigmoid(logits[:, 1] - logits[:, 0]).tolist()
    assert [p for _, p in values] == expected


def test_packaged_model_hash_single_context_and_loading_without_filename_hint(tmp_path):
    path = Path(predictor.__file__).parent / "models" / MODEL_FILENAME
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256
    renamed = tmp_path / "checkpoint.bin"
    renamed.write_bytes(path.read_bytes())
    with patch.object(torch, "load", return_value=torch.nn.Linear(1, 2)) as load:
        clf = HyenaDNAClassifier(model_path=str(renamed), device="cpu")
    assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": False}
    assert clf.model.training is False
    assert clf.use_dual_models is False
    assert clf.small_model is None and clf.large_model is None
    assert clf.window_size == clf.tokenizer.model_max_length == 4096


def test_corrupt_checkpoint_is_rejected(tmp_path):
    model = tmp_path / "bad.bin"
    model.write_bytes(b"wrong checkpoint")
    with pytest.raises(ClassifierPolicyError, match="SHA256"):
        HyenaDNAClassifier(model_path=str(model), device="cpu")


def test_auto_device_does_not_use_unvalidated_mps():
    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
        torch.backends.mps, "is_available", return_value=True
    ), patch.object(torch, "load", return_value=torch.nn.Linear(1, 2)):
        assert HyenaDNAClassifier().device == "cpu"
    with pytest.raises(ClassifierPolicyError, match="MPS"):
        HyenaDNAClassifier(device="mps")


@pytest.fixture
def tf32_enabled():
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("probability", [0.29999, 0.3, 0.30001, 0.31])
def test_selected_device_probabilities_are_used_without_rechecking(
    device, probability, tf32_enabled
):
    clf = make_classifier()
    clf.device = device
    seen = []

    def batch(self, sequences):
        assert self is clf
        seen.append((self.device, sequences))
        if device == "cuda":
            assert torch.backends.cuda.matmul.allow_tf32 is False
        return [dict(eukaryote_prob=p, num_windows=1) for p in [probability, 0.8]]

    sequences = ["A" * 1000, "C" * 1000]
    with patch.object(HyenaDNAClassifier, "_predict_batch", batch):
        result = clf.predict_contigs(sequences)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    assert [r["eukaryote_prob"] for r in result] == [probability, 0.8]
    assert seen == [(device, sequences)]


def test_cuda_precision_setting_is_restored_on_failure(tf32_enabled):
    clf = make_classifier()
    clf.device = "cuda"
    with patch.object(
        HyenaDNAClassifier,
        "_predict_batch",
        side_effect=RuntimeError("inference failed"),
    ):
        with pytest.raises(RuntimeError, match="inference failed"):
            clf.predict_contig("A" * 1000)
        assert torch.backends.cuda.matmul.allow_tf32 is True


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_probabilities_are_fatal(p):
    with pytest.raises(ClassifierPolicyError, match="invalid probability"):
        make_classifier(p).predict_contig("A" * 1000)
