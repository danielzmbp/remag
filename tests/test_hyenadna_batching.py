"""Tests for batched HyenaDNA classification helpers."""

import math
from types import MethodType, SimpleNamespace

import pytest

from remag.hyenadna_classifier import predictor
from remag.hyenadna_classifier.predictor import HyenaDNAClassifier


def _make_dummy_classifier(batch_size=16, euk_prob=0.8, non_euk_prob=0.2):
    classifier = HyenaDNAClassifier.__new__(HyenaDNAClassifier)
    classifier.batch_size = batch_size
    classifier.min_contig_length = 1
    classifier.use_dual_models = False
    classifier.length_threshold = 4096
    classifier.device = "cpu"
    classifier.model = object()
    classifier.tokenizer = SimpleNamespace(model_max_length=4)
    classifier.window_size = 4
    classifier.stride = 4

    def window_prob(window):
        return euk_prob if window.startswith("A") else non_euk_prob

    def predict_window_stats(self, windows, model=None, tokenizer=None):
        probs = [window_prob(window) for window in windows]
        return sum(prob >= 0.5 for prob in probs), sum(probs), len(probs)

    def predict_window_predictions(self, windows, model=None, tokenizer=None):
        return [(prob >= 0.5, prob) for prob in (window_prob(w) for w in windows)]

    classifier._predict_window_stats = MethodType(predict_window_stats, classifier)
    classifier._predict_window_predictions = MethodType(
        predict_window_predictions, classifier
    )
    return classifier


def test_predict_contigs_matches_single_contig_predictions():
    classifier = _make_dummy_classifier()
    sequences = [
        "AAAAAAAAAAAA",
        "CCCCCCCCCCCC",
        "AAAACCCCAAAA",
    ]

    single_results = [classifier.predict_contig(sequence) for sequence in sequences]
    batched_results = classifier.predict_contigs(sequences)

    for batched, single in zip(batched_results, single_results):
        assert batched["prediction"] == single["prediction"]
        assert batched["eukaryote_prob"] == pytest.approx(single["eukaryote_prob"])
        assert batched["confidence"] == pytest.approx(single["confidence"])
        assert batched["num_windows"] == single["num_windows"]
        assert batched["length"] == single["length"]
        assert batched["resampled"] == single["resampled"]
        assert batched["model_used"] == single["model_used"]


def test_predict_contigs_falls_back_for_huge_contigs():
    classifier = _make_dummy_classifier(batch_size=2)
    sequences = ["AAAAAAAAAAAA"]

    results = classifier.predict_contigs(sequences)

    assert results == [classifier.predict_contig(sequences[0])]


def test_two_window_borderline_contigs_are_resampled():
    classifier = _make_dummy_classifier(euk_prob=0.54, non_euk_prob=0.44)

    result = classifier.predict_contig("AACCCC")

    assert result["resampled"] is True
    assert result["num_windows"] > 2


def _scripted_classifier(monkeypatch, probabilities, batch_size, extra_prob=0.9):
    """Track full-pass ordering and bound the number of live sliding windows."""
    classifier = _make_dummy_classifier(batch_size=batch_size)
    state = {"scored": 0, "sampling_calls": [], "batch_sizes": []}
    values = {f"initial_{i}": prob for i, prob in enumerate(probabilities)}

    def sliding(sequence, window_size, stride):
        for i in range(len(probabilities)):
            assert i < state["scored"] + batch_size
            yield f"initial_{i}"

    def random_windows(sequence, window_size, count, seed=42):
        assert seed == 42
        assert state["scored"] == len(probabilities)
        state["sampling_calls"].append(count)
        windows = [f"random_{i}" for i in range(count)]
        values.update({window: extra_prob for window in windows})
        return windows

    def predictions(self, windows, model=None, tokenizer=None):
        assert len(windows) <= batch_size
        state["batch_sizes"].append(len(windows))
        state["scored"] += sum(w.startswith("initial_") for w in windows)
        return [(values[w] >= 0.5, values[w]) for w in windows]

    def stats(self, windows, model=None, tokenizer=None):
        scores = predictions(self, windows, model, tokenizer)
        return sum(vote for vote, _ in scores), sum(p for _, p in scores), len(scores)

    monkeypatch.setattr(predictor, "sliding_window", sliding)
    monkeypatch.setattr(
        predictor, "estimate_window_count", lambda *_: len(probabilities)
    )
    monkeypatch.setattr(predictor, "generate_random_windows", random_windows)
    classifier._predict_window_predictions = MethodType(predictions, classifier)
    classifier._predict_window_stats = MethodType(stats, classifier)
    return classifier, state


def _predict_mode(classifier, mode, length, **kwargs):
    sequence = "A" * length
    if mode == "single":
        return classifier.predict_contig(sequence, **kwargs)
    return classifier.predict_contigs([sequence], **kwargs)[0]


@pytest.mark.parametrize("mode", ["single", "grouped", "streaming"])
@pytest.mark.parametrize("early_stopping", [False, True])
@pytest.mark.parametrize(
    "probs,threshold",
    [
        ([0.9] * 3 + [0.1] * 9, 0.75),  # Old 90% agreement break.
        ([0.9, 0.9, 0.1] + [0.1] * 9, 0.75),  # Old intermediate resampling/85% break.
        ([0.9, 0.9, 0.1] * 2 + [0.9] * 3 + [0.1] * 3, 0.5),  # Old majority break.
    ],
)
def test_all_initial_windows_precede_sampling(
    monkeypatch, mode, early_stopping, probs, threshold
):
    classifier, state = _scripted_classifier(
        monkeypatch, probs, batch_size=64 if mode == "grouped" else 3
    )
    result = _predict_mode(
        classifier,
        mode,
        5000,
        early_stopping=early_stopping,
        low_confidence_threshold=threshold,
    )
    assert state["scored"] == len(probs)
    assert state["sampling_calls"] == []
    assert result["num_windows"] == len(probs)
    assert result["eukaryote_prob"] == pytest.approx(sum(probs) / len(probs))
    assert result["resampled"] is False


@pytest.mark.parametrize("mode", ["single", "grouped", "streaming"])
@pytest.mark.parametrize("length,count", [(2000, 16), (5000, 8)])
@pytest.mark.parametrize("early_stopping", [False, True])
def test_low_confidence_sampling_is_retained_once(
    monkeypatch, mode, length, count, early_stopping
):
    probs = [0.1, 0.6, 0.6] * 2
    classifier, state = _scripted_classifier(
        monkeypatch, probs, batch_size=64 if mode == "grouped" else 3
    )
    result = _predict_mode(classifier, mode, length, early_stopping=early_stopping)
    assert state["sampling_calls"] == [count]
    assert result["num_windows"] == len(probs) + count
    assert result["eukaryote_prob"] == pytest.approx(
        (sum(probs) + count * 0.9) / (len(probs) + count)
    )
    assert result["resampled"] is True


@pytest.mark.parametrize("mode", ["single", "grouped", "streaming"])
@pytest.mark.parametrize("length,count", [(2000, 10), (5000, 6)])
@pytest.mark.parametrize(
    "prob,resample",
    [
        (0.4499, False),
        (0.45, True),
        (0.5, True),
        (math.nextafter(0.55, 0.0), True),
        (0.55, False),
        (0.5501, False),
    ],
)
def test_two_window_sampling_preserves_native_band(
    monkeypatch, mode, length, count, prob, resample
):
    # Keep the exact native abs(p - .5) <= .05 comparison: the float literal
    # .55 is just outside it; nextafter is the last included upper value.
    classifier, state = _scripted_classifier(
        monkeypatch, [prob, prob], batch_size=64 if mode == "grouped" else 1
    )
    result = _predict_mode(classifier, mode, length)
    assert state["sampling_calls"] == ([count] if resample else [])
    assert result["num_windows"] == 2 + (count if resample else 0)
    assert result["resampled"] is resample


@pytest.mark.parametrize("mode", ["single", "grouped", "streaming"])
@pytest.mark.parametrize(
    "probs,length", [([0.9] * 6, 5000), ([0.5], 4), ([0.1, 0.6, 0.6], 4), ([], 5000)]
)
def test_no_sampling_and_fallback_cases(monkeypatch, mode, probs, length):
    classifier, state = _scripted_classifier(
        monkeypatch, probs, batch_size=64 if mode == "grouped" else 1
    )
    result = _predict_mode(classifier, mode, length)
    assert state["sampling_calls"] == []
    assert result["resampled"] is False
    assert result["num_windows"] == len(probs)
    if probs:
        assert result["prediction"] == (
            "eukaryote" if sum(probs) / len(probs) >= 0.5 else "non_eukaryote"
        )
    else:
        assert result["prediction"] == "non_eukaryote"
        assert result["eukaryote_prob"] == result["confidence"] == 0.0


@pytest.mark.parametrize("length,count", [(2000, 6), (5000, 3)])
def test_configurable_resampling_arguments(monkeypatch, length, count):
    classifier, state = _scripted_classifier(monkeypatch, [0.9, 0.1, 0.9], 2)
    result = classifier.predict_contig(
        "A" * length, low_confidence_threshold=0.8, additional_random_windows=3
    )
    assert state["sampling_calls"] == [count]
    assert result["num_windows"] == 3 + count
