"""Tests for batched HyenaDNA classification helpers."""

from types import MethodType, SimpleNamespace

import pytest

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
