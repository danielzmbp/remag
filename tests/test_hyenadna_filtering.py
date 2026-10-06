"""Tests for HyenaDNA FASTA filtering decisions."""

import csv
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from remag.features import HYENADNA_PREDICTION_POLICY, filter_bacterial_contigs


class DummyHyenaClassifier:
    def __init__(self):
        self.results = [
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.45,
                "confidence": 0.5,
                "num_windows": 2,
                "resampled": True,
            },
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.44,
                "confidence": 0.5,
                "num_windows": 2,
                "resampled": True,
            },
        ]

    def predict_contigs(self, sequences):
        return self.results[: len(sequences)]


@pytest.mark.parametrize("batched", [False, True])
def test_filter_keeps_contigs_at_recall_friendly_threshold(tmp_path, batched):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        ">keep_me\nACGTACGT\n" ">filter_me\nTGCATGCA\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    classifier = DummyHyenaClassifier()
    if not batched:
        results = iter(classifier.results)
        classifier = SimpleNamespace(predict_contig=lambda _: next(results))

    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier",
        return_value=classifier,
    ):
        filtered_fasta = filter_bacterial_contigs(
            str(fasta),
            str(output),
            min_contig_length=1,
            hyenadna_batch_size=8,
            save_filtered_contigs=True,
        )

    assert "keep_me" in open(filtered_fasta, encoding="utf-8").read()
    assert "filter_me" not in open(filtered_fasta, encoding="utf-8").read()

    non_eukaryotic_fasta = output / "contigs_non_eukaryotic.fasta"
    assert "filter_me" in non_eukaryotic_fasta.read_text(encoding="utf-8")

    results_tsv = output / "contigs_hyenadna_classification.tsv"
    with results_tsv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert rows[0]["prediction"] == "eukaryote"
    assert rows[0]["eukaryote_prob"] == "0.4500"
    assert rows[1]["prediction"] == "non_eukaryote"
    assert rows[1]["eukaryote_prob"] == "0.4400"


@pytest.mark.parametrize(
    "policy", [None, "legacy", "invalid_json", HYENADNA_PREDICTION_POLICY]
)
def test_cached_filtering_identifies_older_or_unknown_policy(tmp_path, policy):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">c1\nAAAA\n")
    output = tmp_path / "out"
    output.mkdir()
    cached = output / "input_eukaryotic_filtered.fasta"
    cached.write_text(">old\nCCCC\n")
    params = output / "params.json"
    if policy is not None:
        params.write_text(
            "{"
            if policy == "invalid_json"
            else json.dumps({"hyenadna_prediction_policy": policy})
        )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with patch("remag.hyenadna_classifier.HyenaDNAClassifier") as initialize, patch(
        "remag.features.logger.warning"
    ) as warning:
        assert filter_bacterial_contigs(str(fasta), str(output)) == str(cached)
    initialize.assert_not_called()
    assert warning.call_count == (0 if policy == HYENADNA_PREDICTION_POLICY else 1)
    if warning.called:
        assert "prediction is not rerun" in warning.call_args.args[0]
        assert "--force" in warning.call_args.args[0]
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("skip_filter", [False, True])
def test_new_run_parameters_record_prediction_policy(tmp_path, skip_filter):
    from remag.core import main

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">c1\nAAAA\n")
    output = tmp_path / "out"
    args = Namespace(
        fasta=str(fasta),
        output=str(output),
        min_contig_length=1,
        bam=None,
        tsv=None,
        force=False,
        verbose=False,
        filter_only=True,
        skip_bacterial_filter=skip_filter,
    )
    with patch("remag.core.filter_bacterial_contigs", return_value=str(fasta)):
        main(args)
    saved = json.loads((output / "params.json").read_text())
    assert saved["hyenadna_prediction_policy"] == (
        None if skip_filter else HYENADNA_PREDICTION_POLICY
    )
