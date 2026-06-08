"""Tests for HyenaDNA FASTA filtering decisions."""

import csv
from unittest.mock import patch

from remag.features import filter_bacterial_contigs


class DummyHyenaClassifier:
    def __init__(self):
        self.results = [
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.46,
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


def test_filter_keeps_contigs_at_recall_friendly_threshold(tmp_path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        ">keep_me\nACGTACGT\n" ">filter_me\nTGCATGCA\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"

    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier",
        return_value=DummyHyenaClassifier(),
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
    assert rows[0]["eukaryote_prob"] == "0.4600"
    assert rows[1]["prediction"] == "non_eukaryote"
    assert rows[1]["eukaryote_prob"] == "0.4400"
