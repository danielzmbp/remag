"""Keep a failed classification batch from duplicating or dropping records."""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from remag import features
from remag.utils import fasta_iter


def prediction(probability=0.8):
    return {"eukaryote_prob": probability, "confidence": 0.8, "num_windows": 1}


def filter_records(tmp_path, classifier, batch_size=2, count=2):
    fasta = tmp_path / "input.fa"
    fasta.write_text("".join(f">c{i}\nACGT\n" for i in range(1, count + 1)))
    with patch("remag.hyenadna_classifier.HyenaDNAClassifier", return_value=classifier):
        result = features.filter_bacterial_contigs(
            str(fasta),
            str(tmp_path / "out"),
            min_contig_length=1,
            hyenadna_batch_size=batch_size,
            save_filtered_contigs=True,
        )
    records = list(fasta_iter(result))
    with (tmp_path / "out/input_hyenadna_classification.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return records, rows


@pytest.mark.parametrize("batch_size", [2, 8], ids=["full-batch", "final-batch"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing-field",
        "bad-number",
        "too-few",
        "too-many",
        "iterator-error",
        "prediction-error",
    ],
)
def test_failed_batch_keeps_each_record_once(tmp_path, batch_size, failure):
    def predict(_sequences):
        good, bad = prediction(), prediction()
        if failure == "missing-field":
            del bad["confidence"]
        elif failure == "bad-number":
            bad["confidence"] = "invalid"
        elif failure == "too-few":
            return [good]
        elif failure == "too-many":
            return [good, good, good]
        elif failure == "iterator-error":

            def broken_iterator():
                yield good
                raise RuntimeError("prediction iteration failed")

            return broken_iterator()
        elif failure == "prediction-error":
            raise RuntimeError("prediction failed")
        return [good, bad]

    records, rows = filter_records(
        tmp_path, SimpleNamespace(predict_contigs=predict), batch_size
    )
    assert records == [("c1", "ACGT"), ("c2", "ACGT")]
    assert rows == []
    assert not (tmp_path / "out/input_non_eukaryotic.fasta").exists()


@pytest.mark.parametrize("batch_size", [2, 8])
def test_failed_batch_does_not_also_write_rejected_records(tmp_path, batch_size):
    classifier = SimpleNamespace(predict_contigs=lambda _: [prediction(0.2), {}])
    records, rows = filter_records(tmp_path, classifier, batch_size)
    assert records == [("c1", "ACGT"), ("c2", "ACGT")]
    assert rows == []
    assert not (tmp_path / "out/input_non_eukaryotic.fasta").exists()


def test_earlier_successful_batch_is_preserved(tmp_path):
    batches = iter([[prediction(), prediction()], [prediction(), {}]])
    records, rows = filter_records(
        tmp_path, SimpleNamespace(predict_contigs=lambda _: next(batches)), count=4
    )
    assert records == [(f"c{i}", "ACGT") for i in range(1, 5)]
    assert [row["contig_id"] for row in rows] == ["c1", "c2"]


def test_write_failure_stops_without_repeating_records(tmp_path, monkeypatch):
    original_writer = features._write_fasta_record
    calls = []

    def fail_second_record(handle, header, sequence):
        calls.append(header)
        if len(calls) == 2:
            raise OSError("cannot write sequence")
        original_writer(handle, header, sequence)

    monkeypatch.setattr(features, "_write_fasta_record", fail_second_record)
    classifier = SimpleNamespace(predict_contigs=lambda _: [prediction(), prediction()])
    with pytest.raises(OSError, match="cannot write sequence"):
        filter_records(tmp_path, classifier)
    assert calls == ["c1", "c2"]
    output = tmp_path / "out/input_eukaryotic_filtered.fasta"
    assert not output.exists()
    assert Path(str(output) + ".tmp").read_text() == ">c1\nACGT\n"
