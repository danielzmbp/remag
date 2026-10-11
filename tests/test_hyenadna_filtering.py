"""Tests for HyenaDNA FASTA filtering decisions."""

import csv
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from remag.features import HYENADNA_PREDICTION_POLICY, filter_bacterial_contigs
from remag.utils import fasta_iter


class DummyHyenaClassifier:
    def __init__(self):
        self.results = [
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.30,
                "confidence": 0.5,
                "num_windows": 2,
                "resampled": False,
            },
            {
                "prediction": "non_eukaryote",
                "eukaryote_prob": 0.29,
                "confidence": 0.5,
                "num_windows": 2,
                "resampled": False,
            },
        ]

    def predict_contigs(self, sequences):
        return self.results[: len(sequences)]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("save_rejected", [False, True])
@pytest.mark.parametrize("length", [4, 1000])
def test_filter_preserves_empty_output_when_no_contigs_pass(
    tmp_path, batched, save_rejected, length
):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">c1\n" + "A" * length + "\n")
    output = tmp_path / "out"
    result = {
        "eukaryote_prob": 0.1,
        "confidence": 1.0,
        "num_windows": 1,
    }
    classifier = SimpleNamespace(
        **(
            {"predict_contigs": lambda sequences: [result] * len(sequences)}
            if batched
            else {"predict_contig": lambda sequence: result}
        )
    )
    with patch("remag.hyenadna_classifier.HyenaDNAClassifier", return_value=classifier):
        filtered = filter_bacterial_contigs(
            str(fasta), str(output), save_filtered_contigs=save_rejected
        )
    assert filtered != str(fasta)
    assert Path(filtered).read_bytes() == b""
    with (output / "contigs_hyenadna_classification.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == (1 if length >= 1000 else 0)
    if rows:
        assert rows[0]["prediction"] == "non_eukaryote"
    rejected = output / "contigs_non_eukaryotic.fasta"
    if save_rejected and length >= 1000:
        assert list(fasta_iter(str(rejected))) == list(fasta_iter(str(fasta)))
    else:
        assert not rejected.exists()
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("filter_only", [False, True])
def test_no_retained_contigs_finish_successfully_before_binning(
    tmp_path, cached, filter_only
):
    from remag.core import main

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">c1\n" + "A" * 1000 + "\n")
    output = tmp_path / "out"
    args = Namespace(
        fasta=str(fasta),
        output=str(output),
        min_contig_length=1000,
        bam=None,
        tsv=None,
        force=False,
        verbose=False,
        filter_only=filter_only,
        skip_bacterial_filter=False,
    )
    if cached:
        output.mkdir()
        (output / "input_eukaryotic_filtered.fasta").write_bytes(b"")
        (output / "params.json").write_text(
            json.dumps(
                {
                    "min_contig_length": 1000,
                    "hyenadna_prediction_policy": HYENADNA_PREDICTION_POLICY,
                }
            )
        )
    classifier = SimpleNamespace(
        predict_contigs=lambda sequences: [
            {"eukaryote_prob": 0.1, "confidence": 1.0, "num_windows": 1}
            for sequence in sequences
        ]
    )
    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier", return_value=classifier
    ) as initialize, patch("remag.core.get_features") as features, patch(
        "remag.core.train_siamese_network"
    ) as train, patch(
        "remag.core.cluster_contigs"
    ) as cluster, patch(
        "remag.core.save_clusters_as_fasta"
    ) as export:
        assert main(args) is None
    assert initialize.call_count == (0 if cached else 1)
    features.assert_not_called()
    train.assert_not_called()
    cluster.assert_not_called()
    export.assert_not_called()
    assert (output / "input_eukaryotic_filtered.fasta").read_bytes() == b""
    assert not (output / "bins.csv").exists()
    assert not (output / "bins").exists()


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
    assert float(rows[0]["eukaryote_prob"]) == 0.30
    assert rows[1]["prediction"] == "non_eukaryote"
    assert float(rows[1]["eukaryote_prob"]) == 0.29


@pytest.mark.parametrize(
    "policy",
    [
        None,
        "legacy",
        HYENADNA_PREDICTION_POLICY.replace("uniform16_v2:", "uniform16_v1:"),
        "invalid_json",
        HYENADNA_PREDICTION_POLICY,
    ],
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
    from remag.filter_policy import HYENADNA_FILTER_CONFIG

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
    assert saved["hyenadna_filter"] == (None if skip_filter else HYENADNA_FILTER_CONFIG)


def test_filter_keeps_headers_case_and_order_with_reference_batches(tmp_path):
    from remag.filter_policy import INFERENCE_BATCH_SIZE

    fasta = tmp_path / "mixed.fa"
    records = [
        (f"c{i} description {i}", ("aCnRy" if i % 2 else "TgNac") * 200)
        for i in range(130)
    ]
    fasta.write_text("".join(f">{h}\n{s}\n" for h, s in records))
    seen = []

    def predict(sequences):
        seen.append(sequences)
        return [
            dict(
                eukaryote_prob=0.7 if s[0] == "A" else 0.1,
                confidence=1.0,
                num_windows=1,
                resampled=False,
            )
            for s in sequences
        ]

    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier",
        return_value=SimpleNamespace(predict_contigs=predict),
    ) as initialize:
        result = filter_bacterial_contigs(
            str(fasta), str(tmp_path / "out"), save_filtered_contigs=True
        )
    assert [len(s) for s in seen] == [64, 64, 2]
    assert [s for batch in seen for s in batch] == [s.upper() for _, s in records]
    assert list(fasta_iter(result)) == records[1::2]
    assert (
        list(fasta_iter(str(tmp_path / "out/mixed_non_eukaryotic.fasta")))
        == records[::2]
    )
    assert initialize.call_args.kwargs == dict(
        device="auto",
        min_contig_length=1000,
        batch_size=INFERENCE_BATCH_SIZE,
        use_dual_models=False,
        window_size=4096,
    )


def test_invalid_prediction_never_restores_input(tmp_path):
    from remag.filter_policy import ClassifierPolicyError

    fasta = tmp_path / "input.fa"
    fasta.write_text(">a\n" + "a" * 1000 + "\n")
    output = tmp_path / "out"

    def failed(_):
        raise ClassifierPolicyError("invalid probability")

    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier",
        return_value=SimpleNamespace(predict_contigs=failed),
    ):
        with pytest.raises(ClassifierPolicyError, match="invalid probability"):
            filter_bacterial_contigs(
                str(fasta), str(output), save_filtered_contigs=True
            )
    assert not (output / "input_eukaryotic_filtered.fasta").exists()
    assert (output / "input_eukaryotic_filtered.fasta.tmp").read_bytes() == b""


def test_model_policy_failure_does_not_disable_filtering(tmp_path):
    from remag.filter_policy import ClassifierPolicyError

    with patch(
        "remag.hyenadna_classifier.HyenaDNAClassifier",
        side_effect=ClassifierPolicyError("checkpoint SHA256"),
    ):
        with pytest.raises(ClassifierPolicyError, match="SHA256"):
            filter_bacterial_contigs("input.fa", str(tmp_path / "out"))
