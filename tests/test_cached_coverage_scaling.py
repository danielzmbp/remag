"""Regression tests for coverage added to cached features."""

import gzip
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from remag.features import CoverageError, get_features


def features(fasta, output, bam_files=None, tsv_files=None):
    output.mkdir(exist_ok=True)
    return get_features(
        str(fasta),
        bam_files=bam_files,
        tsv_files=tsv_files,
        output_dir=str(output),
        min_contig_length=20,
        cores=1,
        num_augmentations=0,
        args=SimpleNamespace(keep_intermediate=True),
    )[0]


@pytest.fixture
def cached_input(tmp_path):
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(
        "".join(f">c{i}\n" + base * 120 + "\n" for i, base in enumerate("ACG"))
    )
    output = tmp_path / "cached"
    kmer_only = features(fasta, output)
    return fasta, output, kmer_only


@pytest.mark.parametrize(
    "filename", ["sample.tsv", "sample.tsv.gz", "sample.cov", "sample.cov.gz"]
)
def test_added_coverage_matches_fresh_features(tmp_path, cached_input, filename):
    fasta, output, _ = cached_input
    coverage = tmp_path / filename
    content = (
        "c0\t0\t120\t0\nc1\t60\t120\t18\nc2\t0\t120\t99\n"
        if ".cov" in filename
        else "c0\t0\nc1\t9\nc2\t99\n"
    )
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(coverage, "wt", encoding="utf-8") as handle:
        handle.write(content)
    inputs = {"tsv_files": [str(coverage)]}
    fresh = features(fasta, tmp_path / "fresh", **inputs)

    with patch(
        "remag.features._calculate_kmer_composition",
        side_effect=AssertionError("K-mers must be reused"),
    ):
        recalculated = features(fasta, output, **inputs)
    pd.testing.assert_frame_equal(recalculated, fresh, atol=1e-12, rtol=1e-12)

    with patch(
        "remag.features.TSVCoverageCalculator.calculate_coverage",
        side_effect=AssertionError("Coverage must be reused"),
    ):
        reused = features(fasta, output, **inputs)
    pd.testing.assert_frame_equal(reused, recalculated, atol=1e-12, rtol=1e-12)


def test_added_bam_coverage_matches_fresh_features(tmp_path, cached_input):
    fasta, output, kmer_only = cached_input
    raw = pd.DataFrame(
        {"sample_coverage": [0.0, 9.0, 99.0], "sample_coverage_std": [0.0, 3.0, 15.0]},
        index=kmer_only.index,
    )
    inputs = {"bam_files": [str(tmp_path / "sample.bam")]}
    with patch(
        "remag.features.BAMCoverageCalculator.calculate_coverage", return_value=raw
    ) as calculator:
        fresh = features(fasta, tmp_path / "fresh", **inputs)
        with patch(
            "remag.features._calculate_kmer_composition",
            side_effect=AssertionError("K-mers must be reused"),
        ):
            recalculated = features(fasta, output, **inputs)
    assert calculator.call_count == 2
    pd.testing.assert_frame_equal(recalculated, fresh, atol=1e-12, rtol=1e-12)


def test_empty_added_coverage_preserves_cached_kmers(tmp_path, cached_input):
    fasta, output, kmer_only = cached_input
    coverage = tmp_path / "sample.tsv"
    coverage.write_text("")
    with patch(
        "remag.features._calculate_kmer_composition",
        side_effect=AssertionError("K-mers must be reused"),
    ):
        actual = features(fasta, output, tsv_files=[str(coverage)])
    pd.testing.assert_frame_equal(actual, kmer_only, atol=1e-12, rtol=1e-12)


def test_recalculated_coverage_uses_cached_feature_rows(tmp_path, cached_input):
    fasta, output, kmer_only = cached_input
    retained = kmer_only.iloc[:2]
    retained.to_csv(output / "features.csv")
    coverage = tmp_path / "sample.tsv"
    coverage.write_text("c0\t0\nc1\t9\nc2\t99\n")
    with patch(
        "remag.features._calculate_kmer_composition",
        side_effect=AssertionError("K-mers must be reused"),
    ):
        actual = features(fasta, output, tsv_files=[str(coverage)])
    pd.testing.assert_frame_equal(
        actual.iloc[:, :136], retained, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(actual["sample"], [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    "old_names, new_names",
    [
        (["sampleA.tsv"], ["sampleA.tsv", "sampleB.tsv"]),
        (["sampleA.tsv"], ["sampleB.tsv"]),
        (["sampleA.tsv"], ["sampleB.cov"]),
        (["sampleA.cov"], ["sampleB.tsv"]),
        (["sampleA_coverage.tsv"], ["sampleB.tsv"]),
        (["sampleA.tsv"], ["empty.tsv"]),
    ],
    ids=[
        "add-sample",
        "replace-sample",
        "plain-to-interval",
        "interval-to-plain",
        "coverage-name",
        "empty-replacement",
    ],
)
def test_recalculation_replaces_all_cached_coverage(
    tmp_path, cached_input, old_names, new_names
):
    fasta, output, kmer_only = cached_input
    for name in set(old_names + new_names):
        content = "c0\t0\nc1\t9\nc2\t99\n"
        if name.endswith(".cov"):
            content = "c0\t0\t120\t0\nc1\t0\t120\t9\nc2\t0\t120\t99\n"
        elif name == "empty.tsv":
            content = ""
        (tmp_path / name).write_text(content)
    features(fasta, output, tsv_files=[str(tmp_path / name) for name in old_names])
    inputs = {"tsv_files": [str(tmp_path / name) for name in new_names]}
    fresh = features(fasta, tmp_path / "fresh", **inputs)

    with patch(
        "remag.features._calculate_kmer_composition",
        side_effect=AssertionError("K-mers must be reused"),
    ):
        actual = features(fasta, output, **inputs)

    assert actual.columns.is_unique
    pd.testing.assert_frame_equal(actual, fresh, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(
        actual.iloc[:, :136], kmer_only, atol=1e-12, rtol=1e-12
    )
    saved = pd.read_csv(output / "features.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, fresh, atol=1e-12, rtol=1e-12)


def test_recalculation_removes_columns_duplicated_by_older_runs(tmp_path, cached_input):
    fasta, output, kmer_only = cached_input
    cached = kmer_only.copy()
    cached["old_sample"] = [0.0, 0.5, 1.0]
    cached["old_sample.1"] = [0.0, 0.5, 1.0]
    cached.to_csv(output / "features.csv")
    coverage = tmp_path / "new_sample.tsv"
    coverage.write_text("c0\t0\nc1\t9\nc2\t99\n")
    inputs = {"tsv_files": [str(coverage)]}
    fresh = features(fasta, tmp_path / "fresh", **inputs)

    actual = features(fasta, output, **inputs)

    pd.testing.assert_frame_equal(actual, fresh, atol=1e-12, rtol=1e-12)


def test_failed_coverage_preserves_cache_without_regenerating(tmp_path, cached_input):
    fasta, output, _ = cached_input
    before = (output / "features.csv").read_bytes()
    with (
        patch("remag.features.pysam.AlignmentFile", side_effect=MemoryError()),
        patch("remag.features._calculate_kmer_composition") as kmers,
    ):
        with pytest.raises(CoverageError, match="reduce --cores"):
            features(fasta, output, bam_files=[str(tmp_path / "sample.bam")])
    kmers.assert_not_called()
    assert (output / "features.csv").read_bytes() == before
