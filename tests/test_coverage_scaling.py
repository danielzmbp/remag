"""Regression tests for filename-independent coverage scaling."""

import gzip
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from remag.features import get_features


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "contigs.fasta"
    path.write_text(
        ">c0\n" + "ACGT" * 30 + "\n>c1\n" + "AAAA" * 30 + "\n>c2\n" + "CCCC" * 30 + "\n"
    )
    return path


def features(fasta, output, coverage_files=None):
    output.mkdir()
    return get_features(
        str(fasta),
        bam_files=None,
        tsv_files=coverage_files,
        output_dir=str(output),
        min_contig_length=20,
        cores=1,
        num_augmentations=0,
        args=SimpleNamespace(keep_intermediate=False),
    )[0]


@pytest.mark.parametrize(
    "filename",
    [
        "sample.tsv",
        "coverage.tsv",
        "sample.tsv.gz",
        "sample.bedgraph",
        "sample.bedgraph.gz",
    ],
)
def test_coverage_scaling_does_not_depend_on_filename(tmp_path, fasta, filename):
    coverage_file = tmp_path / filename
    if "bedgraph" in filename:
        content = "c0\t0\t120\t0\nc1\t0\t120\t9\nc2\t0\t120\t99\n"
    else:
        content = "c0\t0\nc1\t9\nc2\t99\n"
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(coverage_file, "wt", encoding="utf-8") as handle:
        handle.write(content)

    actual = features(fasta, tmp_path / "with_coverage", [str(coverage_file)])
    kmer_only = features(fasta, tmp_path / "without_coverage")

    pd.testing.assert_frame_equal(actual.iloc[:, :136], kmer_only)
    np.testing.assert_allclose(actual.iloc[:, 136], [0.0, 0.5, 1.0], atol=1e-12)
    if "bedgraph" in filename:
        np.testing.assert_array_equal(actual.iloc[:, 137], 0.0)


def test_each_coverage_sample_is_scaled_independently(tmp_path, fasta):
    sample1 = tmp_path / "sample1.tsv"
    sample2 = tmp_path / "sample2_coverage.tsv"
    sample1.write_text("c0\t0\nc1\t9\nc2\t99\n")
    sample2.write_text("c0\t15\nc1\t3\nc2\t0\n")

    actual = features(fasta, tmp_path / "output", [str(sample1), str(sample2)])

    np.testing.assert_allclose(actual["sample1"], [0.0, 0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(actual["sample2_coverage"], [1.0, 0.5, 0.0], atol=1e-12)
