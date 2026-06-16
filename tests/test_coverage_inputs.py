"""Tests for precomputed coverage input formats."""

import gzip
from collections import OrderedDict
from math import isclose, sqrt
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from remag.cli import main_cli
from remag.features import calculate_coverage_from_tsv, get_features


def example_fragments_dict():
    return OrderedDict(
        {
            "ctg0": {
                "sequence": "A" * 100,
                "fragments": ["ctg0.original", "ctg0.0", "ctg0.1"],
                "fragment_info": {
                    "ctg0.original": {"start_pos": 0, "length": 100},
                    "ctg0.0": {"start_pos": 10, "length": 20},
                    "ctg0.1": {"start_pos": 5, "length": 10},
                },
            },
            "ctg1": {
                "sequence": "C" * 50,
                "fragments": ["ctg1.original"],
                "fragment_info": {
                    "ctg1.original": {"start_pos": 0, "length": 50},
                },
            },
        }
    )


def test_contig_level_tsv_assigns_same_coverage_to_fragments(tmp_path):
    coverage_file = tmp_path / "sample.tsv"
    coverage_file.write_text("ctg0\t7.5\nctg1\t2.0\n", encoding="utf-8")

    coverage_df = calculate_coverage_from_tsv(
        [str(coverage_file)], example_fragments_dict()
    )

    assert list(coverage_df.columns) == ["sample"]
    assert coverage_df.loc["ctg0.original", "sample"] == 7.5
    assert coverage_df.loc["ctg0.0", "sample"] == 7.5
    assert coverage_df.loc["ctg0.1", "sample"] == 7.5
    assert coverage_df.loc["ctg1.original", "sample"] == 2.0


def test_interval_coverage_computes_fragment_specific_mean_and_std(tmp_path):
    coverage_file = tmp_path / "sample.bam.cov.gz"
    with gzip.open(coverage_file, "wt", encoding="utf-8") as handle:
        handle.write(
            "track type=bedGraph name=sample\n"
            "ctg0\t0\t10\t1\n"
            "ctg0\t10\t30\t3\n"
            "ctg1\t0\t50\t4\n"
        )

    coverage_df = calculate_coverage_from_tsv(
        [str(coverage_file)], example_fragments_dict()
    )

    assert list(coverage_df.columns) == ["sample_coverage", "sample_coverage_std"]
    assert isclose(coverage_df.loc["ctg0.original", "sample_coverage"], 0.7)
    assert isclose(
        coverage_df.loc["ctg0.original", "sample_coverage_std"], sqrt(1.41)
    )
    assert coverage_df.loc["ctg0.0", "sample_coverage"] == 3.0
    assert coverage_df.loc["ctg0.0", "sample_coverage_std"] == 0.0
    assert coverage_df.loc["ctg0.1", "sample_coverage"] == 2.0
    assert coverage_df.loc["ctg0.1", "sample_coverage_std"] == 1.0
    assert coverage_df.loc["ctg1.original", "sample_coverage"] == 4.0
    assert coverage_df.loc["ctg1.original", "sample_coverage_std"] == 0.0


def test_get_features_uses_interval_coverage_for_augmented_fragments(tmp_path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">ctg0\n" + "A" * 120 + "\n", encoding="utf-8")

    coverage_file = tmp_path / "sample.cov.gz"
    with gzip.open(coverage_file, "wt", encoding="utf-8") as handle:
        handle.write(
            "ctg0\t0\t40\t1\n"
            "ctg0\t40\t80\t5\n"
            "ctg0\t80\t120\t9\n"
        )

    features_df, _ = get_features(
        str(fasta),
        bam_files=None,
        tsv_files=[str(coverage_file)],
        output_dir=str(tmp_path),
        min_contig_length=20,
        cores=1,
        num_augmentations=4,
        args=SimpleNamespace(coverage_batch_size=100000, keep_intermediate=False),
    )

    assert "sample_coverage" in features_df.columns
    assert "sample_coverage_std" in features_df.columns
    assert "ctg0.original" in features_df.index
    assert any(fragment != "ctg0.original" for fragment in features_df.index)
    assert features_df["sample_coverage"].notna().all()
    assert features_df["sample_coverage_std"].notna().all()


def test_cli_accepts_interval_coverage_as_precomputed_input(tmp_path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">ctg0\nATGC\n", encoding="utf-8")
    coverage_file = tmp_path / "sample.bam.cov.gz"
    with gzip.open(coverage_file, "wt", encoding="utf-8") as handle:
        handle.write("ctg0\t0\t4\t1\n")

    runner = CliRunner()
    with patch("remag.cli.run_remag") as mock_run_remag:
        result = runner.invoke(main_cli, [str(fasta), "-c", str(coverage_file)])

    assert result.exit_code == 0, result.output
    args = mock_run_remag.call_args[0][0]
    assert args.bam is None
    assert args.tsv == [str(coverage_file)]


def test_cli_rejects_mixed_alignment_and_precomputed_coverage(tmp_path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">ctg0\nATGC\n", encoding="utf-8")
    bam = tmp_path / "sample.bam"
    bam.touch()
    coverage_file = tmp_path / "sample.cov.gz"
    with gzip.open(coverage_file, "wt", encoding="utf-8") as handle:
        handle.write("ctg0\t0\t4\t1\n")

    runner = CliRunner()
    result = runner.invoke(main_cli, [str(fasta), "-c", str(bam), "-c", str(coverage_file)])

    assert result.exit_code != 0
    assert "Cannot mix BAM/CRAM files" in result.output
    assert "precomputed coverage files" in result.output
