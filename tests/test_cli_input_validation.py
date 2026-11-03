"""Invalid training settings must fail before any pipeline or cleanup work."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from remag.cli import main_cli


@pytest.mark.parametrize(
    "option",
    [
        "--epochs",
        "--min-contig-length",
        "--threads",
        "--batch-size",
        "--embedding-dim",
        "--max-positive-pairs",
        "--coverage-batch-size",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("force", [False, True])
def test_nonpositive_values_leave_existing_results_untouched(
    tmp_path, option, value, force
):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c\n" + "A" * 1200 + "\n")
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "siamese_model.pt"
    existing.write_bytes(b"previous model")
    command = [str(fasta), "-o", str(output), option, str(value)]
    if force:
        command.append("--force")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 2, result.output
    assert option in result.output
    assert "not in the range" in result.output
    run.assert_not_called()
    assert existing.read_bytes() == b"previous model"


@pytest.mark.parametrize("epochs,minimum", [(1, 1), (100, 1000), (2, 4096)])
def test_positive_values_are_forwarded_unchanged(tmp_path, epochs, minimum):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c\nACGT\n")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(
            main_cli,
            [str(fasta), "--epochs", str(epochs), "--min-contig-length", str(minimum)],
        )
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.epochs == epochs
    assert args.min_contig_length == minimum


@pytest.mark.parametrize("position", ["before", "after"])
def test_positional_fasta_may_follow_coverage_paths(tmp_path, position):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c\n" + "A" * 1200 + "\n")
    bams = [tmp_path / "s1.bam", tmp_path / "s2.bam"]
    for bam in bams:
        bam.touch()
    coverage = ["-c", *map(str, bams)]
    command = (
        [str(fasta), *coverage] if position == "before" else [*coverage, str(fasta)]
    )
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.fasta == str(fasta)
    assert args.bam == sorted(map(str, bams))
