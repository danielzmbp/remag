"""Tests for REMAG CLI default behavior."""

import os
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner  # Import CliRunner

# Import the main CLI function and the core run function
from remag.cli import main_cli
from remag.core import main as run_remag


@pytest.fixture
def mock_run_remag():
    """Fixture to mock remag.core.main and capture its arguments."""
    with patch("remag.cli.run_remag") as mock:
        yield mock


@pytest.fixture
def temp_fasta(tmp_path):
    """Creates a dummy FASTA file for testing."""
    fasta_path = tmp_path / "contigs.fasta"
    fasta_path.write_text(">contig1\nATGC\n>contig2\nTGCA\n")
    return str(fasta_path)


@pytest.fixture
def temp_bam(tmp_path):
    """Creates a dummy BAM file for testing."""
    bam_path = tmp_path / "sample.bam"
    # Create an empty file, as content isn't relevant for these tests
    bam_path.touch()
    return str(bam_path)


class TestCliDefaults:
    """Test default values set by the CLI based on input."""

    def test_default_values_no_coverage(self, mock_run_remag, temp_fasta):
        """Test default learning rate and lambda when no coverage is provided."""
        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--output",
                "remag_output",
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        # Get the args object passed to remag.core.main
        args = mock_run_remag.call_args[0][0]

        # Assert expected values
        assert args.base_learning_rate == 0.005
        assert args.barlow_lambda == 0.003

    def test_default_values_single_coverage(self, mock_run_remag, temp_fasta, temp_bam):
        """Test default learning rate and lambda with a single coverage file."""
        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--coverage",
                temp_bam,
                "--output",
                "remag_output",
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        args = mock_run_remag.call_args[0][0]

        assert args.base_learning_rate == 0.005
        assert args.barlow_lambda == 0.003

    def test_default_values_multiple_coverage_coassembly(
        self, mock_run_remag, temp_fasta, tmp_path
    ):
        """Test default learning rate and lambda for coassembly (multiple coverage files)."""
        temp_bam1 = tmp_path / "sample1.bam"
        temp_bam1.touch()
        temp_bam2 = tmp_path / "sample2.bam"
        temp_bam2.touch()

        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--coverage",
                str(temp_bam1),
                "--coverage",
                str(temp_bam2),
                "--output",
                "remag_output",
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        args = mock_run_remag.call_args[0][0]

        # Assert the new default values for coassembly
        assert args.base_learning_rate == 0.0005
        assert args.barlow_lambda == 0.02

    def test_user_specified_values_override_defaults(
        self, mock_run_remag, temp_fasta, tmp_path
    ):
        """Test that user-specified learning rate and lambda override defaults."""
        temp_bam1 = tmp_path / "sample1.bam"
        temp_bam1.touch()
        temp_bam2 = tmp_path / "sample2.bam"
        temp_bam2.touch()

        user_lr = 0.01
        user_lambda = 0.05

        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--coverage",
                str(temp_bam1),
                "--coverage",
                str(temp_bam2),
                "--output",
                "remag_output",
                "--base-learning-rate",
                str(user_lr),
                "--barlow-lambda",
                str(user_lambda),
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        args = mock_run_remag.call_args[0][0]

        # Assert user-specified values are used
        assert args.base_learning_rate == user_lr
        assert args.barlow_lambda == user_lambda

    def test_user_specified_default_learning_rate_is_respected(
        self, mock_run_remag, temp_fasta, tmp_path
    ):
        """Test that an explicitly provided default-valued learning rate is not rewritten."""
        temp_bam1 = tmp_path / "sample1.bam"
        temp_bam1.touch()
        temp_bam2 = tmp_path / "sample2.bam"
        temp_bam2.touch()

        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--coverage",
                str(temp_bam1),
                "--coverage",
                str(temp_bam2),
                "--output",
                "remag_output",
                "--base-learning-rate",
                "0.005",
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        args = mock_run_remag.call_args[0][0]

        assert args.base_learning_rate == 0.005

    def test_single_cell_mode_uses_larger_default_knn(
        self, mock_run_remag, temp_fasta, temp_bam
    ):
        """Test single-cell mode applies the documented larger default k-NN size."""
        runner = CliRunner()
        result = runner.invoke(
            main_cli,
            [
                temp_fasta,
                "--coverage",
                temp_bam,
                "--mode",
                "single-cell",
                "--output",
                "remag_output",
            ],
        )
        assert result.exit_code == 0, f"CLI command failed: {result.exception}"

        args = mock_run_remag.call_args[0][0]

        assert args.leiden_k_neighbors == 30
