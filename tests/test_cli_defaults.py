"""Tests for REMAG CLI default behavior."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner  # Import CliRunner

# Import the main CLI function
from remag.cli import main_cli


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
        assert args.epochs == 100

    def test_training_defaults_are_scenario_independent(
        self, mock_run_remag, temp_fasta, tmp_path
    ):
        """Lambda and the epoch budget must not vary with the number of coverage files.

        Both were benchmarked as single values across coassembly and single-sample
        scenarios and all three sequencing technologies; a scenario-dependent lambda
        was measured to be worse for coassembly, which is the case the old auto rule
        tried to special-case.
        """
        runner = CliRunner()
        invocations = [[temp_fasta, "--output", "remag_output"]]
        for n in (1, 3):
            bams = []
            for i in range(n):
                bam = tmp_path / f"cov{n}_{i}.bam"
                bam.touch()
                bams.append(str(bam))
            args_list = [temp_fasta, "--output", "remag_output"]
            for bam in bams:
                args_list += ["--coverage", bam]
            invocations.append(args_list)

        for args_list in invocations:
            result = runner.invoke(main_cli, args_list)
            assert result.exit_code == 0, f"CLI command failed: {result.exception}"
            args = mock_run_remag.call_args[0][0]
            assert args.barlow_lambda == 0.003
            assert args.epochs == 100

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

        # Coassembly lowers the base learning rate but does not change lambda:
        # 0.003 is the single default for every scenario.
        assert args.base_learning_rate == 0.0005
        assert args.barlow_lambda == 0.003

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

    @pytest.mark.parametrize("coverage_count", [0, 1, 2])
    @pytest.mark.parametrize("extension", ["bam", "tsv"])
    def test_standard_defaults_follow_coverage_count(
        self, mock_run_remag, temp_fasta, tmp_path, coverage_count, extension
    ):
        """All input layouts use the standard graph and filtering defaults."""
        command = [temp_fasta, "--output", str(tmp_path / "out")]
        for index in range(coverage_count):
            coverage = tmp_path / f"sample{index}.{extension}"
            coverage.touch()
            command.extend(["--coverage", str(coverage)])

        result = CliRunner().invoke(main_cli, command)
        assert result.exit_code == 0, result.output
        args = mock_run_remag.call_args.args[0]
        assert args.leiden_k_neighbors == 15
        assert args.skip_bacterial_filter is False
        assert args.min_contig_length == (4096 if coverage_count > 1 else 1000)
        assert args.base_learning_rate == (0.0005 if coverage_count > 1 else 0.005)
        assert not hasattr(args, "mode")

    @pytest.mark.parametrize("coverage_count,min_length", [(0, 4096), (2, 1000)])
    def test_explicit_graph_filter_and_length_settings_override_defaults(
        self, mock_run_remag, temp_fasta, tmp_path, coverage_count, min_length
    ):
        """Explicit settings can reproduce previous presets without a mode flag."""
        command = [
            temp_fasta,
            "--output",
            str(tmp_path / "out"),
            "--min-contig-length",
            str(min_length),
            "--leiden-k-neighbors",
            "30",
            "--skip-bacterial-filter",
        ]
        for index in range(coverage_count):
            coverage = tmp_path / f"sample{index}.bam"
            coverage.touch()
            command.extend(["--coverage", str(coverage)])

        result = CliRunner().invoke(main_cli, command)
        assert result.exit_code == 0, result.output
        args = mock_run_remag.call_args.args[0]
        assert args.min_contig_length == min_length
        assert args.leiden_k_neighbors == 30
        assert args.skip_bacterial_filter is True

    @pytest.mark.parametrize("flag", ["-m", "--mode"])
    @pytest.mark.parametrize(
        "mode", ["metagenomics", "single-cell", "short-reads", "sr"]
    )
    def test_removed_modes_fail_before_starting_workflow(
        self, mock_run_remag, temp_fasta, flag, mode
    ):
        """Old mode commands must fail instead of silently changing behavior."""
        result = CliRunner().invoke(main_cli, [temp_fasta, flag, mode])
        assert result.exit_code == 2, result.output
        assert "No such option" in result.output
        assert flag in result.output
        mock_run_remag.assert_not_called()

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_help_does_not_advertise_modes(self, flag):
        with patch("sys.argv", ["remag", flag]):
            result = CliRunner().invoke(main_cli, [flag])
        assert result.exit_code == 0, result.output
        assert "--mode" not in result.output
        assert "single-cell" not in result.output
        assert "short-reads" not in result.output
