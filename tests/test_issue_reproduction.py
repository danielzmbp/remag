
import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner
from remag.cli import main_cli

@pytest.fixture
def mock_run_remag():
    with patch('remag.cli.run_remag') as mock:
        yield mock

@pytest.fixture
def temp_fasta(tmp_path):
    p = tmp_path / "contigs.fasta"
    p.write_text(">c1\nATGC")
    return str(p)

@pytest.fixture
def multiple_bams(tmp_path):
    b1 = tmp_path / "s1.bam"
    b2 = tmp_path / "s2.bam"
    b1.touch()
    b2.touch()
    return [str(b1), str(b2)]

def test_force_lower_min_contig_length(mock_run_remag, temp_fasta, multiple_bams):
    """
    Test that user can force min_contig_length to 1000 even with multiple samples.
    """
    runner = CliRunner()
    
    # Explicitly set --min-contig-length 1000
    result = runner.invoke(main_cli, [
        temp_fasta,
        '-c', multiple_bams[0], '-c', multiple_bams[1],
        '--min-contig-length', '1000'
    ])
    
    assert result.exit_code == 0
    args = mock_run_remag.call_args[0][0]
    
    # This assertion ensures the user's value is respected
    assert args.min_contig_length == 1000

def test_defaults_stay_1000(mock_run_remag, temp_fasta, multiple_bams):
    """
    Test that it defaults to 1000 even if user DOES NOT specify length (no auto-bump to 4096).
    """
    runner = CliRunner()
    
    result = runner.invoke(main_cli, [
        temp_fasta,
        '-c', multiple_bams[0], '-c', multiple_bams[1]
    ])
    
    assert result.exit_code == 0
    args = mock_run_remag.call_args[0][0]
    
    assert args.min_contig_length == 1000

def test_single_sample_default(mock_run_remag, temp_fasta, multiple_bams):
    """
    Test that single sample defaults to 1000.
    """
    runner = CliRunner()
    
    result = runner.invoke(main_cli, [
        temp_fasta,
        '-c', multiple_bams[0]
    ])
    
    assert result.exit_code == 0
    args = mock_run_remag.call_args[0][0]
    
    assert args.min_contig_length == 1000

