
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from remag.cli import main_cli

def test_single_cell_rescue_defaults():
    """Test that single-cell mode uses relaxed rescue parameters by default."""
    runner = CliRunner()
    
    with patch('remag.cli.run_remag') as mock_run:
        # Create a dummy fasta file
        with runner.isolated_filesystem():
            with open("test.fasta", "w") as f:
                f.write(">seq1\nATGC\n")
            
            result = runner.invoke(main_cli, [
                "test.fasta", 
                "--mode", "single-cell",
                "--output", "out_dir"
            ])
            
            assert result.exit_code == 0
            
            # Get the args passed to run_remag
            args = mock_run.call_args[0][0]
            
            # Check rescue parameters
            # Should match coassembly defaults: 0.7 and 5.0
            assert args.rescue_similarity_threshold == 0.7
            assert args.rescue_max_duplication == 5.0
            
            # Check feedback message
            assert "Single-cell mode detected: Using relaxed rescue criteria" in result.output

def test_single_cell_rescue_overrides():
    """Test that user can override relaxed defaults in single-cell mode."""
    runner = CliRunner()
    
    with patch('remag.cli.run_remag') as mock_run:
        with runner.isolated_filesystem():
            with open("test.fasta", "w") as f:
                f.write(">seq1\nATGC\n")
            
            result = runner.invoke(main_cli, [
                "test.fasta", 
                "--mode", "single-cell",
                "--rescue-similarity-threshold", "0.85",
                "--rescue-max-duplication", "2.5",
                "--output", "out_dir"
            ])
            
            assert result.exit_code == 0
            args = mock_run.call_args[0][0]
            
            assert args.rescue_similarity_threshold == 0.85
            assert args.rescue_max_duplication == 2.5

def test_metagenomics_rescue_defaults():
    """Test that metagenomics mode still uses strict defaults for single sample."""
    runner = CliRunner()
    
    with patch('remag.cli.run_remag') as mock_run:
        with runner.isolated_filesystem():
            with open("test.fasta", "w") as f:
                f.write(">seq1\nATGC\n")
            
            result = runner.invoke(main_cli, [
                "test.fasta", 
                "--mode", "metagenomics",
                "--output", "out_dir"
            ])
            
            assert result.exit_code == 0
            args = mock_run.call_args[0][0]
            
            # Strict defaults
            assert args.rescue_similarity_threshold == 0.9
            assert args.rescue_max_duplication == 3.0
            assert "Single sample detected: Using strict rescue criteria" in result.output
