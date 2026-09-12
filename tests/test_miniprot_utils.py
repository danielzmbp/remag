"""Tests for miniprot utilities and security fixes."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from remag import miniprot_utils
from remag.miniprot_utils import (
    check_core_gene_duplications,
    check_miniprot_available,
    get_gene_mappings_cache_path,
    load_or_generate_gene_mappings,
)


def test_load_or_generate_gene_mappings_reuses_cache(tmp_path):
    args = SimpleNamespace(output=str(tmp_path), cores=2, verbose=False)
    expected = {"contig_1": {"gene_1": {"score": 0.8}}}
    cache_path = get_gene_mappings_cache_path(args)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(expected, cache_file)

    with patch("remag.miniprot_utils.subprocess.run") as run_miniprot:
        result = load_or_generate_gene_mappings({}, args)

    assert result == expected
    run_miniprot.assert_not_called()


def test_load_or_generate_gene_mappings_returns_new_mappings(tmp_path):
    args = SimpleNamespace(
        output=str(tmp_path), cores=2, verbose=False, keep_intermediate=True
    )
    fragments = {"contig_1": {"sequence": "ATGC"}}

    def write_paf(*_args, stdout, **_kwargs):
        stdout.write("gene_1_1\t100\t0\t80\t+\tcontig_1\t1000\t0\t100\t70\t80\n")
        return SimpleNamespace(returncode=0)

    with patch(
        "remag.miniprot_utils.check_miniprot_available", return_value=True
    ), patch("remag.miniprot_utils.subprocess.run", side_effect=write_paf):
        result = load_or_generate_gene_mappings(fragments, args)

    assert set(result) == {"contig_1"}
    assert set(result["contig_1"]) == {"gene"}
    assert (tmp_path / "temp_gene_mapping").is_dir()
    with open(get_gene_mappings_cache_path(args), encoding="utf-8") as cache_file:
        assert json.load(cache_file) == result


class TestMiniprot:
    """Test miniprot utility functions."""

    def test_check_miniprot_available_true(self):
        """Test miniprot availability check when installed."""
        with patch("shutil.which", return_value="/usr/bin/miniprot"):
            assert check_miniprot_available() is True

    def test_check_miniprot_available_false(self):
        """Test miniprot availability check when not installed."""
        with patch("shutil.which", return_value=None):
            assert check_miniprot_available() is False


@pytest.fixture
def miniprot_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(miniprot_utils, "check_miniprot_available", lambda: True)
    clusters = pd.DataFrame({"contig": ["contig_1"], "cluster": ["cluster_1"]})
    fragments = {"contig_1": {"sequence": "ATCGATCG", "length": 8}}
    args = SimpleNamespace(
        output=str(tmp_path),
        cores=4,
        verbose=False,
        min_bin_size=1,
        keep_intermediate=True,
    )
    return clusters, fragments, args


@pytest.mark.parametrize(
    "directory_name",
    ["normal", "with spaces", "with;semicolon", "with$(substitution)", "with'quotes"],
)
def test_miniprot_command_arguments(miniprot_inputs, tmp_path, directory_name):
    """Require a real invocation path and preserve filenames as literal arguments."""
    clusters, fragments, args = miniprot_inputs
    args.output = str(tmp_path / directory_name)

    def write_paf(*_args, stdout, **_kwargs):
        stdout.write("gene_1_1\t100\t0\t80\t+\tcontig_1\t8\t0\t8\t70\t80\n")
        return SimpleNamespace(returncode=0)

    with patch("remag.miniprot_utils.subprocess.run", side_effect=write_paf) as run:
        result = check_core_gene_duplications(clusters, fragments, args)

    run.assert_called_once()
    fasta = Path(args.output) / "temp_miniprot" / "cluster_1.fa"
    database = Path(miniprot_utils.__file__).parent / "db" / "refseq_db.faa.gz"
    assert run.call_args.args == (
        ["miniprot", "-I", "-t", "4", "--outs=0.95", str(fasta), str(database)],
    )
    options = run.call_args.kwargs
    assert options.get("shell", False) is False
    assert options["timeout"] == 14400
    assert options["check"] is False
    assert Path(options["stdout"].name) == fasta.with_suffix(".paf")
    assert Path(options["stderr"].name) == fasta.with_suffix(".stderr")
    assert fasta.read_text() == ">contig_1\nATCGATCG\n"
    assert result.loc[0, "total_core_genes_found"] == 1
    assert result.loc[0, "single_copy_genes_count"] == 1


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("miniprot disappeared"),
        PermissionError("execution denied"),
        subprocess.TimeoutExpired(cmd=["miniprot"], timeout=14400),
    ],
    ids=["missing-executable", "permission-denied", "timeout"],
)
def test_miniprot_execution_error_is_reported(miniprot_inputs, error):
    """Exercise and report each failure rather than accepting an untested path."""
    clusters, fragments, args = miniprot_inputs
    with patch("remag.miniprot_utils.subprocess.run", side_effect=error) as run, patch(
        "remag.miniprot_utils.logger.warning"
    ) as warning:
        result = check_core_gene_duplications(clusters, fragments, args)

    run.assert_called_once()
    warning.assert_any_call(f"Error running miniprot for cluster_1: {error}")
    pd.testing.assert_frame_equal(result[["contig", "cluster"]], clusters)
