"""Tests for explicit output replacement without changing cache reuse defaults."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from remag.cli import main_cli
from remag.output import prepare_output_directory


def make_args(tmp_path, force=False):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c1\nACGT\n")
    return Namespace(
        fasta=str(fasta),
        output=str(tmp_path / "out"),
        bam=None,
        tsv=None,
        force=force,
        verbose=False,
        filter_only=True,
        skip_bacterial_filter=False,
        keep_intermediate=False,
        min_contig_length=1,
    )


def populate_outputs(output):
    """Create representative cached stages and outputs from an earlier input."""
    names = [
        "features.csv",
        "fragments.pkl",
        "siamese_model.pt",
        "embeddings.csv",
        "kmer_embeddings.csv",
        "coverage_embeddings.csv",
        "params.json",
        "gene_contig_mappings.json",
        "core_gene_duplication_results.json",
        "knn_graph_edges.csv",
        "knn_graph_stats.json",
        "bins.csv",
        "bins/bin_0.fa",
        "bins/bin_999.fa",
        "old_eukaryotic_filtered.fasta",
        "old_hyenadna_classification.tsv",
        "old_non_eukaryotic.fasta",
        "old_eukaryotic_filtered.fasta.tmp",
        "old_non_eukaryotic.fasta.tmp",
        "temp_gene_mapping/all_contigs.paf",
        "temp_miniprot/bin_0.paf",
        "umap_coordinates.csv",
        "umap_plot.pdf",
        "remag.log",
    ]
    paths = []
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing result")
        paths.append(path)
    return paths


def test_default_keeps_existing_results(tmp_path):
    args = make_args(tmp_path)
    paths = populate_outputs(Path(args.output))
    assert prepare_output_directory(args)
    assert all(path.read_text() == "existing result" for path in paths)


def test_force_removes_old_stages_and_preserves_unrelated_files(tmp_path):
    args = make_args(tmp_path, force=True)
    paths = populate_outputs(Path(args.output))
    notes = Path(args.output) / "notes.txt"
    notes.write_text("keep my notes")
    args._gene_mappings_cache = {"old": {}}
    assert prepare_output_directory(args)
    assert all(not path.exists() for path in paths)
    assert not (Path(args.output) / "temp_gene_mapping").exists()
    assert not (Path(args.output) / "temp_miniprot").exists()
    assert notes.read_text() == "keep my notes"
    assert Path(args.fasta).read_text() == ">c1\nACGT\n"
    assert not hasattr(args, "_gene_mappings_cache")


@pytest.mark.parametrize("source", ["bins/bin_0.fa", "temp_miniprot/input.fa"])
def test_force_checks_inputs_before_removing_any_output(tmp_path, source):
    args = make_args(tmp_path, force=True)
    paths = populate_outputs(Path(args.output))
    input_path = Path(args.output) / source
    input_path.write_text(">input\nACGT\n")
    args.fasta = str(input_path)
    with pytest.raises(ValueError, match="would remove an input"):
        prepare_output_directory(args)
    assert input_path.read_text() == ">input\nACGT\n"
    assert all(path.exists() for path in paths)


def test_force_unlinks_temporary_directory_symlink_without_deleting_target(tmp_path):
    args = make_args(tmp_path, force=True)
    output = Path(args.output)
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    valuable = external / "keep.txt"
    valuable.write_text("keep")
    link = output / "temp_miniprot"
    link.symlink_to(external, target_is_directory=True)
    assert prepare_output_directory(args)
    assert not link.is_symlink()
    assert valuable.read_text() == "keep"


@pytest.mark.parametrize("force", [False, True])
def test_empty_output_is_a_normal_run(tmp_path, force):
    args = make_args(tmp_path, force=force)
    assert prepare_output_directory(args) is False


@pytest.mark.parametrize("force", [False, True])
def test_core_prepares_outputs_before_filtering(tmp_path, force, capsys):
    from remag.core import main

    args = make_args(tmp_path, force=force)
    paths = populate_outputs(Path(args.output))

    def check_outputs(*_args, **_kwargs):
        # The log is reopened by setup_logging; other outputs follow force.
        assert all(path.exists() != force for path in paths if path.name != "remag.log")
        return args.fasta

    with patch("remag.core.filter_bacterial_contigs", side_effect=check_outputs):
        main(args)
    message = "Existing REMAG results found. Reusing available outputs. Use --force to recompute."
    assert (message in capsys.readouterr().out) != force


@pytest.mark.parametrize("force", [False, True])
def test_cli_forwards_force(tmp_path, force):
    args = make_args(tmp_path)
    command = [args.fasta, "--filter-only", "-o", args.output]
    if force:
        command.append("--force")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].force is force


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_force_is_in_basic_and_full_help(flag):
    with patch("sys.argv", ["remag", flag]):
        result = CliRunner().invoke(main_cli, [flag])
    assert result.exit_code == 0
    assert "--force" in result.output
