"""Fixed length defaults and protection against incompatible reruns."""

import gzip
import json
from argparse import Namespace
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from remag.cli import main_cli
from remag.core import main
from remag.features import get_features
from remag.output import binning_settings, validate_cached_min_contig_length


def write_fasta(path, lengths):
    content = "".join(f">c{i}\n{'A' * length}\n" for i, length in enumerate(lengths))
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            handle.write(content)
    else:
        path.write_text(content)
    return str(path)


@pytest.mark.parametrize(
    "coverage_count,lengths",
    [
        (0, []),
        (1, []),
        (3, []),
        (0, [0, 999]),
        (1, [0, 999]),
        (3, [0, 999]),
    ],
)
def test_empty_selection_stops_before_forced_cleanup(tmp_path, lengths, coverage_count):
    fasta = write_fasta(tmp_path / "input.fa", lengths)
    output = tmp_path / "out"
    output.mkdir()
    previous = output / "embeddings.csv"
    previous.write_text("existing result")
    command = [fasta, "-o", str(output), "--force"]
    for i in range(coverage_count):
        coverage = tmp_path / f"sample{i}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 1
    assert "no input contigs" in result.output.lower()
    assert previous.read_text() == "existing result"
    run.assert_not_called()


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
@pytest.mark.parametrize("skip_filter", [False, True])
@pytest.mark.parametrize("suffix", [".fa", ".fa.gz"])
@pytest.mark.parametrize("lengths", [[1000, 1500, 6000], [2500], [5000]])
def test_cli_defaults_do_not_depend_on_assembly_lengths(
    tmp_path, coverage_count, skip_filter, suffix, lengths
):
    fasta = write_fasta(tmp_path / f"input{suffix}", lengths)
    command = [fasta, "-o", str(tmp_path / "out")]
    for i in range(coverage_count):
        coverage = tmp_path / f"sample{i}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    if skip_filter:
        command.append("--skip-bacterial-filter")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.min_contig_length == 1000
    assert args.graph_min_contig_length == (1000 if coverage_count == 1 else 3000)
    assert not hasattr(args, "contig_length_median")
    assert args.skip_bacterial_filter is skip_filter
    assert args.base_learning_rate == (0.0005 if coverage_count > 1 else 0.005)


@pytest.mark.parametrize("extension", ["bam", "cram", "tsv", "cov.gz"])
@pytest.mark.parametrize("length", [2500, 10000])
def test_single_coverage_uses_1000_for_admission_and_graph(tmp_path, extension, length):
    fasta = write_fasta(tmp_path / "input.fa.gz", [length])
    coverage = tmp_path / f"sample.{extension}"
    coverage.touch()
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, [fasta, "-c", str(coverage)])
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.min_contig_length == 1000
    assert args.graph_min_contig_length == 1000


@pytest.mark.parametrize("minimum", [1, 1000, 4096, 5000])
@pytest.mark.parametrize("coverage_count", [0, 1, 3])
def test_explicit_minimum_skips_length_scan(tmp_path, minimum, coverage_count):
    fasta = write_fasta(tmp_path / "input.fa", [4])
    command = [fasta, "--min-contig-length", str(minimum)]
    for i in range(coverage_count):
        coverage = tmp_path / f"sample{i}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    with (
        patch("remag.utils.fasta_iter") as scan,
        patch("remag.cli.run_remag") as run,
    ):
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    scan.assert_not_called()
    args = run.call_args.args[0]
    assert args.min_contig_length == minimum
    assert args.graph_min_contig_length == max(
        minimum, 1000 if coverage_count == 1 else 3000
    )
    assert not hasattr(args, "contig_length_median")


def test_default_validation_stops_at_first_admitted_contig(tmp_path):
    fasta = write_fasta(tmp_path / "input.fa", [1000])

    def records():
        yield "c", "A" * 1000
        raise AssertionError("Default selection must not scan the whole assembly")

    with (
        patch("remag.utils.fasta_iter", return_value=records()),
        patch("remag.cli.run_remag") as run,
    ):
        result = CliRunner().invoke(main_cli, [fasta])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].min_contig_length == 1000


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
@pytest.mark.parametrize("graph_minimum", [1000, 5000])
def test_explicit_graph_minimum_overrides_scenario_default(
    tmp_path, coverage_count, graph_minimum
):
    fasta = write_fasta(tmp_path / "input.fa", [1000, 6000])
    command = [fasta, "--graph-min-contig-length", str(graph_minimum)]
    for index in range(coverage_count):
        coverage = tmp_path / f"sample{index}.tsv"
        coverage.touch()
        command += ["-c", str(coverage)]
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].min_contig_length == 1000
    assert run.call_args.args[0].graph_min_contig_length == graph_minimum


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
def test_default_features_match_explicit_1000_admission_and_training(
    tmp_path, coverage_count
):
    lengths = [999, 1000, 1800, 3000, 5000, 6000]
    fasta = write_fasta(tmp_path / "input.fa", lengths)
    command = [fasta]
    for index in range(coverage_count):
        coverage = tmp_path / f"sample{index}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    (tmp_path / "default").mkdir()
    (tmp_path / "explicit").mkdir()
    default, default_fragments = get_features(
        fasta, None, None, str(tmp_path / "default"), args.min_contig_length, 1, 4, args
    )
    explicit, explicit_fragments = get_features(
        fasta, None, None, str(tmp_path / "explicit"), 1000, 1, 4
    )
    pd.testing.assert_frame_equal(default, explicit)
    assert default_fragments == explicit_fragments
    assert set(default_fragments) == {
        f"c{i}" for i, length in enumerate(lengths) if length >= 1000
    }
    assert all(
        fragment["length"] >= 1000
        for contig in default_fragments.values()
        for fragment in contig["fragment_info"].values()
    )


def filter_only_args(tmp_path, minimum=1000, force=False):
    return Namespace(
        fasta=write_fasta(tmp_path / "input.fa", [1200, 6000]),
        output=str(tmp_path / "out"),
        min_contig_length=minimum,
        verbose=False,
        filter_only=True,
        skip_bacterial_filter=False,
        keep_intermediate=False,
        force=force,
    )


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
@pytest.mark.parametrize("coverage_type", ["bam", "tsv"])
def test_core_records_the_same_graph_defaults_as_cli(
    tmp_path, coverage_count, coverage_type
):
    args = filter_only_args(tmp_path)
    setattr(args, coverage_type, [f"sample{i}" for i in range(coverage_count)])
    with patch("remag.core.filter_bacterial_contigs", return_value=args.fasta):
        main(args)
    params = json.loads((tmp_path / "out" / "params.json").read_text())
    assert params["min_contig_length"] == 1000
    assert params["graph_min_contig_length"] == (1000 if coverage_count == 1 else 3000)
    assert (
        params["binning"]["graph_min_contig_length"]
        == params["graph_min_contig_length"]
    )
    assert "contig_length_median" not in params


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "broken",
        "[]",
        "null",
        "{}",
        '{"min_contig_length": true}',
        '{"min_contig_length": 1000.0}',
        '{"min_contig_length": 0}',
        '{"min_contig_length": 4096}',
    ],
)
def test_incompatible_rerun_preserves_files_and_stops_before_filter(tmp_path, metadata):
    args = filter_only_args(tmp_path)
    args.keep_intermediate = True
    output = tmp_path / "out"
    output.mkdir()
    (output / "embeddings.csv").write_text("existing embeddings")
    (output / "remag.log").write_text("original log")
    if metadata is not None:
        (output / "params.json").write_text(metadata)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with patch("remag.core.filter_bacterial_contigs") as filtering:
        with pytest.raises(SystemExit, match="1"):
            main(args)
    filtering.assert_not_called()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    with pytest.raises(ValueError, match="--force.*new output directory"):
        validate_cached_min_contig_length(str(output), 1000)


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
def test_default_admission_rejects_cached_4096_run(tmp_path, coverage_count):
    fasta = write_fasta(tmp_path / "input.fa", [5000])
    output = tmp_path / "out"
    output.mkdir()
    (output / "params.json").write_text('{"min_contig_length": 4096}')
    (output / "embeddings.csv").write_text("existing embeddings")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    command = [fasta, "-o", str(output)]
    for index in range(coverage_count):
        coverage = tmp_path / f"sample{index}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    with patch("remag.core.filter_bacterial_contigs") as filtering:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 1
    filtering.assert_not_called()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("coverage_count", [0, 3])
def test_new_graph_default_rejects_cached_1000_graph(tmp_path, coverage_count):
    fasta = write_fasta(tmp_path / "input.fa", [1000, 6000])
    output = tmp_path / "out"
    command = [fasta, "-o", str(output)]
    for index in range(coverage_count):
        coverage = tmp_path / f"sample{index}.bam"
        coverage.touch()
        command += ["-c", str(coverage)]
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    old_args = run.call_args.args[0]
    old_args.graph_min_contig_length = 1000
    output.mkdir()
    (output / "params.json").write_text('{"min_contig_length": 1000}')
    (output / "pre_rescue.csv").write_text("existing assignments")
    (output / "clustering_provenance.json").write_text(
        json.dumps({"settings": binning_settings(old_args)})
    )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with patch("remag.core.filter_bacterial_contigs") as filtering:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 1
    filtering.assert_not_called()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("keep_intermediate", [False, True])
def test_matching_legacy_rerun_preserves_original_parameters(
    tmp_path, keep_intermediate
):
    args = filter_only_args(tmp_path)
    args.keep_intermediate = keep_intermediate
    output = tmp_path / "out"
    output.mkdir()
    original = '{"version": "0.5.0", "min_contig_length": 1000}\n'
    (output / "params.json").write_text(original)
    with patch(
        "remag.core.filter_bacterial_contigs", return_value=args.fasta
    ) as filtering:
        main(args)
    filtering.assert_called_once()
    assert (output / "params.json").read_text() == original


@pytest.mark.parametrize("force", [False, True])
def test_new_and_forced_runs_always_record_resolved_cutoff(tmp_path, force, capsys):
    args = filter_only_args(tmp_path, minimum=4096, force=force)
    output = tmp_path / "out"
    if force:
        output.mkdir()
        (output / "params.json").write_text('{"min_contig_length": 1000}')
        (output / "embeddings.csv").write_text("old embeddings")
    with patch(
        "remag.core.filter_bacterial_contigs", return_value=args.fasta
    ) as filtering:
        main(args)
    assert filtering.call_args.kwargs["min_contig_length"] == 4096
    params = json.loads((output / "params.json").read_text())
    assert params["min_contig_length"] == 4096
    assert params["graph_min_contig_length"] == 4096
    assert "contig_length_median" not in params
    assert params["keep_intermediate"] is False
    assert not (output / "embeddings.csv").exists()
    assert "Median input contig" not in (output / "remag.log").read_text()
    assert "4,096" in capsys.readouterr().out
