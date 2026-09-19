"""Automatic length selection and protection against incompatible reruns."""

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
from remag.output import validate_cached_min_contig_length
from remag.utils import select_min_contig_length


def write_fasta(path, lengths):
    content = "".join(f">c{i}\n{'A' * length}\n" for i, length in enumerate(lengths))
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            handle.write(content)
    else:
        path.write_text(content)
    return str(path)


@pytest.mark.parametrize("suffix", [".fa", ".fa.gz"])
@pytest.mark.parametrize(
    "lengths,minimum,median",
    [
        ([999, 1000, 1200], 1000, 1100),
        ([1000], 1000, 1000),
        ([2499, 2499, 4096], 1000, 2499),
        ([1000, 2500, 4096], 4096, 2500),
        ([2499, 2500, 2500, 4096], 4096, 2500),
        ([2498, 2499, 2500, 4096], 1000, 2499.5),
        ([4096], 4096, 4096),
    ],
)
def test_unfiltered_median_boundaries_and_order(
    tmp_path, suffix, lengths, minimum, median
):
    fasta = tmp_path / f"input{suffix}"
    for ordered in (lengths, list(reversed(lengths))):
        assert select_min_contig_length(write_fasta(fasta, ordered)) == (
            minimum,
            median,
        )


@pytest.mark.parametrize("lengths", [[], [0, 999], [2500], [3000, 4095]])
def test_empty_selection_stops_before_forced_cleanup(tmp_path, lengths):
    fasta = write_fasta(tmp_path / "input.fa", lengths)
    output = tmp_path / "out"
    output.mkdir()
    previous = output / "embeddings.csv"
    previous.write_text("existing result")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, [fasta, "-o", str(output), "--force"])
    assert result.exit_code == 1
    assert "no input contigs" in result.output.lower()
    assert previous.read_text() == "existing result"
    run.assert_not_called()


@pytest.mark.parametrize("coverage_count", [0, 1, 3])
@pytest.mark.parametrize("skip_filter", [False, True])
@pytest.mark.parametrize(
    "lengths,minimum", [([1000, 1500, 6000], 1000), ([5000], 4096)]
)
def test_cli_resolves_before_filtering_independently_of_coverage(
    tmp_path, coverage_count, skip_filter, lengths, minimum
):
    fasta = write_fasta(tmp_path / "input.fa", lengths)
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
    assert args.min_contig_length == minimum
    assert args.contig_length_median is not None
    assert args.skip_bacterial_filter is skip_filter
    assert args.base_learning_rate == (0.0005 if coverage_count > 1 else 0.005)


@pytest.mark.parametrize("minimum", [1, 1000, 4096, 5000])
def test_explicit_minimum_skips_length_scan(tmp_path, minimum):
    fasta = write_fasta(tmp_path / "input.fa", [4])
    with (
        patch("remag.utils.select_min_contig_length") as select,
        patch("remag.cli.run_remag") as run,
    ):
        result = CliRunner().invoke(
            main_cli, [fasta, "--min-contig-length", str(minimum)]
        )
    assert result.exit_code == 0, result.output
    select.assert_not_called()
    assert run.call_args.args[0].min_contig_length == minimum
    assert run.call_args.args[0].contig_length_median is None


@pytest.mark.parametrize(
    "minimum,lengths",
    [(1000, [999, 1000, 1800, 6000]), (4096, [1000, 4096, 5000, 6000])],
)
def test_automatic_features_match_explicit_admission_and_training_fragments(
    tmp_path, minimum, lengths
):
    fasta = write_fasta(tmp_path / "input.fa", lengths)
    selected, _ = select_min_contig_length(fasta)
    assert selected == minimum
    (tmp_path / "auto").mkdir()
    (tmp_path / "explicit").mkdir()
    auto, auto_fragments = get_features(
        fasta, None, None, str(tmp_path / "auto"), selected, 1, 4
    )
    explicit, explicit_fragments = get_features(
        fasta, None, None, str(tmp_path / "explicit"), minimum, 1, 4
    )
    pd.testing.assert_frame_equal(auto, explicit)
    assert auto_fragments == explicit_fragments
    assert set(auto_fragments) == {
        f"c{i}" for i, length in enumerate(lengths) if length >= minimum
    }
    assert all(
        fragment["length"] >= minimum
        for contig in auto_fragments.values()
        for fragment in contig["fragment_info"].values()
    )


def filter_only_args(tmp_path, minimum=1000, force=False):
    return Namespace(
        fasta=write_fasta(tmp_path / "input.fa", [1200, 6000]),
        output=str(tmp_path / "out"),
        min_contig_length=minimum,
        contig_length_median=None,
        verbose=False,
        filter_only=True,
        skip_bacterial_filter=False,
        keep_intermediate=False,
        force=force,
    )


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
    args.contig_length_median = 3600
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
    assert params["contig_length_median"] == 3600
    assert params["keep_intermediate"] is False
    assert not (output / "embeddings.csv").exists()
    assert "3600 bp" in (output / "remag.log").read_text()
    assert "4,096" in capsys.readouterr().out
