"""Literal coverage paths take precedence over grouped paths and wildcards."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from remag.cli import main_cli


def invoke_coverage(tmp_path, options):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c\n" + "A" * 1200 + "\n")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, [str(fasta), *options])
    return result, run


@pytest.mark.parametrize("name", ["sample 1", "sample[1]", "read files/sample [1]"])
@pytest.mark.parametrize(
    "suffix,field",
    [(".bam", "bam"), (".cram", "bam"), (".tsv", "tsv"), (".cov.gz", "tsv")],
)
def test_literal_coverage_paths_are_preserved(tmp_path, name, suffix, field):
    coverage = tmp_path / f"{name}{suffix}"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.touch()
    result, run = invoke_coverage(tmp_path, ["-c", str(coverage)])
    assert result.exit_code == 0, result.output
    assert getattr(run.call_args.args[0], field) == [str(coverage)]


def test_literal_brackets_do_not_select_another_file(tmp_path):
    literal = tmp_path / "sample[1].bam"
    wildcard_match = tmp_path / "sample1.bam"
    literal.touch()
    wildcard_match.touch()
    result, run = invoke_coverage(tmp_path, ["-c", str(literal)])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].bam == [str(literal)]


@pytest.mark.parametrize("syntax", ["repeated", "multiple", "grouped", "glob"])
def test_existing_multi_file_syntax_is_preserved(tmp_path, syntax):
    paths = [tmp_path / "a.bam", tmp_path / "b.bam"]
    for path in paths:
        path.touch()
    first, second = map(str, paths)
    options = {
        "repeated": ["-c", first, "-c", second],
        "multiple": ["-c", first, second],
        "grouped": ["-c", f"{first} {second}"],
        "glob": ["-c", str(tmp_path / "*.bam")],
    }[syntax]
    result, run = invoke_coverage(tmp_path, options)
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].bam == [first, second]


@pytest.mark.parametrize("syntax", ["repeated", "multiple", "glob"])
def test_multiple_paths_can_each_contain_spaces(tmp_path, syntax):
    paths = [tmp_path / "sample 1.bam", tmp_path / "sample 2.bam"]
    for path in paths:
        path.touch()
    first, second = map(str, paths)
    options = {
        "repeated": ["-c", first, "-c", second],
        "multiple": ["-c", first, second],
        "glob": ["-c", str(tmp_path / "*.bam")],
    }[syntax]
    result, run = invoke_coverage(tmp_path, options)
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].bam == [first, second]


@pytest.mark.parametrize("kind", ["missing", "directory", "unmatched-glob"])
def test_invalid_paths_still_fail_before_pipeline(tmp_path, kind):
    coverage = tmp_path / "missing.bam"
    if kind == "directory":
        coverage.mkdir()
    elif kind == "unmatched-glob":
        coverage = tmp_path / "missing*.bam"
    result, run = invoke_coverage(tmp_path, ["-c", str(coverage)])
    assert result.exit_code == 2, result.output
    run.assert_not_called()
