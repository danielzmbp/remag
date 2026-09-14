"""Protect inputs and external files during forced output cleanup."""

import os
from pathlib import Path

import pytest

from remag.output import prepare_output_directory
from tests.test_force import make_args, populate_outputs


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("target_kind", ["external", "internal", "missing"])
def test_force_refuses_a_linked_bins_directory(tmp_path, force, target_kind):
    args = make_args(tmp_path, force=force)
    output = Path(args.output)
    output.mkdir()
    marker = output / "features.csv"
    marker.write_text("existing features")
    target = (
        output / "other_bins" if target_kind == "internal" else tmp_path / "outside"
    )
    valuable = target / "bin_0.fa"
    if target_kind != "missing":
        target.mkdir()
        valuable.write_text(">valuable\nACGT\n")
    (output / "bins").symlink_to(target, target_is_directory=True)

    if force:
        with pytest.raises(ValueError, match="symlinked bins"):
            prepare_output_directory(args)
    else:
        assert prepare_output_directory(args)

    assert marker.read_text() == "existing features"
    assert (output / "bins").is_symlink()
    if target_kind != "missing":
        assert valuable.read_text() == ">valuable\nACGT\n"


@pytest.mark.parametrize("kind", ["fasta", "bam", "tsv"])
@pytest.mark.parametrize("directory", ["temp_miniprot", "temp_gene_mapping"])
def test_force_preserves_input_links_inside_removed_directories(
    tmp_path, kind, directory
):
    args = make_args(tmp_path, force=True)
    outputs = populate_outputs(Path(args.output))
    original = tmp_path / "original_input"
    original.write_text("valuable input")
    link = Path(args.output) / directory / "input_link"
    link.symlink_to(original)
    setattr(args, kind, str(link) if kind == "fasta" else [str(link)])

    with pytest.raises(ValueError, match="containing symbolic links"):
        prepare_output_directory(args)

    assert link.is_symlink()
    assert link.read_text() == "valuable input"
    assert all(path.exists() for path in outputs)


@pytest.mark.parametrize(
    "route", ["relative", "parent-alias", "file-chain", "directory-chain"]
)
def test_force_protects_indirect_input_link_routes(tmp_path, monkeypatch, route):
    args = make_args(tmp_path, force=True)
    outputs = populate_outputs(Path(args.output))
    output = Path(args.output)
    original = tmp_path / "original.fa"
    original.write_text(">valuable\nACGT\n")
    inner = output / "temp_miniprot" / "input.fa"
    inner.symlink_to(original)
    if route == "relative":
        monkeypatch.chdir(tmp_path)
        args.output = "out"
        args.fasta = os.path.relpath(inner, tmp_path)
    elif route == "parent-alias":
        alias = tmp_path / "alias"
        alias.symlink_to(output / "temp_miniprot", target_is_directory=True)
        args.fasta = str(alias / "input.fa")
    elif route == "file-chain":
        outer = tmp_path / "outer.fa"
        outer.symlink_to(inner)
        args.fasta = str(outer)
    else:
        middle = output / "temp_gene_mapping" / "linked_directory"
        middle.symlink_to(original.parent, target_is_directory=True)
        outer = tmp_path / "outer_directory"
        outer.symlink_to(middle, target_is_directory=True)
        args.fasta = str(outer / original.name)

    with pytest.raises(ValueError, match="containing symbolic links"):
        prepare_output_directory(args)

    assert Path(args.fasta).read_text() == ">valuable\nACGT\n"
    assert all(path.exists() for path in outputs)


def test_force_preserves_input_link_with_a_recognized_output_name(tmp_path):
    args = make_args(tmp_path, force=True)
    outputs = populate_outputs(Path(args.output))
    original = Path(args.fasta)
    link = Path(args.output) / "input_eukaryotic_filtered.fasta"
    link.symlink_to(original)
    args.fasta = str(link)

    with pytest.raises(ValueError, match="would remove an input"):
        prepare_output_directory(args)

    assert link.is_symlink()
    assert all(path.exists() for path in outputs)


def test_force_still_unlinks_an_unrelated_output_symlink(tmp_path):
    args = make_args(tmp_path, force=True)
    output = Path(args.output)
    output.mkdir()
    target = tmp_path / "valuable.csv"
    target.write_text("keep")
    link = output / "features.csv"
    link.symlink_to(target)

    assert prepare_output_directory(args)

    assert not link.is_symlink()
    assert target.read_text() == "keep"


@pytest.mark.parametrize("kind", ["file", "directory", "broken"])
def test_force_refuses_temporary_directories_with_unrelated_links(tmp_path, kind):
    args = make_args(tmp_path, force=True)
    outputs = populate_outputs(Path(args.output))
    external = tmp_path / "external"
    if kind == "directory":
        external.mkdir()
        (external / "valuable.txt").write_text("keep")
    elif kind == "file":
        external.write_text("keep")
    nested = Path(args.output) / "temp_miniprot" / "nested"
    nested.mkdir()
    link = nested / ".unrelated_link"
    link.symlink_to(external, target_is_directory=kind == "directory")

    with pytest.raises(ValueError, match="containing symbolic links"):
        prepare_output_directory(args)

    assert link.is_symlink()
    assert all(path.exists() for path in outputs)
    if kind == "directory":
        assert (external / "valuable.txt").read_text() == "keep"
    elif kind == "file":
        assert external.read_text() == "keep"
    else:
        assert not external.exists()
