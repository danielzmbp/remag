"""Separate graph membership from admission/training and validate cache identity."""

import json
from argparse import Namespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from remag.cli import main_cli
from remag.clustering import _construct_knn_graph, cluster_contigs
from remag.output import embedding_identity, prepare_output_directory


def inputs(tmp_path):
    names = ["long_b", "short", "long_a"]
    emb = pd.DataFrame([[0.0, 1.0], [0.7, 0.7], [1.0, 0.0]], index=names)
    fr = {c: {"sequence": "A" * (1000 if c == "short" else 4000)} for c in names}
    args = Namespace(
        output=str(tmp_path),
        min_contig_length=1000,
        graph_min_contig_length=3000,
        keep_intermediate=True,
        cores=1,
    )
    return emb, fr, {}, args


def test_graph_subset_keeps_shared_embedding_order_and_noise(tmp_path):
    emb, fr, genes, args = inputs(tmp_path)
    with patch(
        "remag.clustering._greedy_leiden_clustering", return_value=[1, 0]
    ) as leiden:
        result = cluster_contigs(emb, fr, genes, args)
    assert leiden.call_args.kwargs["contig_names"] == ["long_b", "long_a"]
    np.testing.assert_array_equal(
        leiden.call_args.args[0], emb.loc[["long_b", "long_a"]].values
    )
    assert result.to_dict("list") == {
        "contig": list(emb.index),
        "cluster": ["bin_1", "noise", "bin_0"],
    }
    assert not (tmp_path / "bins.csv").exists()
    assert pd.read_csv(tmp_path / "pre_rescue.csv").equals(result)
    assert pd.read_csv(tmp_path / "knn_graph_contigs.csv").contig.tolist() == [
        "long_b",
        "long_a",
    ]


@pytest.mark.parametrize(
    "change",
    [
        "cutoff",
        "order",
        "embedding",
        "sequence",
        "genes",
        "rescue",
        "skip",
        "size",
        "algorithm",
    ],
)
def test_incompatible_cache_preserves_previous_outputs(tmp_path, change):
    emb, fr, genes, args = inputs(tmp_path)
    with patch("remag.clustering._greedy_leiden_clustering", return_value=[0, 0]):
        cluster_contigs(emb, fr, genes, args)
    (tmp_path / "bins.csv").write_text("contig,cluster\nshort,old_final\n")
    if change == "cutoff":
        args.graph_min_contig_length = 4000
    elif change == "order":
        emb = emb.iloc[::-1]
    elif change == "embedding":
        emb.iloc[0, 0] = 0.1
    elif change == "sequence":
        fr["short"]["sequence"] = "C" * 1000
    elif change == "genes":
        genes["short"] = {"g": {}}
    elif change == "rescue":
        args.rescue_max_duplication_increase = 4.0
    elif change == "skip":
        args.skip_rescue = True
    elif change == "size":
        args.min_bin_size = 600000
    else:
        meta = tmp_path / "clustering_provenance.json"
        data = json.loads(meta.read_text())
        data["settings"]["rescue_algorithm"] = "older-rescue"
        meta.write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="new output directory.*--force"):
        cluster_contigs(emb, fr, genes, args)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_final_bins_cannot_replace_initial_assignments(tmp_path):
    emb, fr, genes, args = inputs(tmp_path)
    with patch("remag.clustering._greedy_leiden_clustering", return_value=[0, 0]):
        expected = cluster_contigs(emb, fr, genes, args)
    (tmp_path / "bins.csv").write_text("contig,cluster\nshort,bin_0\n")
    with patch("remag.clustering._greedy_leiden_clustering") as leiden:
        actual = cluster_contigs(emb, fr, genes, args)
        leiden.assert_not_called()
    pd.testing.assert_frame_equal(actual, expected)
    (tmp_path / "pre_rescue.csv").unlink()
    with pytest.raises(ValueError, match="missing pre-rescue"):
        cluster_contigs(emb, fr, genes, args)


def test_legacy_bins_are_not_reused(tmp_path):
    emb, fr, genes, args = inputs(tmp_path)
    old = tmp_path / "bins.csv"
    old.write_text("contig,cluster\nlong_a,bin_0\n")
    with pytest.raises(ValueError, match="missing unified-rescue provenance"):
        cluster_contigs(emb, fr, genes, args)
    assert old.read_text() == "contig,cluster\nlong_a,bin_0\n"


@pytest.mark.parametrize("change", ["values", "ids", "order", "edges"])
def test_graph_cache_rejects_changed_inputs_even_at_same_vertex_count(tmp_path, change):
    values = np.eye(3)
    names = ["a", "b", "c"]
    args = Namespace(output=str(tmp_path), keep_intermediate=True)
    _construct_knn_graph(values, k=2, args=args, contig_names=names)
    if change == "values":
        values[0, 1] = 0.2
    elif change == "ids":
        names[0] = "different"
    elif change == "order":
        names.reverse()
    else:
        with (tmp_path / "knn_graph_edges.csv").open("a") as handle:
            handle.write("0,1,1.000000\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="graph cache"):
        _construct_knn_graph(values, k=2, args=args, contig_names=names)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_no_core_seeds_returns_all_noise(tmp_path):
    emb, fr, genes, args = inputs(tmp_path)
    args.graph_min_contig_length = 5000
    result = cluster_contigs(emb, fr, genes, args)
    assert result.contig.tolist() == list(emb.index)
    assert result.cluster.tolist() == ["noise"] * len(emb)


@pytest.mark.parametrize("coverage_count,expected", [(0, 4096), (1, 1000), (2, 4096)])
def test_graph_default_follows_existing_admission_rule(
    tmp_path, coverage_count, expected
):
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">c\n" + "A" * 10000 + "\n")
    command = [str(fasta)]
    for i in range(coverage_count):
        cov = tmp_path / f"s{i}.tsv"
        cov.write_text("c\t1\n")
        command += ["-c", str(cov)]
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(main_cli, command)
    assert result.exit_code == 0, result.output
    args = run.call_args.args[0]
    assert args.min_contig_length == args.graph_min_contig_length == expected


def test_explicit_graph_cutoff_does_not_change_admission_or_training(tmp_path):
    fasta = tmp_path / "assembly.fa"
    fasta.write_text(">short\n" + "A" * 1000 + "\n>long\n" + "C" * 3000 + "\n")
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(
            main_cli,
            [
                str(fasta),
                "--min-contig-length",
                "1000",
                "--graph-min-contig-length",
                "3000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].min_contig_length == 1000
    assert run.call_args.args[0].graph_min_contig_length == 3000
    with patch("remag.cli.run_remag") as run:
        result = CliRunner().invoke(
            main_cli,
            [
                str(fasta),
                "--min-contig-length",
                "3000",
                "--graph-min-contig-length",
                "1000",
                "--force",
            ],
        )
    assert result.exit_code != 0
    run.assert_not_called()


@pytest.mark.parametrize("names", [["a", "b"], ["001", "002"], ["NA", "null"]])
def test_exact_generated_embeddings_survive_cache_reload(tmp_path, names):
    import torch

    from remag import models

    values = np.array([[0.123456789, 0.987654321], [0.34, 0.56]], dtype=np.float32)
    features = pd.DataFrame([[1.0], [2.0]], index=[c + ".original" for c in names])
    model = Mock()
    model.get_embedding.return_value = torch.tensor(values)
    args = Namespace(output=str(tmp_path), batch_size=2, keep_intermediate=False)
    with patch("remag.models.get_torch_device", return_value="cpu"):
        fresh = models.generate_embeddings(model, features, args)
        cached = models.generate_embeddings(model, features, args)
    assert embedding_identity(fresh.values, fresh.index) == embedding_identity(
        cached.values, cached.index
    )
    np.testing.assert_array_equal(fresh.values.astype(float), cached.values)
    assert cached.index.tolist() == names


def test_force_includes_new_provenance_files(tmp_path):
    for name in [
        "pre_rescue.csv",
        "rescue_assignments.csv",
        "clustering_provenance.json",
        "knn_graph_contigs.csv",
    ]:
        (tmp_path / name).write_text("old")
    args = Namespace(
        output=str(tmp_path), fasta=str(tmp_path.parent / "input.fa"), force=True
    )
    assert prepare_output_directory(args)
    assert not list(tmp_path.iterdir())


def test_cached_assignments_preserve_literal_fasta_ids(tmp_path):
    emb, fr, genes, args = inputs(tmp_path)
    names = ["001", "NA", "null"]
    fr = dict(zip(names, fr.values()))
    emb.index = names
    with patch("remag.clustering._greedy_leiden_clustering", return_value=[0, 0]):
        expected = cluster_contigs(emb, fr, genes, args)
    with patch("remag.clustering._greedy_leiden_clustering") as cluster:
        actual = cluster_contigs(emb, fr, genes, args)
        cluster.assert_not_called()
    pd.testing.assert_frame_equal(actual, expected)
