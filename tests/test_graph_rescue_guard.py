"""The tested graph requirement for whole-bin merging and cached graph context."""

import json
from argparse import Namespace
from unittest.mock import patch

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from remag.clustering import cluster_contigs
from remag.output import file_sha256
from remag.rescue import _graph_bin_support, rescue_fragmented_bins


def named_graph(names, edges=()):
    return ig.Graph(n=len(names), edges=edges, vertex_attrs={"name": names})


def rescue(names, labels, vectors, lengths, graph, genes=None):
    frame = pd.DataFrame({"contig": names, "cluster": labels})
    emb = pd.DataFrame(vectors, index=names)
    fragments = {c: {"sequence": "A" * length} for c, length in zip(names, lengths)}
    return (
        rescue_fragmented_bins(
            frame,
            emb,
            fragments,
            Namespace(_gene_mappings_cache=genes or {}),
            graph=graph,
        )
        .set_index("contig")["cluster"]
        .to_dict()
    )


@pytest.mark.parametrize("edges", [[], [(1, 0)], [(0, 1), (1, 0)]])
def test_one_original_edge_is_sufficient_without_weight_or_reciprocity(edges):
    graph = named_graph(["s", "t"], edges)
    graph.es["weight"] = [0.1] * graph.ecount()
    result = rescue(["s", "t"], ["S", "T"], [[1, 0]] * 2, [1000, 4000], graph)
    assert result == {"s": "T" if edges else "S", "t": "T"}


def test_unsupported_best_target_does_not_try_supported_second_target():
    names = ["s", "nearest", "second"]
    graph = named_graph(names, [(0, 2)])
    result = rescue(
        names, ["S", "T", "U"], [[1, 0], [1, 0], [0.8, 0.6]], [1000, 4000, 5000], graph
    )
    assert result == dict(zip(names, ["S", "T", "U"]))


def test_chained_merge_uses_connections_from_an_absorbed_bin():
    names = ["a", "b", "c"]
    angles = np.deg2rad([0, 40, 85])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    # a joins b first; only absorbed a has an edge to c. b's center stays fixed.
    result = rescue(
        names, names, vectors, [1000, 2000, 3000], named_graph(names, [(0, 1), (0, 2)])
    )
    assert result == dict.fromkeys(names, "c")


@pytest.mark.parametrize("supported", [False, True])
def test_nonworsening_marker_exception_still_requires_graph_support(supported):
    names = ["t1", "t2", "s"]
    genes = {
        "t1": {f"g{i}": {} for i in range(20)},
        "t2": {"g0": {}, "g1": {}},
        "s": {"g0": {}},
    }
    graph = named_graph(names, [(0, 2)] if supported else [])
    result = rescue(
        names, ["T", "T", "S"], [[1, 0]] * 3, [10000, 10000, 1000], graph, genes
    )
    assert result["s"] == ("T" if supported else "S")


@pytest.mark.parametrize("similarity,accepted", [(0.699999, False), (0.70, True)])
def test_supported_merge_retains_inclusive_similarity_boundary(similarity, accepted):
    with patch("remag.rescue.cosine_similarity", return_value=np.array([[similarity]])):
        result = rescue(
            ["s", "t"],
            ["S", "T"],
            [[1, 0]] * 2,
            [1000, 4000],
            named_graph(["s", "t"], [(0, 1)]),
        )
    assert result["s"] == ("T" if accepted else "S")


def test_noise_edges_do_not_supply_bin_support_but_recruitment_remains_available():
    names = ["s", "t", "noise"]
    graph = named_graph(names, [(0, 2), (1, 2)])
    result = rescue(names, ["S", "T", "noise"], [[1, 0]] * 3, [1000, 4000, 1000], graph)
    assert result == {"s": "S", "t": "T", "noise": "S"}


@pytest.mark.parametrize("empty_core", [False, True])
def test_valid_edgeless_graph_blocks_merges_and_allows_recruitment(empty_core):
    names = ["s", "t", "short"]
    graph = named_graph([] if empty_core else names[:2])
    result = rescue(names, ["S", "T", "noise"], [[1, 0]] * 3, [1000, 4000, 1000], graph)
    assert result == {"s": "S", "t": "T", "short": "S"}


def test_graph_subset_order_and_literal_names_are_used_instead_of_embedding_indices():
    names = ["short", "001", "NA", "null"]
    # Edge indices refer to NA/null, not short/001 or 001/NA in the full table.
    graph = named_graph(["NA", "null", "001"], [(0, 1)])
    result = rescue(
        names, ["noise", "A", "B", "C"], [[1, 0]] * 4, [1000, 10000, 5000, 1000], graph
    )
    assert result == {"short": "A", "001": "A", "NA": "B", "null": "B"}


@pytest.mark.parametrize(
    "graph",
    [None, ig.Graph(n=2), named_graph(["s", "unknown"]), named_graph(["s", "s"])],
)
def test_missing_or_invalid_graph_cannot_bypass_guard(graph):
    with pytest.raises(ValueError, match="graph"):
        rescue(["s", "t"], ["S", "T"], [[1, 0]] * 2, [1000, 4000], graph)


def test_support_is_mapped_to_assignments_at_rescue_entry():
    graph = named_graph(["s", "t"], [(0, 1)])
    before = pd.DataFrame({"contig": ["s", "t"], "cluster": ["old", "old"]})
    after = before.assign(cluster=["S", "T"])
    assert _graph_bin_support(graph, before) == set()
    assert _graph_bin_support(graph, after) == {frozenset(("S", "T"))}


def cache_inputs(tmp_path, core_count=3, keep=False):
    names = ["short", "001", "NA", "null"][: core_count + 1]
    emb = pd.DataFrame([[1, 0]] * len(names), index=names)
    fragments = {c: {"sequence": "A" * (1000 if c == "short" else 3000)} for c in names}
    args = Namespace(
        output=str(tmp_path),
        min_contig_length=1000,
        graph_min_contig_length=3000,
        keep_intermediate=keep,
        cores=1,
    )
    return emb, fragments, args


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("core_count", [0, 1, 3])
def test_fresh_and_cached_graphs_survive_without_optional_intermediates(
    tmp_path, keep, core_count
):
    emb, fragments, args = cache_inputs(tmp_path, core_count, keep)
    fresh, graph = cluster_contigs(emb, fragments, {}, args, return_graph=True)
    paths = [
        tmp_path / name
        for name in (
            "knn_graph_edges.csv",
            "knn_graph_contigs.csv",
            "knn_graph_stats.json",
        )
    ]
    assert all(p.exists() for p in paths)
    assert graph.vs["name"] == list(emb.index[1:])
    assert graph.vcount() == core_count
    if core_count < 2:
        assert graph.ecount() == 0
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with patch(
        "remag.clustering.NearestNeighbors.fit",
        side_effect=AssertionError("must not rebuild graph"),
    ), patch(
        "remag.clustering._greedy_leiden_clustering",
        side_effect=AssertionError("must not recluster"),
    ):
        cached, loaded = cluster_contigs(emb, fragments, {}, args, return_graph=True)
    pd.testing.assert_frame_equal(fresh, cached)
    assert loaded.vs["name"] == graph.vs["name"]
    assert loaded.get_edgelist() == graph.get_edgelist()
    assert loaded.es["weight"] == graph.es["weight"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_original_graph_support_is_the_same_on_fresh_and_cached_rescue(tmp_path):
    emb, fragments, args = cache_inputs(tmp_path)
    args._gene_mappings_cache = {}
    with patch("remag.clustering._greedy_leiden_clustering", return_value=[0, 1, 2]):
        frame, graph = cluster_contigs(emb, fragments, {}, args, return_graph=True)
    assert graph.vs["name"] == ["001", "NA", "null"]
    fresh = rescue_fragmented_bins(frame, emb, fragments, args, graph=graph)
    cached, loaded = cluster_contigs(emb, fragments, {}, args, return_graph=True)
    replay = rescue_fragmented_bins(cached, emb, fragments, args, graph=loaded)
    pd.testing.assert_frame_equal(fresh, replay)
    assert fresh.cluster.tolist() == ["bin_1"] * 4


@pytest.mark.parametrize(
    "change",
    [
        "missing_all",
        "missing_nodes",
        "missing_edges",
        "node_order",
        "malformed_stats",
        "changed_graph",
        "old_algorithm",
    ],
)
def test_missing_incompatible_or_replaced_graph_preserves_cached_outputs(
    tmp_path, change
):
    emb, fragments, args = cache_inputs(tmp_path)
    cluster_contigs(emb, fragments, {}, args)
    if change == "missing_all":
        for path in tmp_path.glob("knn_graph_*"):
            path.unlink()
    elif change in ("missing_nodes", "missing_edges"):
        (
            tmp_path
            / (
                "knn_graph_contigs.csv"
                if change == "missing_nodes"
                else "knn_graph_edges.csv"
            )
        ).unlink()
    elif change == "node_order":
        (tmp_path / "knn_graph_contigs.csv").write_text("contig\nnull\nNA\n001\n")
    elif change == "malformed_stats":
        (tmp_path / "knn_graph_stats.json").write_text("[]")
    elif change == "changed_graph":
        path = tmp_path / "knn_graph_edges.csv"
        path.write_text("source,target,weight\n0,1,1.000000\n")
        stats_path = tmp_path / "knn_graph_stats.json"
        stats = json.loads(stats_path.read_text())
        stats.update(edges_sha256=file_sha256(path), n_edges=1)
        stats_path.write_text(
            json.dumps(stats)
        )  # Even internally valid replacement is rejected.
    else:
        path = tmp_path / "clustering_provenance.json"
        data = json.loads(path.read_text())
        data["settings"]["rescue_algorithm"] = "unified-v1"
        path.write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with patch(
        "remag.clustering.NearestNeighbors.fit",
        side_effect=AssertionError("must not rebuild"),
    ), pytest.raises(ValueError, match="new output directory.*--force"):
        cluster_contigs(emb, fragments, {}, args)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_core_uses_assignments_after_duplication_check_for_graph_guard(tmp_path):
    from remag import core

    names = ["s", "t"]
    emb = pd.DataFrame([[1, 0]] * 2, index=names)
    fragments = {"s": {"sequence": "A" * 1000}, "t": {"sequence": "C" * 4000}}
    initial = pd.DataFrame({"contig": names, "cluster": ["old", "old"]})
    checked = initial.assign(cluster=["S", "T"])
    args = Namespace(
        output=str(tmp_path / "out"),
        fasta=str(tmp_path / "input.fa"),
        min_contig_length=1000,
        min_bin_size=1,
        verbose=False,
        bam=None,
        tsv=None,
        num_augmentations=0,
        cores=1,
        skip_bacterial_filter=True,
    )
    with patch("remag.core.setup_logging"), patch(
        "remag.core.get_features", return_value=(pd.DataFrame([[1]]), fragments)
    ), patch("remag.core.train_siamese_network"), patch(
        "remag.core.generate_embeddings", return_value=emb
    ), patch(
        "remag.core.load_or_generate_gene_mappings", return_value={}
    ), patch(
        "remag.core.cluster_contigs",
        return_value=(initial, named_graph(names, [(0, 1)])),
    ) as cluster, patch(
        "remag.core.check_core_gene_duplications_from_cache", return_value=checked
    ):
        core.main(args)
    assert cluster.call_args.kwargs["return_graph"] is True
    saved = pd.read_csv(tmp_path / "out/bins.csv", dtype=str, keep_default_na=False)
    assert saved.cluster.tolist() == ["T", "T"]
