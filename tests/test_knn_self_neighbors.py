"""Regression tests for excluding each contig from its own neighbors."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from remag.clustering import _construct_knn_graph


@pytest.mark.parametrize("k", [1, 2, 15])
@pytest.mark.parametrize("case", ["duplicate-pair", "all-identical", "zero-vectors"])
def test_tied_neighbors_exclude_self_and_keep_requested_count(case, k):
    if case == "duplicate-pair":
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elif case == "all-identical":
        embeddings = np.tile([1.0, 0.0], (5, 1))
    else:
        embeddings = np.zeros((4, 2))

    graph = _construct_knn_graph(embeddings, k=k, similarity_threshold=0.0)

    assert graph.vcount() == len(embeddings)
    assert not any(graph.is_loop())
    assert graph.ecount() == len(embeddings) * min(k, len(embeddings) - 1)
    assert all(count <= 2 for count in graph.count_multiple())
    if case == "all-identical":
        assert graph.es["weight"] == [1.0] * graph.ecount()
    elif case == "zero-vectors":
        assert graph.es["weight"] == [0.0] * graph.ecount()


def test_unique_neighbors_preserve_edges_and_weights():
    embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])

    graph = _construct_knn_graph(embeddings, k=1, similarity_threshold=0.0)

    assert graph.get_edgelist() == [(0, 1), (0, 1), (1, 2), (2, 3)]
    assert graph.es["weight"] == pytest.approx([0.8, 0.8, 0.6, 0.0])


@pytest.mark.parametrize("count", [2, 5])
def test_zero_neighbors_still_produces_an_empty_graph(count):
    graph = _construct_knn_graph(np.ones((count, 2)), k=0)

    assert graph.vcount() == count
    assert graph.ecount() == 0


def test_saved_and_reloaded_graph_has_no_self_neighbors(tmp_path):
    embeddings = np.tile([1.0, 0.0], (5, 1))
    args = SimpleNamespace(output=str(tmp_path), keep_intermediate=True)

    graph = _construct_knn_graph(embeddings, k=2, args=args)
    saved = pd.read_csv(tmp_path / "knn_graph_edges.csv", comment="#")
    reloaded = _construct_knn_graph(embeddings, k=2, args=args)

    assert (saved["source"] != saved["target"]).all()
    assert not any(reloaded.is_loop())
    assert reloaded.get_edgelist() == graph.get_edgelist()
    assert reloaded.es["weight"] == graph.es["weight"]
