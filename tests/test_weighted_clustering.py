"""Behavioral regressions for cosine-weighted greedy Leiden clustering."""

from types import SimpleNamespace

import numpy as np
import pytest

from remag.clustering import _construct_knn_graph, _greedy_leiden_clustering


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("background", [0.0, 0.1])
def test_similar_pairs_stay_separate_across_greedy_iterations(
    tmp_path, cached, background
):
    # All six nodes are connected, but each identical pair has a stronger edge.
    # Unit edge weights merge all six; weighted Leiden recovers all three pairs.
    embeddings = np.repeat(np.eye(3) + background, 2, axis=0)
    names = [f"c{i}" for i in range(6)]
    genes = {name: {f"g{i % 2}": {}} for i, name in enumerate(names)}
    args = SimpleNamespace(output=str(tmp_path), keep_intermediate=cached)
    if cached:
        graph = _construct_knn_graph(
            embeddings, k=5, similarity_threshold=0.0, args=args
        )
        assert graph.ecount() == 30

    labels = _greedy_leiden_clustering(
        embeddings,
        names,
        genes,
        k=5,
        similarity_threshold=0.0,
        resolutions=[0.5, 1.0, 2.0],
        args=args,
    )

    assert all(labels >= 0)
    groups = {frozenset(np.flatnonzero(labels == label)) for label in set(labels)}
    assert groups == {frozenset([0, 1]), frozenset([2, 3]), frozenset([4, 5])}


def test_zero_similarity_does_not_create_a_bin():
    labels = _greedy_leiden_clustering(
        np.zeros((4, 2)),
        ["a", "b", "c", "d"],
        {},
        k=2,
        similarity_threshold=0.0,
    )
    assert labels.tolist() == [-1, -1, -1, -1]
