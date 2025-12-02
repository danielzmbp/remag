"""Tests for adaptive resolution selection."""

import numpy as np
import pandas as pd

from remag import adaptive_resolution


def test_coassembly_resolution_sweep_includes_single_sample_range(
    monkeypatch, sample_embeddings_df, sample_fragments_dict, mock_args
):
    """Coassemblies should test all single-sample resolutions plus higher ones."""
    mock_args.mode = "metagenomics"
    mock_args.bam = ["sample1.bam", "sample2.bam"]
    mock_args.tsv = []

    captured = {}

    def fake_test_multiple_resolutions(embeddings_df, gene_mappings_cache, args, test_resolutions, is_coassembly=False):
        captured["resolutions"] = list(test_resolutions)
        return 0.5, {}

    monkeypatch.setattr(
        adaptive_resolution,
        "test_multiple_resolutions",
        fake_test_multiple_resolutions,
    )

    # Provide minimal gene mappings to bypass organism estimation
    gene_mappings = {"contig_0": {"g1": {}, "g2": {}}}

    best_resolution = adaptive_resolution.determine_optimal_resolution(
        sample_embeddings_df,
        sample_fragments_dict,
        mock_args,
        gene_mappings=gene_mappings,
    )

    expected_resolutions = sorted([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0, 1.2, 1.5, 2.0])
    assert captured["resolutions"] == expected_resolutions
    assert best_resolution == 0.5


def test_coassembly_prefers_fewer_clusters_after_duplication_minimization(monkeypatch, mock_args):
    """After minimizing duplications, coassemblies should pick fewer clusters."""
    mock_args.mode = "metagenomics"
    mock_args.bam = ["sample1.bam", "sample2.bam"]
    mock_args.tsv = []

    embeddings_df = pd.DataFrame(
        np.random.randn(3, 2),
        index=["contig_a", "contig_b", "contig_c"],
        columns=["dim_0", "dim_1"],
    )

    monkeypatch.setattr(adaptive_resolution, "_construct_knn_graph", lambda *args, **kwargs: "graph")

    def fake_leiden(graph, resolution, random_state=42):
        return [0, 0, 1] if resolution < 1.0 else [0, 0, 0]

    monkeypatch.setattr(adaptive_resolution, "_leiden_clustering_on_graph", fake_leiden)

    def fake_duplication_check(df, cache, args):
        df = df.copy()
        df["single_copy_genes_count"] = 10
        df["duplicated_core_genes_count"] = 0
        df["has_duplicated_core_genes"] = False
        return df

    monkeypatch.setattr(
        adaptive_resolution,
        "check_core_gene_duplications_from_cache",
        fake_duplication_check,
    )

    best_resolution, results = adaptive_resolution.test_multiple_resolutions(
        embeddings_df,
        {},
        mock_args,
        [0.8, 1.0],
        is_coassembly=True,
    )

    assert best_resolution == 1.0
    assert results[0.8]["n_clusters"] == 2
    assert results[1.0]["n_clusters"] == 1
