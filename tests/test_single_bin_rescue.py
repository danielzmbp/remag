"""Regression tests for rescuing contigs when only one bin exists."""

from types import SimpleNamespace

import pandas as pd
import pytest

from remag.rescue import rescue_fragmented_bins


@pytest.fixture
def rescue_input():
    clusters = pd.DataFrame(
        {"contig": ["anchor", "candidate"], "cluster": ["bin1", "noise"]}
    )
    embeddings = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], index=clusters["contig"])
    fragments = {name: {"sequence": "A" * 1000} for name in clusters["contig"]}
    args = SimpleNamespace(
        _gene_mappings_cache={
            "anchor": {f"g{i}": {} for i in range(50)},
            "candidate": {"extra": {}},
        }
    )
    return clusters, embeddings, fragments, args


@pytest.mark.parametrize(
    "vector, genes, max_increase, max_total, expected",
    [
        ([1.0, 0.0], ["extra"], 5.0, 5.0, "bin1"),
        ([1.0, 0.0], [], 5.0, 5.0, "bin1"),
        ([0.0, 1.0], ["extra"], 5.0, 5.0, "noise"),
        ([1.0, 0.0], ["g0", "g1"], 3.0, 5.0, "noise"),
        ([1.0, 0.0], ["g0", "g1", "g2"], 10.0, 5.0, "noise"),
        ([1.0, 0.0], ["g0", "g1"], 4.0, 10.0, "noise"),
    ],
    ids=[
        "compatible",
        "no-core-genes",
        "dissimilar",
        "increase-limit",
        "total-limit",
        "increase-boundary",
    ],
)
def test_one_bin_rescue_preserves_similarity_and_gene_limits(
    rescue_input, vector, genes, max_increase, max_total, expected
):
    clusters, embeddings, fragments, args = rescue_input
    embeddings.loc["candidate"] = vector
    args._gene_mappings_cache["candidate"] = {gene: {} for gene in genes}

    actual = rescue_fragmented_bins(
        clusters,
        embeddings,
        fragments,
        args,
        max_duplication_increase=max_increase,
        max_total_duplication=max_total,
    ).set_index("contig")["cluster"]

    assert actual.to_dict() == {"anchor": "bin1", "candidate": expected}


@pytest.mark.parametrize(
    "scenario",
    [
        "no-bins",
        "no-gene-mappings",
        "missing-bin-embedding",
        "missing-noise-embedding",
        "no-noise",
    ],
)
def test_one_bin_rescue_handles_missing_inputs(rescue_input, scenario):
    clusters, embeddings, fragments, args = rescue_input
    if scenario == "no-bins":
        clusters["cluster"] = "noise"
    elif scenario == "no-gene-mappings":
        args._gene_mappings_cache = {}
    elif scenario == "missing-bin-embedding":
        embeddings = embeddings.drop(index="anchor")
    elif scenario == "missing-noise-embedding":
        embeddings = embeddings.drop(index="candidate")
    elif scenario == "no-noise":
        clusters = clusters.iloc[:1].copy()
    expected = clusters.copy(deep=True)

    actual = rescue_fragmented_bins(clusters, embeddings, fragments, args)

    pd.testing.assert_frame_equal(actual, expected)


def test_one_bin_rescue_checks_cumulative_gene_duplication(rescue_input):
    clusters, embeddings, fragments, args = rescue_input
    clusters.loc[len(clusters)] = ["second", "noise"]
    embeddings.loc["second"] = [1.0, 0.0]
    fragments["second"] = {"sequence": "A" * 1000}
    args._gene_mappings_cache["second"] = {"extra": {}}

    actual = rescue_fragmented_bins(
        clusters,
        embeddings,
        fragments,
        args,
        max_duplication_increase=1.0,
    ).set_index("contig")["cluster"]

    assert actual.to_dict() == {
        "anchor": "bin1",
        "candidate": "bin1",
        "second": "noise",
    }


def test_rescue_does_not_require_an_unrelated_second_bin(rescue_input):
    clusters, embeddings, fragments, args = rescue_input
    one_bin = rescue_fragmented_bins(clusters.copy(), embeddings, fragments, args)
    clusters.loc[len(clusters)] = ["unrelated", "bin2"]
    embeddings.loc["unrelated"] = [0.0, 1.0]
    fragments["unrelated"] = {"sequence": "A" * 2000}
    args._gene_mappings_cache["unrelated"] = {"unrelated_gene": {}}

    two_bins = rescue_fragmented_bins(clusters, embeddings, fragments, args)

    pd.testing.assert_frame_equal(one_bin, two_bins.iloc[:2])
    assert two_bins.iloc[2]["cluster"] == "bin2"
