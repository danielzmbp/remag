"""Scientific invariants of the frozen unified rescue and its pipeline integration."""

from argparse import Namespace
from itertools import combinations
from unittest.mock import patch

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from remag.rescue import (
    _merge_fragmented_bins,
    _recruit_contigs,
    rescue_fragmented_bins,
)


def run_rescue(names, vectors, labels, lengths, genes, **kwargs):
    embeddings = pd.DataFrame(vectors, index=names)
    clusters = pd.DataFrame({"contig": list(labels), "cluster": list(labels.values())})
    fragments = {c: {"sequence": "A" * lengths[c]} for c in names}
    graph = ig.Graph(
        n=len(labels),
        edges=list(combinations(range(len(labels)), 2)),
        vertex_attrs={"name": list(labels)},
    )
    result = rescue_fragmented_bins(
        clusters,
        embeddings,
        fragments,
        Namespace(_gene_mappings_cache=genes),
        graph=graph,
        **kwargs,
    )
    return result.set_index("contig")["cluster"].to_dict()


def test_combined_pool_interleaves_short_and_long_candidates():
    names = ["seed", "short", "long"]
    labels = {"seed": "A", "long": "noise"}  # Old pre-rescue core rows only.
    genes = {
        "seed": {f"g{i}": {} for i in range(30)},
        "short": {"g0": {}},
        "long": {"g1": {}},
    }
    result = run_rescue(
        names,
        [[1.0, 0.0]] * 3,
        labels,
        {"seed": 10000, "short": 1000, "long": 3000},
        genes,
    )
    assert result == {"seed": "A", "long": "noise", "short": "A"}


def test_marker_counts_accumulate_with_strict_increase_and_inclusive_total():
    names = ["seed", "x", "y", "z"]
    labels = dict(zip(names, ["A", "noise", "noise", "noise"]))
    genes = {
        "seed": {f"g{i}": {} for i in range(40)},
        "x": {"g0": {}},
        "y": {"g1": {}},
        "z": {"g2": {}},
    }
    result = run_rescue(
        names, [[1.0, 0.0]] * 4, labels, dict.fromkeys(names, 3000), genes
    )
    assert result == {"seed": "A", "x": "A", "y": "A", "z": "noise"}
    genes["seed"] = {f"g{i}": {} for i in range(20)}
    result = run_rescue(
        names, [[1.0, 0.0]] * 4, labels, dict.fromkeys(names, 3000), genes
    )
    assert all(result[c] == "noise" for c in names[1:])  # Exactly +5 is rejected.


@pytest.mark.parametrize("similarity,accepted", [(0.949999, False), (0.95, True)])
def test_nonworsening_exception_is_only_for_whole_bins(similarity, accepted):
    names = ["t1", "t2", "source"]
    frame = pd.DataFrame({"contig": names, "cluster": ["T", "T", "S"]})
    emb = pd.DataFrame([[1.0, 0.0]] * 3, index=names)
    lengths = dict(t1=10000, t2=10000, source=1000)
    genes = {
        "t1": {f"g{i}": {} for i in range(20)},
        "t2": {"g0": {}, "g1": {}},
        "source": {"g0": {}},
    }
    with patch("remag.rescue.cosine_similarity", return_value=np.array([[similarity]])):
        merged = _merge_fragmented_bins(
            frame, emb, lengths, genes, 0.7, 5.0, 5.0, {frozenset(("T", "S"))}
        )
    assert merged.iloc[-1]["cluster"] == ("T" if accepted else "S")
    recruited = _recruit_contigs({"T": ["t1", "t2"]}, ["source"], emb, genes, lengths)
    assert recruited == {"T": ["t1", "t2"]}  # No singleton exception above the ceiling.
    genes["source"] = {}
    assert (
        _recruit_contigs({"T": ["t1", "t2"]}, ["source"], emb, genes, lengths)["T"]
        == names
    )


def test_recruitment_centroids_stay_fixed_after_large_addition():
    names = ["a", "b", "x", "y"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.72, 0.69], [0.6, 0.8]]
    result = run_rescue(
        names,
        vectors,
        dict(zip(names, ["A", "B", "noise", "noise"])),
        dict(a=3000, b=3000, x=100000, y=3000),
        {},
    )
    assert result == dict(a="A", b="B", x="A", y="B")


def test_ties_follow_target_order_without_alphabetical_sorting():
    names = ["z_seed", "a_seed", "free"]
    genes = {c: {f"g{i}": {} for i in range(20)} for c in names[:2]}
    result = run_rescue(
        names,
        [[1.0, 0.0]] * 3,
        dict(zip(names, ["z_bin", "a_bin", "noise"])),
        dict.fromkeys(names, 3000),
        genes,
    )
    assert result["free"] == "z_bin"


def test_nearest_target_marker_failure_does_not_try_second_best():
    names = ["a", "b", "x"]
    genes = {
        "a": {f"g{i}": {} for i in range(20)},
        "b": {f"h{i}": {} for i in range(20)},
        "x": {"g0": {}},
    }
    emb = pd.DataFrame([[1.0, 0.0], [0.8, 0.6], [1.0, 0.0]], index=names)
    result = _recruit_contigs(
        {"A": ["a"], "B": ["b"]}, ["x"], emb, genes, dict.fromkeys(names, 3000)
    )
    assert result == {"A": ["a"], "B": ["b"]}


def test_no_seeds_cannot_create_short_only_bins():
    result = run_rescue(["s"], [[1.0, 0.0]], {}, {"s": 1000}, {})
    assert result == {"s": "noise"}


@pytest.mark.parametrize("kind", ["embedding", "assignment", "ambiguous"])
def test_duplicate_or_ambiguous_ids_are_rejected(kind):
    emb = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], index=["c", "d"])
    frame = pd.DataFrame({"contig": ["c", "d"], "cluster": ["A", "noise"]})
    fr = {c: {"sequence": "A" * 1000} for c in emb.index}
    if kind == "embedding":
        emb.index = ["c", "c"]
    elif kind == "assignment":
        frame["contig"] = ["c", "c"]
    else:
        fr["c.original"] = fr.pop("c")  # Must not guess an alias for c.
    with pytest.raises(ValueError):
        rescue_fragmented_bins(frame, emb, fr, Namespace(_gene_mappings_cache={}))


def test_real_fasta_names_with_fragment_like_suffixes_remain_distinct():
    names = ["c.original", "c"]
    result = run_rescue(
        names, [[1.0, 0.0]] * 2, {names[0]: "A"}, dict.fromkeys(names, 3000), {}
    )
    assert result == {names[0]: "A", names[1]: "A"}


@pytest.mark.parametrize("skip", [False, True])
def test_small_seed_crosses_export_size_only_after_recruitment(tmp_path, skip):
    import json

    from remag import core
    from remag.utils import fasta_iter

    names = ["seed", "short"]
    fr = {"seed": {"sequence": "A" * 499000}, "short": {"sequence": "C" * 2000}}
    emb = pd.DataFrame([[1.0, 0.0]] * 2, index=names)
    genes = {"seed": {"g0": {}}, "short": {"g1": {}}}
    args = Namespace(
        output=str(tmp_path / "out"),
        fasta=str(tmp_path / "input.fa"),
        bam=None,
        tsv=None,
        min_contig_length=1000,
        graph_min_contig_length=3000,
        min_bin_size=500000,
        verbose=False,
        num_augmentations=0,
        cores=1,
        skip_bacterial_filter=True,
        skip_rescue=skip,
        keep_intermediate=True,
    )
    with (
        patch("remag.core.setup_logging"),
        patch("remag.core.get_features", return_value=(pd.DataFrame([[1]]), fr)),
        patch("remag.core.train_siamese_network"),
        patch("remag.core.generate_embeddings", return_value=emb),
        patch("remag.core.load_or_generate_gene_mappings", return_value=genes),
        patch(
            "remag.clustering._greedy_leiden_clustering", return_value=[0]
        ) as cluster,
    ):
        core.main(args)
        assert cluster.call_args.kwargs["contig_names"] == ["seed"]
        cluster.reset_mock()
        core.main(args)  # Rerun from pre-rescue assignments, never final bins.csv.
        cluster.assert_not_called()
    saved = pd.read_csv(tmp_path / "out/bins.csv")
    provenance = pd.read_csv(tmp_path / "out/rescue_assignments.csv")
    assert provenance.contig.tolist() == names
    stats = json.loads(
        (tmp_path / "out/core_gene_duplication_results.json").read_text()
    )
    if skip:
        assert saved.empty and stats == {}
        assert not list((tmp_path / "out/bins").glob("*.fa"))
    else:
        assert saved.contig.tolist() == names
        assert dict(fasta_iter(str(tmp_path / "out/bins/bin_0.fa"))) == {
            c: fr[c]["sequence"] for c in names
        }
        assert stats["bin_0"]["total_genes_found"] == 2
        assert stats["bin_0"]["duplicated_genes"] == {}
        assert provenance.exported.all()


def test_equal_size_merges_keep_original_bin_order():
    names = ["z", "a", "b"]
    result = run_rescue(
        names,
        [[1.0, 0.0]] * 3,
        {c: f"bin_{c}" for c in names},
        dict.fromkeys(names, 1000),
        {},
    )
    # z merges into a first. a then grows too large to merge into b.
    assert result == dict.fromkeys(names, "bin_a")


def test_merge_centroids_stay_fixed_while_target_sizes_grow():
    names = ["a", "b", "c"]
    angles = np.deg2rad([0.0, 40.0, 85.0])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    result = run_rescue(
        names, vectors, dict(zip(names, names)), dict(a=1000, b=2000, c=3000), {}
    )
    # a joins b; b uses its original 40-degree center and updated 3 kb size
    # when merging into c. Refreshing b's center would reject that second merge.
    assert result == dict.fromkeys(names, "c")
