"""Regressions for distinct core-gene copies within one contig."""

import itertools
import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from remag.clustering import _calculate_bin_quality
from remag.miniprot_utils import (
    _parse_paf_gene_mappings,
    check_core_gene_duplications,
    check_core_gene_duplications_from_cache,
    load_or_generate_gene_mappings,
    parse_and_cache_paf_files,
)
from remag.rescue import get_bin_scg_stats, rescue_fragmented_bins


def paf(
    start, end, gene="g_ref", contig="contig", strand="+", coverage=100, matches=300
):
    return (
        f"{gene}\t100\t0\t{coverage}\t{strand}\t{contig}\t10000\t"
        f"{start}\t{end}\t{matches}\t300\t0\n"
    )


def parse(tmp_path, lines, mappings=None):
    path = tmp_path / "alignments.paf"
    path.write_text("".join(lines))
    return _parse_paf_gene_mappings(path, 0.6, 0.4, mappings)


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([(0, 300), (0, 300)], 1),
        ([(0, 300), (30, 330)], 1),
        ([(0, 900), (300, 600)], 1),
        ([(0, 300), (1000, 1300)], 2),
        ([(0, 300), (300, 600)], 2),
        ([(0, 300), (1000, 1300), (2000, 2300)], 3),
    ],
    ids=["repeated", "overlap", "nested", "separate", "adjacent", "three-copies"],
)
def test_distinct_gene_locations(tmp_path, intervals, expected):
    mappings = parse(
        tmp_path,
        [paf(start, end, gene=f"g_ref{i}") for i, (start, end) in enumerate(intervals)],
    )
    result = check_core_gene_duplications_from_cache(
        pd.DataFrame({"contig": ["contig"], "cluster": ["bin1"]}),
        mappings,
        SimpleNamespace(output=str(tmp_path)),
    )
    assert result.loc[0, "total_core_genes_found"] == 1
    assert result.loc[0, "single_copy_genes_count"] == (1 if expected == 1 else 0)
    stats = json.loads((tmp_path / "core_gene_duplication_results.json").read_text())
    assert stats["bin1"]["duplicated_genes"] == (
        {"g": expected} if expected > 1 else {}
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_bridging_overlap_is_one_locus_in_any_order(tmp_path, order):
    lines = [paf(0, 300), paf(500, 800), paf(200, 600)]
    mappings = parse(tmp_path, [lines[i] for i in order])
    assert mappings["contig"]["g"]["loci"] == [[0, 800]]


def test_opposite_strand_overlap_is_counted_conservatively(tmp_path):
    mappings = parse(tmp_path, [paf(0, 300), paf(0, 300, strand="-")])
    assert mappings["contig"]["g"]["loci"] == [[0, 300]]


def test_thresholds_and_best_hit_metrics_are_preserved(tmp_path):
    mappings = parse(
        tmp_path,
        [
            paf(0, 300, coverage=60, matches=120),
            paf(1000, 1300, coverage=59),
            paf(2000, 2300, matches=119),
            paf(50, 350, coverage=80, matches=270),
        ],
    )
    info = mappings["contig"]["g"]
    assert info["loci"] == [[0, 350]]
    assert info["coverage"] == 0.8
    assert info["identity"] == 0.9
    assert info["score"] == pytest.approx(0.72)


@pytest.mark.parametrize("start,end", [(0, 0), (300, 0), (-1, 300), ("bad", 300)])
def test_invalid_target_positions_are_not_gene_copies(tmp_path, start, end):
    assert parse(tmp_path, [paf(start, end)]) == {}


def test_cache_round_trip_and_multi_file_merge_do_not_double_count(tmp_path):
    lines = [paf(0, 300), paf(1000, 1300)]
    for bin_name in ["bin1", "bin2"]:
        (tmp_path / f"{bin_name}.paf").write_text("".join(lines))
    args = SimpleNamespace(output=str(tmp_path), keep_intermediate=True)
    mappings = parse_and_cache_paf_files(tmp_path, {"bin1": [], "bin2": []}, args)
    assert mappings["contig"]["g"]["loci"] == [[0, 300], [1000, 1300]]
    with patch("remag.miniprot_utils.subprocess.run") as run:
        assert load_or_generate_gene_mappings({}, args) == mappings
    run.assert_not_called()


def test_reports_count_copies_but_binning_counts_contig_presence(tmp_path):
    mappings = parse(
        tmp_path,
        [
            paf(0, 300),
            paf(1000, 1300),
            paf(0, 300, contig="other"),
            paf(3000, 3300, gene="h_ref"),
        ],
    )
    score, scg, dups = _calculate_bin_quality(["contig", "other"], mappings)
    completeness, precision = 2 / 133, 2 / 3
    assert score == pytest.approx(
        2 * completeness * precision / (completeness + precision)
    )
    assert (scg, dups) == (1, 1)
    assert get_bin_scg_stats(["contig", "other"], mappings) == (50.0, 2)
    check_core_gene_duplications_from_cache(
        pd.DataFrame({"contig": ["contig", "other"], "cluster": ["bin1", "bin1"]}),
        mappings,
        SimpleNamespace(output=str(tmp_path)),
    )
    stats = json.loads((tmp_path / "core_gene_duplication_results.json").read_text())
    assert stats["bin1"] == {
        "has_duplications": True,
        "duplicated_genes": {"g": 3},
        "total_genes_found": 2,
        "single_copy_genes_count": 1,
        "within_contig_duplications": {"contig": {"g": 2}},
    }


def test_per_bin_annotation_matches_cached_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr("remag.miniprot_utils.check_miniprot_available", lambda: True)
    lines = [paf(0, 300), paf(1000, 1300)]

    def write_paf(*_args, stdout, **_kwargs):
        stdout.write("".join(lines))
        return SimpleNamespace(returncode=0)

    clusters = pd.DataFrame({"contig": ["contig"], "cluster": ["bin1"]})
    args = SimpleNamespace(
        output=str(tmp_path),
        cores=1,
        verbose=False,
        min_bin_size=1,
        keep_intermediate=True,
    )
    with patch("remag.miniprot_utils.subprocess.run", side_effect=write_paf):
        direct = check_core_gene_duplications(
            clusters, {"contig": {"sequence": "A" * 10000}}, args
        )
    before = (tmp_path / "core_gene_duplication_results.json").read_bytes()
    cached = check_core_gene_duplications_from_cache(
        clusters, args._gene_mappings_cache, args
    )
    pd.testing.assert_frame_equal(direct, cached)
    assert (tmp_path / "core_gene_duplication_results.json").read_bytes() == before
    assert json.loads(before)["bin1"]["duplicated_genes"] == {"g": 2}


@pytest.mark.parametrize("candidate_cluster", ["noise", "small"])
def test_within_contig_duplication_does_not_change_rescue(tmp_path, candidate_cluster):
    mappings = parse(
        tmp_path, [paf(0, 300, contig="candidate"), paf(1000, 1300, contig="candidate")]
    )
    mappings["anchor"] = {f"h{i}": {"loci": [[0, 300]]} for i in range(9)}
    clusters = pd.DataFrame(
        {"contig": ["anchor", "candidate"], "cluster": ["large", candidate_cluster]}
    )
    embeddings = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], index=clusters["contig"])
    fragments = {
        "anchor": {"sequence": "A" * 5000},
        "candidate": {"sequence": "A" * 2000},
    }
    result = rescue_fragmented_bins(
        clusters.copy(),
        embeddings,
        fragments,
        SimpleNamespace(_gene_mappings_cache=mappings),
    )
    assert result.set_index("contig")["cluster"].to_dict() == {
        "anchor": "large",
        "candidate": "large",
    }


@pytest.mark.parametrize("with_hits", [True, False])
def test_old_disk_cache_regenerates_annotations_only(tmp_path, with_hits):
    path = tmp_path / "gene_contig_mappings.json"
    path.write_text(json.dumps({"contig": {"g": {"score": 1.0}}}))
    preserved = {}
    for name in ["bins.csv", "embeddings.csv", "siamese_model.pt"]:
        (tmp_path / name).write_bytes(b"existing output")
        preserved[name] = (tmp_path / name).read_bytes()
    args = SimpleNamespace(
        output=str(tmp_path), cores=1, verbose=False, keep_intermediate=False
    )

    def write_paf(*_args, stdout, **_kwargs):
        if with_hits:
            stdout.write(paf(0, 300) + paf(1000, 1300))
        return SimpleNamespace(returncode=0)

    with patch(
        "remag.miniprot_utils.check_miniprot_available", return_value=True
    ), patch("remag.miniprot_utils.subprocess.run", side_effect=write_paf) as run:
        mappings = load_or_generate_gene_mappings(
            {"contig": {"sequence": "A" * 10000}}, args
        )
        assert run.call_count == 1
        assert load_or_generate_gene_mappings({}, args) == mappings
        assert run.call_count == 1
    if with_hits:
        assert mappings["contig"]["g"]["loci"] == [[0, 300], [1000, 1300]]
    else:
        assert mappings == {}
    assert json.loads(path.read_text()) == mappings
    assert all(
        (tmp_path / name).read_bytes() == data for name, data in preserved.items()
    )


def test_within_contig_copies_do_not_lower_clustering_score(tmp_path):
    mappings = parse(tmp_path, [paf(0, 300), paf(1000, 1300)])
    single_match = parse(tmp_path, [paf(0, 300)])
    assert _calculate_bin_quality(["contig"], mappings) == _calculate_bin_quality(
        ["contig"], single_match
    )
    assert get_bin_scg_stats(["contig"], mappings) == (0.0, 1)


def test_reported_within_contig_duplications_do_not_block_pipeline_rescue(tmp_path):
    from argparse import Namespace

    import numpy as np

    from remag.core import main

    names = ["anchor", "candidate", "unbinned"]
    fragments = {
        name: {"sequence": "A" * size} for name, size in zip(names, [5000, 2000, 1000])
    }
    mappings = parse(
        tmp_path,
        [
            paf(0, 300, contig="candidate"),
            paf(1000, 1300, contig="candidate"),
            paf(0, 300, contig="unbinned", gene="h_ref"),
            paf(600, 900, contig="unbinned", gene="h_ref"),
        ],
    )
    mappings["anchor"] = {f"other{i}": {"loci": [[0, 300]]} for i in range(9)}
    args = Namespace(
        fasta=str(tmp_path / "input.fa"),
        output=str(tmp_path / "output"),
        bam=None,
        tsv=None,
        verbose=False,
        keep_intermediate=False,
        skip_bacterial_filter=True,
        filter_only=False,
        min_contig_length=1,
        min_bin_size=1,
        num_augmentations=1,
        cores=1,
        skip_rescue=False,
    )
    with (
        patch("remag.core.setup_logging"),
        patch(
            "remag.core.get_features",
            return_value=(pd.DataFrame(np.ones((3, 136))), fragments),
        ),
        patch("remag.core.train_siamese_network", return_value=object()),
        patch(
            "remag.core.generate_embeddings",
            return_value=pd.DataFrame([[1.0, 0.0]] * 3, index=names),
        ),
        patch("remag.core.load_or_generate_gene_mappings", return_value=mappings),
        patch(
            "remag.core.cluster_contigs",
            return_value=pd.DataFrame(
                {"contig": names, "cluster": ["large", "small", "noise"]}
            ),
        ),
    ):
        main(args)
    output = tmp_path / "output"
    saved = pd.read_csv(output / "bins.csv")
    assert saved.set_index("contig")["cluster"].to_dict() == dict.fromkeys(
        names, "large"
    )
    stats = json.loads((output / "core_gene_duplication_results.json").read_text())
    assert stats["large"] == {
        "has_duplications": True,
        "duplicated_genes": {"g": 2, "h": 2},
        "total_genes_found": 11,
        "single_copy_genes_count": 9,
        "within_contig_duplications": {"candidate": {"g": 2}, "unbinned": {"h": 2}},
    }
