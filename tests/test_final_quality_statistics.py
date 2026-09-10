"""Regression for statistics matching the final saved bins."""

import json
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from remag.core import main
from remag.miniprot_utils import check_core_gene_duplications_from_cache
from remag.utils import fasta_iter


@pytest.mark.parametrize(
    "scenario", ["merged", "skip_rescue", "below_size", "no_hits", "fallback"]
)
def test_quality_statistics_match_final_saved_bins(tmp_path, scenario):
    """Keep real rescue/output code while supplying controlled upstream results."""
    lengths = {"small": 1000, "large": 4000, "tiny": 400, "unbinned": 1000}
    fragments = {name: {"sequence": "A" * size} for name, size in lengths.items()}
    clusters = pd.DataFrame(
        {
            "contig": list(lengths),
            "cluster": ["bin_small", "bin_large", "bin_tiny", "noise"],
        }
    )
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], index=list(lengths)
    )
    mappings = {
        "large": {f"g{i}": {} for i in range(25)},
        "small": {"g0": {}, "g25": {}},
        "tiny": {"tiny_gene": {}},
        "unbinned": {"g26": {}},
    }
    if scenario == "no_hits":
        mappings = {}

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
        min_bin_size=10000 if scenario == "below_size" else 2000,
        num_augmentations=1,
        cores=1,
        skip_rescue=scenario == "skip_rescue",
    )

    def full_annotation(cluster_frame, _fragments, run_args, **_kwargs):
        # Preserve the fallback's role of populating mappings before rescue.
        run_args._gene_mappings_cache = mappings
        return check_core_gene_duplications_from_cache(
            cluster_frame, mappings, run_args
        )

    mapping_error = RuntimeError("mapping generation failed")
    with (
        patch("remag.core.setup_logging"),
        patch(
            "remag.core.get_features",
            return_value=(pd.DataFrame(np.ones((4, 136))), fragments),
        ),
        patch("remag.core.train_siamese_network", return_value=object()),
        patch("remag.core.generate_embeddings", return_value=embeddings),
        patch(
            "remag.core.load_or_generate_gene_mappings",
            return_value=mappings,
            side_effect=mapping_error if scenario == "fallback" else None,
        ),
        patch("remag.core.cluster_contigs", return_value=clusters.copy()),
        patch(
            "remag.core.check_core_gene_duplications", side_effect=full_annotation
        ) as fallback,
    ):
        main(args)

    assert fallback.call_count == (1 if scenario == "fallback" else 0)
    output = tmp_path / "output"
    saved = pd.read_csv(output / "bins.csv")
    if scenario == "below_size":
        expected_members = set()
    elif scenario in {"skip_rescue", "no_hits"}:
        expected_members = {"large"}
    else:
        expected_members = {"small", "large", "unbinned"}

    assert set(saved["contig"]) == expected_members
    expected_bins = {"bin_large"} if expected_members else set()
    assert set(saved["cluster"]) == expected_bins
    assert {path.stem for path in (output / "bins").glob("*.fa")} == expected_bins
    if expected_members:
        assert dict(fasta_iter(str(output / "bins" / "bin_large.fa"))) == {
            name: fragments[name]["sequence"] for name in expected_members
        }

    if scenario == "below_size":
        expected_stats = {}
    else:
        merged = scenario in {"merged", "fallback"}
        expected_stats = {
            "bin_large": {
                "has_duplications": merged,
                "duplicated_genes": {"g0": 2} if merged else {},
                "total_genes_found": (
                    0 if scenario == "no_hits" else 27 if merged else 25
                ),
                "single_copy_genes_count": (
                    0 if scenario == "no_hits" else 26 if merged else 25
                ),
            }
        }
    stats_path = output / "core_gene_duplication_results.json"
    assert json.loads(stats_path.read_text()) == expected_stats
