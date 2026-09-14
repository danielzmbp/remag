"""Focused tests for top-level pipeline orchestration."""

from argparse import Namespace
from unittest.mock import patch

import pandas as pd
import pytest


def test_filtering_receives_save_filtered_contigs_flag(tmp_path):
    """Core forwards the save-filtered-contigs request to filtering."""
    from remag.core import main

    args = Namespace(
        output=str(tmp_path / "out"),
        verbose=False,
        fasta=str(tmp_path / "contigs.fasta"),
        min_contig_length=1000,
        hyenadna_batch_size=256,
        save_filtered_contigs=True,
        skip_bacterial_filter=False,
        filter_only=True,
        keep_intermediate=False,
    )

    with patch(
        "remag.core.filter_bacterial_contigs", return_value=args.fasta
    ) as mock_filter:
        main(args)

    mock_filter.assert_called_once_with(
        args.fasta,
        args.output,
        min_contig_length=args.min_contig_length,
        hyenadna_batch_size=args.hyenadna_batch_size,
        save_filtered_contigs=True,
    )


@pytest.mark.parametrize("failure_stage", ["mapping", "fallback"])
def test_annotation_failure_stops_pipeline(tmp_path, failure_stage):
    from remag.core import main

    args = Namespace(
        output=str(tmp_path),
        fasta=str(tmp_path / "input.fa"),
        verbose=False,
        skip_bacterial_filter=True,
        min_contig_length=1,
        num_augmentations=1,
        cores=1,
        bam=None,
        tsv=None,
    )
    clusters = pd.DataFrame({"contig": ["c"], "cluster": ["bin_0"]})
    with (
        patch("remag.core.setup_logging"),
        patch("remag.core.get_features", return_value=(pd.DataFrame([[1]]), {})),
        patch("remag.core.train_siamese_network"),
        patch("remag.core.generate_embeddings"),
        patch(
            "remag.core.load_or_generate_gene_mappings",
            return_value={},
            side_effect=(
                RuntimeError("annotation failed")
                if failure_stage == "mapping"
                else None
            ),
        ),
        patch("remag.core.cluster_contigs", return_value=clusters) as cluster,
        patch(
            "remag.core.check_core_gene_duplications_from_cache",
            side_effect=ValueError("cache check failed"),
        ),
        patch(
            "remag.core.check_core_gene_duplications",
            side_effect=RuntimeError("annotation failed"),
        ) as fallback,
        patch("remag.core.rescue_fragmented_bins") as rescue,
        patch("remag.core.save_clusters_as_fasta") as save,
    ):
        with pytest.raises(SystemExit) as caught:
            main(args)

    assert caught.value.code == 1
    assert cluster.call_count == (1 if failure_stage == "fallback" else 0)
    assert fallback.call_count == (1 if failure_stage == "fallback" else 0)
    rescue.assert_not_called()
    save.assert_not_called()
    assert not (tmp_path / "core_gene_duplication_results.json").exists()
