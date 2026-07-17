"""Focused tests for top-level pipeline orchestration."""

from argparse import Namespace
from unittest.mock import patch


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
