"""Regression tests for contig and fragment header handling."""

from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from remag.clustering import cluster_contigs
from remag.models import SequenceDataset, generate_embeddings
from remag.utils import ContigHeaderMapper, extract_base_contig_name

SPADES_HEADER = "NODE_13_length_77980_cov_5.491044"


def test_extract_base_contig_name_preserves_spades_decimal_header():
    assert extract_base_contig_name(SPADES_HEADER) == SPADES_HEADER


def test_extract_base_contig_name_strips_remag_suffixes_from_spades_header():
    known_headers = {
        f"{SPADES_HEADER}.original",
        f"{SPADES_HEADER}.0",
        f"{SPADES_HEADER}.h1.0",
    }

    assert extract_base_contig_name(f"{SPADES_HEADER}.original") == SPADES_HEADER
    assert extract_base_contig_name(f"{SPADES_HEADER}.h1.0") == SPADES_HEADER
    assert (
        extract_base_contig_name(f"{SPADES_HEADER}.0", known_headers=known_headers)
        == SPADES_HEADER
    )


def test_plain_numeric_suffix_requires_fragment_context():
    assert extract_base_contig_name(f"{SPADES_HEADER}.0") == f"{SPADES_HEADER}.0"

    # Missing .original in known_headers
    known_headers = {f"{SPADES_HEADER}.0"}
    assert (
        extract_base_contig_name(f"{SPADES_HEADER}.0", known_headers=known_headers)
        == f"{SPADES_HEADER}.0"
    )


def test_contig_header_mapper_preserves_spades_decimal_header():
    fragments = {
        f"{SPADES_HEADER}.original": {"sequence": "ATCG", "length": 4},
        f"{SPADES_HEADER}.0": {"sequence": "ATCG", "length": 4},
    }

    mapper = ContigHeaderMapper(fragments)

    assert mapper.get_header(SPADES_HEADER) == f"{SPADES_HEADER}.original"
    assert mapper.get_header("NODE_13_length_77980_cov_5") is None


def test_sequence_dataset_groups_spades_fragments_by_full_decimal_header():
    features_df = pd.DataFrame(
        np.ones((3, 2)),
        index=[
            f"{SPADES_HEADER}.original",
            f"{SPADES_HEADER}.0",
            f"{SPADES_HEADER}.h1.0",
        ],
    )

    dataset = SequenceDataset(features_df, max_positive_pairs=10)

    assert list(dataset.contig_to_fragment_indices) == [SPADES_HEADER]
    assert len(dataset.contig_to_fragment_indices[SPADES_HEADER]) == 3


def test_generate_embeddings_removes_only_original_suffix(tmp_path):
    class DummyModel:
        def eval(self):
            return None

        def get_embedding(self, features):
            return torch.ones((features.shape[0], 2), device=features.device)

    internal_original_header = "contig.original_segment"
    features_df = pd.DataFrame(
        np.ones((2, 2)),
        index=[
            f"{SPADES_HEADER}.original",
            f"{internal_original_header}.original",
        ],
    )
    args = Namespace(output=str(tmp_path), batch_size=8, keep_intermediate=False)

    embeddings_df = generate_embeddings(DummyModel(), features_df, args)

    assert SPADES_HEADER in embeddings_df.index
    assert internal_original_header in embeddings_df.index


def test_cluster_contigs_preserves_clean_embedding_ids(tmp_path):
    header = "contig.original"
    embeddings_df = pd.DataFrame([[1.0, 0.0]], index=[header])
    args = Namespace(
        output=str(tmp_path),
        leiden_k_neighbors=1,
        leiden_similarity_threshold=0.0,
        greedy_resolutions=[0.1],
        greedy_max_contamination=0.1,
        cores=1,
        keep_intermediate=False,
    )

    with patch("remag.clustering._greedy_leiden_clustering", return_value=[0]):
        clusters_df = cluster_contigs(embeddings_df, {}, {}, args)

    assert clusters_df.loc[0, "contig"] == header
