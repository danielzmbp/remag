"""Shared test fixtures for REMAG tests."""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_args(temp_dir):
    """Create mock arguments object for testing."""
    args = Mock()
    args.cores = 4
    args.verbose = True
    args.output = temp_dir  # Use actual temp directory
    args.min_bin_size = 100000
    args.batch_size = 32
    args.embedding_dim = 64
    args.epochs = 2
    args.max_positive_pairs = 100
    args.keep_intermediate = False
    # Add clustering-specific parameters with actual values
    args.leiden_k_neighbors = 15
    args.leiden_similarity_threshold = 0.1
    args.leiden_resolution = 1.0
    # Add greedy clustering parameters
    args.greedy_resolutions = [0.1, 0.5, 1.0, 2.0, 5.0]
    args.greedy_max_contamination = 0.1
    # Add rescue parameters
    args.rescue_max_duplication_increase = 5.0
    args.rescue_max_total_duplication = 5.0
    args.skip_rescue = False

    args.fasta = "test.fasta"
    return args


@pytest.fixture
def sample_features_df():
    """Create sample features DataFrame for testing."""
    np.random.seed(42)
    n_samples = 20
    n_kmer_features = 136
    n_coverage_features = 4

    # Create realistic k-mer features (0-1 normalized)
    kmer_features = np.random.dirichlet(np.ones(n_kmer_features), size=n_samples)

    # Create realistic coverage features (log-normal distribution)
    coverage_features = np.random.lognormal(
        mean=1.0, sigma=0.5, size=(n_samples, n_coverage_features)
    )

    # Combine features
    features = np.column_stack([kmer_features, coverage_features])

    # Create index with proper contig naming
    index = [f"contig_{i}.original" for i in range(n_samples)]

    # Create column names
    kmer_cols = [f"kmer_{i}" for i in range(n_kmer_features)]
    coverage_cols = [f"coverage_{i}" for i in range(n_coverage_features)]
    columns = kmer_cols + coverage_cols

    return pd.DataFrame(features, index=index, columns=columns)


@pytest.fixture
def sample_embeddings_df():
    """Create sample embeddings DataFrame for testing."""
    np.random.seed(42)
    n_contigs = 15
    embedding_dim = 64

    # Create normalized embeddings
    embeddings = np.random.randn(n_contigs, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = [f"contig_{i}" for i in range(n_contigs)]
    columns = [f"dim_{i}" for i in range(embedding_dim)]

    return pd.DataFrame(embeddings, index=index, columns=columns)


@pytest.fixture
def sample_fragments_dict():
    """Create sample fragments dictionary for testing."""
    fragments = {}
    for i in range(10):
        # Create sequences of different lengths
        seq_length = np.random.randint(1000, 5000)
        sequence = "".join(np.random.choice(["A", "T", "G", "C"], size=seq_length))

        fragments[f"contig_{i}.original"] = {"sequence": sequence, "length": seq_length}

    return fragments


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing without GPU dependency."""
    return torch.device("cpu")


@pytest.fixture
def sample_fasta_content():
    """Create sample FASTA content for testing."""
    return """>test_contig_1
ATCGATCGATCGATCGATCGATCGATCGATCGATCG
>test_contig_2
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
>test_contig_3
TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA
"""


@pytest.fixture
def create_test_fasta(temp_dir, sample_fasta_content):
    """Create a test FASTA file."""
    fasta_path = os.path.join(temp_dir, "test.fasta")
    with open(fasta_path, "w") as f:
        f.write(sample_fasta_content)
    return fasta_path
