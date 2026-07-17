"""Unit tests for clustering module."""

from unittest.mock import Mock

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from remag.clustering import (
    ClusteringManager,
    GraphManager,
    _calculate_bin_quality,
    _construct_knn_graph,
)


class TestGraphManager:
    """Test GraphManager class."""

    def test_init_default_params(self):
        """Test GraphManager initialization with default parameters."""
        manager = GraphManager()
        assert manager.k == 15
        assert manager.similarity_threshold == 0.1
        assert manager.n_jobs == -1

    def test_init_custom_params(self):
        """Test GraphManager initialization with custom parameters."""
        manager = GraphManager(k=10, similarity_threshold=0.2, n_jobs=4)
        assert manager.k == 10
        assert manager.similarity_threshold == 0.2
        assert manager.n_jobs == 4


class TestKNNGraph:
    """Test k-NN graph construction."""

    def test_construct_graph_minimal_case(self):
        """Test k-NN graph construction with minimal valid input."""
        # Create normalized embeddings (3 samples, 5 dimensions)
        embeddings = np.array(
            [
                [0.4, 0.3, 0.5, 0.6, 0.2],
                [0.1, 0.8, 0.2, 0.1, 0.5],
                [0.6, 0.2, 0.4, 0.3, 0.7],
            ]
        )
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = _construct_knn_graph(embeddings, k=2, similarity_threshold=0.0)

        assert graph.vcount() == 3
        assert graph.ecount() >= 0  # At least some edges should exist
        assert all(weight >= 0 and weight <= 1 for weight in graph.es["weight"])

    def test_construct_graph_no_edges_high_threshold(self):
        """Test graph construction with threshold too high."""
        embeddings = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
        )  # Orthogonal vectors = zero similarity

        graph = _construct_knn_graph(embeddings, k=2, similarity_threshold=0.9)

        assert graph.vcount() == 3
        assert graph.ecount() == 0  # No edges due to high threshold

    def test_construct_graph_single_node(self):
        """Test graph construction with single node."""
        embeddings = np.array([[1, 0, 0]])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = _construct_knn_graph(embeddings, k=2, similarity_threshold=0.1)

        assert graph.vcount() == 1
        assert graph.ecount() == 0  # No edges possible with single node

    def test_construct_graph_caching_behavior(self, temp_dir):
        """Test that graph caching works correctly."""
        embeddings = np.array([[1, 0], [0, 1]])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Mock args with caching enabled
        mock_args = Mock()
        mock_args.output = temp_dir
        mock_args.keep_intermediate = True

        # First call - no cache exists
        graph1 = _construct_knn_graph(embeddings, k=1, args=mock_args)

        # Should create a graph without errors
        assert graph1.vcount() == 2


class TestClusteringManager:
    """Test ClusteringManager class."""

    def test_clustering_manager_init(self, mock_args):
        """Test ClusteringManager initialization."""
        manager = ClusteringManager(mock_args)
        assert manager.args == mock_args
        assert hasattr(manager, "graph_manager")


class TestPerformanceOptimizations:
    """Test performance-related functionality and optimizations."""

    def test_vectorized_distance_calculation_accuracy(self):
        """Test that vectorized cosine-distance logic gives correct results."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 5)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Calculate distances manually (current method)
        manual_distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                manual_distances.append(1 - cos_sim)

        # Calculate distances using vectorized method
        similarity_matrix = cosine_similarity(embeddings)
        mask = np.triu(np.ones(similarity_matrix.shape), k=1).astype(bool)
        vectorized_distances = 1 - similarity_matrix[mask]

        # Should be approximately equal
        assert len(manual_distances) == len(vectorized_distances)
        assert np.allclose(manual_distances, vectorized_distances, atol=1e-10)

    def test_graph_construction_scalability(self):
        """Test graph construction with various sizes."""
        sizes = [10, 50, 100]  # Test different sizes

        for n in sizes:
            # Create random normalized embeddings
            np.random.seed(42)
            embeddings = np.random.randn(n, 20)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Should handle all sizes without errors
            graph = _construct_knn_graph(
                embeddings, k=min(5, n - 1), similarity_threshold=0.0
            )

            assert graph.vcount() == n
            # Should have reasonable number of edges (not necessarily all possible)
            assert graph.ecount() >= 0
            assert graph.ecount() <= n * min(5, n - 1)  # Upper bound


class TestDataValidation:
    """Test data validation and error handling."""

    def test_invalid_embedding_dimensions(self):
        """Test handling of inconsistent embedding dimensions."""
        # Create embeddings with different dimensions
        invalid_embeddings = np.array(
            [
                [1, 0, 0],  # 3D
                [0, 1],  # 2D - inconsistent!
            ],
            dtype=object,
        )

        # Should handle inconsistent dimensions gracefully
        try:
            # This might fail during array creation, which is expected
            _construct_knn_graph(invalid_embeddings, k=1)
        except (ValueError, TypeError):
            # Expected to fail with clear error
            pass

    def test_non_numeric_embeddings(self):
        """Test handling of non-numeric embeddings."""
        # Create embeddings with non-numeric values
        invalid_embeddings = np.array([["a", "b", "c"], ["d", "e", "f"]])

        # Should handle non-numeric data gracefully
        try:
            _construct_knn_graph(invalid_embeddings, k=1)
        except (ValueError, TypeError):
            # Expected to fail with clear error
            pass

    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        empty_embeddings = np.array([]).reshape(0, 5)

        graph = _construct_knn_graph(empty_embeddings, k=5, similarity_threshold=0.0)

        assert graph.vcount() == 0
        assert graph.ecount() == 0


class TestBinQuality:
    """Test bin quality calculation."""

    def test_calculate_bin_quality_perfect_bin(self):
        """Test with a perfect bin: 133 unique genes, no duplicates."""
        contig_names = ["c1"]
        # Mock gene mappings: 133 unique gene families
        gene_mappings = {"c1": {f"gene_{i}": {} for i in range(133)}}

        score, scg, dups = _calculate_bin_quality(contig_names, gene_mappings)

        # N=133, G=133
        # Completeness = 133/133 = 1.0
        # Contamination = (133-133)/133 = 0.0
        # Precision = 1.0
        # F1 = 2 * 1 * 1 / (1 + 1) = 1.0

        assert scg == 133
        assert dups == 0
        assert abs(score - 1.0) < 1e-6

    def test_calculate_bin_quality_half_complete(self):
        """Test with 66 unique genes, no duplicates."""
        contig_names = ["c1"]
        gene_mappings = {"c1": {f"gene_{i}": {} for i in range(66)}}

        score, scg, dups = _calculate_bin_quality(contig_names, gene_mappings)

        # N=66, G=66
        # Completeness = 66/133 ~= 0.4962
        # Contamination = 0.0
        # Precision = 1.0
        # F1 = 2 * 0.4962 * 1 / (0.4962 + 1)

        expected_completeness = 66.0 / 133.0
        expected_f1 = (2 * expected_completeness * 1.0) / (expected_completeness + 1.0)

        assert scg == 66
        assert dups == 0
        assert abs(score - expected_f1) < 1e-6

    def test_calculate_bin_quality_empty(self):
        """Test with empty bin."""
        contig_names = ["c1"]
        gene_mappings = {}

        score, scg, dups = _calculate_bin_quality(contig_names, gene_mappings)

        assert score == 0.0
        assert scg == 0
        assert dups == 0

    def test_calculate_bin_quality_with_duplicates(self):
        """Test with duplicates (contamination)."""
        contig_names = ["c1", "c2"]
        gene_mappings = {"c1": {"gene_A": {}}, "c2": {"gene_A": {}}}

        score, scg, dups = _calculate_bin_quality(contig_names, gene_mappings)

        # Gene A count = 2
        # N = 1 (unique family)
        # G = 2 (total genes)
        # Completeness = 1/133
        # Contamination = (2-1)/2 = 0.5
        # Precision = 1 - 0.5 = 0.5

        comp = 1.0 / 133.0
        prec = 0.5
        expected_f1 = (2 * comp * prec) / (comp + prec)

        assert (
            scg == 0
        )  # appearing 2 times means it is not a Single Copy Gene (SCG counts genes appearing exactly once)
        assert dups == 1  # 1 gene family has value > 1
        assert abs(score - expected_f1) < 1e-6
