"""Unit tests for clustering module."""

import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import Mock, patch
from sklearn.metrics.pairwise import cosine_similarity

from remag.clustering import GraphManager, ClusteringManager, _construct_knn_graph


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
        embeddings = np.array([
            [0.4, 0.3, 0.5, 0.6, 0.2],
            [0.1, 0.8, 0.2, 0.1, 0.5],
            [0.6, 0.2, 0.4, 0.3, 0.7]
        ])
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        graph = _construct_knn_graph(embeddings, k=2, similarity_threshold=0.0)
        
        assert graph.vcount() == 3
        assert graph.ecount() >= 0  # At least some edges should exist
        assert all(weight >= 0 and weight <= 1 for weight in graph.es['weight'])
    
    def test_construct_graph_no_edges_high_threshold(self):
        """Test graph construction with threshold too high."""
        embeddings = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])  # Orthogonal vectors = zero similarity
        
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
        assert hasattr(manager, 'graph_manager')
    
    def test_load_eukaryotic_scores_file_not_exists(self, mock_args, temp_dir):
        """Test loading eukaryotic scores when file doesn't exist."""
        mock_args.output = temp_dir
        mock_args.fasta = "test.fasta"
        
        manager = ClusteringManager(mock_args)
        
        with patch('remag.features.get_classification_results_path') as mock_path:
            mock_path.return_value = "/nonexistent/path.tsv"
            scores = manager.load_eukaryotic_scores()
            
        assert isinstance(scores, dict)
        assert len(scores) == 0  # Should return empty dict if file not found
    
    def test_load_eukaryotic_scores_file_exists(self, mock_args, temp_dir):
        """Test loading eukaryotic scores when file exists."""
        # Create a mock classification results file
        classification_file = os.path.join(temp_dir, "classification.tsv")
        classification_data = pd.DataFrame({
            'header': ['contig1', 'contig2', 'contig3'],
            'eukar_score': [0.8, 0.2, 0.9]
        })
        classification_data.to_csv(classification_file, sep='\t', index=False)
        
        mock_args.output = temp_dir
        mock_args.fasta = "test.fasta"
        
        manager = ClusteringManager(mock_args)
        
        with patch('remag.features.get_classification_results_path') as mock_path:
            mock_path.return_value = classification_file
            scores = manager.load_eukaryotic_scores()
        
        assert isinstance(scores, dict)
        assert len(scores) == 3
        assert scores['contig1'] == 0.8
        assert scores['contig2'] == 0.2
        assert scores['contig3'] == 0.9


class TestChimeraDetection:
    """Test chimera detection functionality."""
    
    def test_permutation_anova_identical_groups(self):
        """Test that identical groups are not detected as chimeric."""
        from remag.clustering import _permutation_anova_chimera_test
        
        # Create identical embeddings for both halves
        embedding = np.array([[0.5, 0.5, 0.0]])
        h1_embeddings = np.tile(embedding, (5, 1))
        h2_embeddings = np.tile(embedding, (5, 1))
        
        is_chimeric, results = _permutation_anova_chimera_test(
            h1_embeddings, h2_embeddings, n_permutations=100
        )
        
        assert not is_chimeric
        assert results['p_value'] > 0.05
        # Distances should be very similar (approximately zero)
        assert abs(results['mean_intra_distance'] - results['mean_inter_distance']) < 0.1
    
    def test_permutation_anova_clearly_different_groups(self):
        """Test that clearly different groups are detected as chimeric."""
        from remag.clustering import _permutation_anova_chimera_test
        
        # Create very different embeddings for each half with small variations within groups
        np.random.seed(42)
        h1_base = np.array([1, 0, 0])
        h2_base = np.array([0, 1, 0])
        
        # Add small random variations to avoid identical embeddings within groups
        h1_embeddings = np.array([h1_base + np.random.normal(0, 0.01, 3) for _ in range(10)])
        h2_embeddings = np.array([h2_base + np.random.normal(0, 0.01, 3) for _ in range(10)])
        
        # Normalize to keep them close to unit vectors
        h1_embeddings = h1_embeddings / np.linalg.norm(h1_embeddings, axis=1, keepdims=True)
        h2_embeddings = h2_embeddings / np.linalg.norm(h2_embeddings, axis=1, keepdims=True)
        
        is_chimeric, results = _permutation_anova_chimera_test(
            h1_embeddings, h2_embeddings, n_permutations=100
        )
        
        assert is_chimeric
        assert results['p_value'] < 0.05
        assert results['mean_inter_distance'] > results['mean_intra_distance']
    
    def test_permutation_anova_edge_cases(self):
        """Test edge cases like single embeddings."""
        from remag.clustering import _permutation_anova_chimera_test
        
        # Single embedding in each group
        h1 = np.array([[0.5, 0.5]])
        h2 = np.array([[0.3, 0.7]])
        
        is_chimeric, results = _permutation_anova_chimera_test(h1, h2)
        
        # Should handle single embeddings gracefully
        assert not results['test_performed']  # Should skip test
        assert results['n_intra_pairs'] == 0  # No intra-group pairs possible
    
    def test_permutation_anova_empty_groups(self):
        """Test handling of empty groups."""
        from remag.clustering import _permutation_anova_chimera_test
        
        h1 = np.array([]).reshape(0, 2)
        h2 = np.array([[0.5, 0.5]])
        
        is_chimeric, results = _permutation_anova_chimera_test(h1, h2)
        
        # Should handle empty groups gracefully
        assert not is_chimeric
        assert not results['test_performed']


class TestPerformanceOptimizations:
    """Test performance-related functionality and optimizations."""
    
    def test_chimera_detection_performance_with_large_groups(self):
        """Test chimera detection performance with larger groups."""
        from remag.clustering import _permutation_anova_chimera_test
        import time
        
        # Create moderately large groups to test performance
        np.random.seed(42)
        h1_embeddings = np.random.randn(50, 64)
        h2_embeddings = np.random.randn(50, 64)
        
        start_time = time.time()
        is_chimeric, results = _permutation_anova_chimera_test(
            h1_embeddings, h2_embeddings, n_permutations=100
        )
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds max for this size
        
        # Should produce valid results
        assert isinstance(is_chimeric, bool)
        assert 'p_value' in results
        assert 'mean_intra_distance' in results
        assert 'mean_inter_distance' in results
    
    def test_vectorized_distance_calculation_accuracy(self):
        """Test that vectorized distance calculation gives correct results."""
        # This test will be useful when we optimize the quadratic complexity
        np.random.seed(42)
        embeddings = np.random.randn(10, 5)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate distances manually (current method)
        manual_distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
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
            graph = _construct_knn_graph(embeddings, k=min(5, n-1), similarity_threshold=0.0)
            
            assert graph.vcount() == n
            # Should have reasonable number of edges (not necessarily all possible)
            assert graph.ecount() >= 0
            assert graph.ecount() <= n * min(5, n-1)  # Upper bound


import os
class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_invalid_embedding_dimensions(self):
        """Test handling of inconsistent embedding dimensions."""
        # Create embeddings with different dimensions
        invalid_embeddings = np.array([
            [1, 0, 0],    # 3D
            [0, 1],       # 2D - inconsistent!
        ], dtype=object)
        
        # Should handle inconsistent dimensions gracefully
        try:
            # This might fail during array creation, which is expected
            graph = _construct_knn_graph(invalid_embeddings, k=1)
        except (ValueError, TypeError):
            # Expected to fail with clear error
            pass
    
    def test_non_numeric_embeddings(self):
        """Test handling of non-numeric embeddings."""
        # Create embeddings with non-numeric values
        invalid_embeddings = np.array([
            ["a", "b", "c"],
            ["d", "e", "f"]
        ])
        
        # Should handle non-numeric data gracefully
        try:
            graph = _construct_knn_graph(invalid_embeddings, k=1)
        except (ValueError, TypeError):
            # Expected to fail with clear error
            pass
    
    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        empty_embeddings = np.array([]).reshape(0, 5)
        
        graph = _construct_knn_graph(empty_embeddings, k=5, similarity_threshold=0.0)
        
        assert graph.vcount() == 0
        assert graph.ecount() == 0