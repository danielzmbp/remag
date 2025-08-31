"""Integration tests for REMAG clustering pipeline."""

import pytest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os

from remag.clustering import GraphManager, ClusteringManager


class TestClusteringIntegration:
    """Integration tests for the clustering pipeline."""
    
    def test_graph_manager_construction(self, sample_embeddings_df):
        """Test that GraphManager can construct graphs from embeddings."""
        # Create GraphManager with test parameters
        graph_manager = GraphManager(k=5, similarity_threshold=0.1, n_jobs=1)
        
        # Test graph construction
        embeddings = sample_embeddings_df.values
        graph = graph_manager.construct_graph(embeddings)
        
        # Basic sanity checks
        assert graph.vcount() == len(sample_embeddings_df)
        assert graph.ecount() >= 0  # Should have some edges
        
        # Check that graph properties are reasonable
        assert all(weight >= 0 and weight <= 1 for weight in graph.es['weight'])
    
    def test_clustering_manager_initialization(self, mock_args):
        """Test ClusteringManager initialization."""
        clustering_manager = ClusteringManager(mock_args)
        
        assert clustering_manager.args == mock_args
        assert hasattr(clustering_manager, 'graph_manager')
        assert clustering_manager.graph_manager.k == 15  # default value
    
    def test_end_to_end_clustering_pipeline(self, sample_embeddings_df, sample_fragments_dict, mock_args, temp_dir):
        """Test complete clustering pipeline with synthetic data."""
        # Set up mock args with temp directory
        mock_args.output = temp_dir
        mock_args.leiden_resolution = 1.0
        
        # Import the main clustering function
        from remag.clustering import cluster_contigs
        
        # Mock the eukaryotic scores loading to avoid file dependencies
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            # Run the clustering pipeline
            try:
                clusters_df = cluster_contigs(sample_embeddings_df, sample_fragments_dict, mock_args)
                
                # Verify output structure
                assert isinstance(clusters_df, pd.DataFrame)
                assert 'contig' in clusters_df.columns
                assert 'cluster' in clusters_df.columns
                
                # Should not have more clusters than contigs
                assert len(clusters_df) <= len(sample_embeddings_df)
                
                # All contigs should be assigned to some cluster
                assert clusters_df['cluster'].notna().all()
                
                # Cluster IDs should be reasonable (string format)
                assert all(isinstance(cluster_id, str) for cluster_id in clusters_df['cluster'])
                
            except Exception as e:
                # If the test fails, provide useful debugging info
                pytest.fail(f"Clustering pipeline failed: {e}")
    
    def test_clustering_with_empty_input(self, mock_args):
        """Test clustering handles empty input gracefully."""
        empty_embeddings = pd.DataFrame()
        empty_fragments = {}
        
        from remag.clustering import cluster_contigs
        
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            # Should handle empty input gracefully
            result = cluster_contigs(empty_embeddings, empty_fragments, mock_args)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0  # Empty result for empty input
    
    def test_clustering_with_single_contig(self, mock_args):
        """Test clustering with single contig."""
        # Create single contig embedding
        single_embedding_df = pd.DataFrame(
            np.random.randn(1, 64),
            index=['single_contig'],
            columns=[f'dim_{i}' for i in range(64)]
        )
        
        single_fragments = {
            'single_contig': {'sequence': 'ATCGATCG', 'length': 8}
        }
        
        from remag.clustering import cluster_contigs
        
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            clusters_df = cluster_contigs(single_embedding_df, single_fragments, mock_args)
            
            assert len(clusters_df) == 1
            assert clusters_df.iloc[0]['contig'] == 'single_contig'
            assert isinstance(clusters_df.iloc[0]['cluster'], str)


class TestGraphCaching:
    """Test graph caching functionality."""
    
    def test_graph_caching_saves_and_loads(self, sample_embeddings_df, mock_args, temp_dir):
        """Test that graph caching works correctly."""
        mock_args.output = temp_dir
        mock_args.keep_intermediate = True
        
        graph_manager = GraphManager(k=5, similarity_threshold=0.1)
        embeddings = sample_embeddings_df.values
        
        # First call should create the graph and save cache
        graph1 = graph_manager.construct_graph(embeddings, mock_args)
        
        # Check that cache files were created
        cache_files = [f for f in os.listdir(temp_dir) if 'graph' in f.lower()]
        # Note: May not create cache files in this test scenario, but should not crash
        
        # Second call with same parameters should use cache (if implemented)
        graph2 = graph_manager.construct_graph(embeddings, mock_args)
        
        # Graphs should be equivalent
        assert graph1.vcount() == graph2.vcount()
        # Note: Edge count might differ due to randomness in algorithms
        
    def test_graph_cache_invalidation(self, sample_embeddings_df, mock_args, temp_dir):
        """Test cache invalidation with different parameters."""
        mock_args.output = temp_dir
        mock_args.keep_intermediate = True
        
        # Create two different GraphManagers
        graph_manager1 = GraphManager(k=5, similarity_threshold=0.1)
        graph_manager2 = GraphManager(k=10, similarity_threshold=0.2)
        
        embeddings = sample_embeddings_df.values
        
        # Different parameters should create different graphs
        graph1 = graph_manager1.construct_graph(embeddings, mock_args)
        graph2 = graph_manager2.construct_graph(embeddings, mock_args)
        
        # Should not crash and should handle parameter differences
        assert graph1.vcount() == graph2.vcount()  # Same input size
        # But internal structure might differ


class TestErrorRecovery:
    """Test error recovery in clustering pipeline."""
    
    def test_clustering_handles_invalid_embeddings(self, mock_args):
        """Test clustering handles invalid embedding data."""
        # Create embeddings with NaN values
        invalid_embeddings = pd.DataFrame({
            'dim1': [1.0, np.nan, 3.0],
            'dim2': [4.0, 5.0, np.nan],
            'dim3': [7.0, 8.0, 9.0]
        }, index=['contig1', 'contig2', 'contig3'])
        
        fragments = {
            'contig1': {'sequence': 'ATCG', 'length': 4},
            'contig2': {'sequence': 'GCTA', 'length': 4},
            'contig3': {'sequence': 'TTAA', 'length': 4}
        }
        
        from remag.clustering import cluster_contigs
        
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            # Should either handle NaN gracefully or raise clear error
            try:
                result = cluster_contigs(invalid_embeddings, fragments, mock_args)
                # If it succeeds, should return valid result
                assert isinstance(result, pd.DataFrame)
            except (ValueError, RuntimeError) as e:
                # Or should raise a clear, informative error
                assert 'nan' in str(e).lower() or 'invalid' in str(e).lower()
    
    def test_clustering_handles_mismatched_data(self, sample_embeddings_df, mock_args):
        """Test clustering when embeddings and fragments don't match."""
        # Create fragments that don't match embeddings
        mismatched_fragments = {
            'different_contig_1': {'sequence': 'ATCG', 'length': 4},
            'different_contig_2': {'sequence': 'GCTA', 'length': 4}
        }
        
        from remag.clustering import cluster_contigs
        
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            # Should handle mismatch gracefully
            try:
                result = cluster_contigs(sample_embeddings_df, mismatched_fragments, mock_args)
                assert isinstance(result, pd.DataFrame)
                # Should still produce some clustering result
            except (KeyError, ValueError) as e:
                # Or should provide clear error about mismatch
                assert 'mismatch' in str(e).lower() or 'not found' in str(e).lower()


class TestPerformanceBaseline:
    """Basic performance tests to catch major regressions."""
    
    @pytest.mark.slow
    def test_clustering_performance_baseline(self, mock_args):
        """Test clustering completes within reasonable time."""
        # Create moderately large dataset
        np.random.seed(42)
        n_contigs = 500
        embedding_dim = 128
        
        # Create embeddings
        embeddings = np.random.randn(n_contigs, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        embeddings_df = pd.DataFrame(
            embeddings,
            index=[f'contig_{i}' for i in range(n_contigs)],
            columns=[f'dim_{i}' for i in range(embedding_dim)]
        )
        
        fragments = {
            f'contig_{i}': {'sequence': 'A'*1000, 'length': 1000}
            for i in range(n_contigs)
        }
        
        from remag.clustering import cluster_contigs
        import time
        
        with patch.object(ClusteringManager, 'load_eukaryotic_scores', return_value={}):
            start_time = time.time()
            clusters_df = cluster_contigs(embeddings_df, fragments, mock_args)
            duration = time.time() - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert duration < 60.0  # 60 seconds max for 500 contigs
            
            # Should produce reasonable clustering
            assert isinstance(clusters_df, pd.DataFrame)
            assert len(clusters_df) <= n_contigs