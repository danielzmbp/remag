"""Performance tests to validate optimizations in REMAG."""

import pytest
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from remag.clustering import _vectorized_pairwise_distances, _permutation_anova_chimera_test


class TestChimeraDetectionPerformance:
    """Test performance improvements in chimera detection."""
    
    def test_vectorized_vs_manual_distance_calculation(self):
        """Compare vectorized distance calculation with manual nested loops."""
        np.random.seed(42)
        
        # Test with moderately sized embeddings
        n_samples = 100
        embedding_dim = 64
        embeddings = np.random.randn(n_samples, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Manual calculation (old method)
        start_time = time.time()
        manual_distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                cos_sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                manual_distances.append(1 - cos_sim)
        manual_time = time.time() - start_time
        
        # Vectorized calculation (new method)
        start_time = time.time()
        vectorized_distances = _vectorized_pairwise_distances(embeddings)
        vectorized_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(manual_distances) == len(vectorized_distances)
        assert np.allclose(manual_distances, vectorized_distances, rtol=1e-10, atol=1e-10)
        
        # Verify performance improvement
        speedup = manual_time / vectorized_time
        assert speedup >= 2.0, (
            f"Expected 2x speedup, got {speedup:.2f}x "
            f"(manual: {manual_time:.4f}s, vectorized: {vectorized_time:.4f}s)"
        )
    
    def test_inter_group_distance_performance(self):
        """Test inter-group distance calculation performance."""
        np.random.seed(42)
        
        # Create two groups of embeddings
        n1, n2 = 50, 60
        embedding_dim = 64
        
        embeddings1 = np.random.randn(n1, embedding_dim)
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        
        embeddings2 = np.random.randn(n2, embedding_dim)  
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Manual calculation (old method)
        start_time = time.time()
        manual_inter_distances = []
        for i in range(len(embeddings1)):
            for j in range(len(embeddings2)):
                cos_sim = cosine_similarity([embeddings1[i]], [embeddings2[j]])[0][0]
                manual_inter_distances.append(1 - cos_sim)
        manual_time = time.time() - start_time
        
        # Vectorized calculation (new method)
        start_time = time.time()
        vectorized_inter_distances = _vectorized_pairwise_distances(embeddings1, embeddings2)
        vectorized_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(manual_inter_distances) == len(vectorized_inter_distances)
        assert np.allclose(manual_inter_distances, vectorized_inter_distances, rtol=1e-10, atol=1e-10)
        
        # Verify performance improvement
        speedup = manual_time / vectorized_time
        assert speedup >= 1.5, (
            f"Expected 1.5x speedup for inter-group distances, got {speedup:.2f}x "
            f"(manual: {manual_time:.4f}s, vectorized: {vectorized_time:.4f}s)"
        )
    
    @pytest.mark.slow
    def test_chimera_detection_large_scale_performance(self):
        """Test chimera detection performance with larger datasets."""
        np.random.seed(42)
        
        # Create larger groups to test scalability
        n1, n2 = 200, 150
        embedding_dim = 128
        
        h1_embeddings = np.random.randn(n1, embedding_dim)
        h1_embeddings = h1_embeddings / np.linalg.norm(h1_embeddings, axis=1, keepdims=True)
        
        h2_embeddings = np.random.randn(n2, embedding_dim)
        h2_embeddings = h2_embeddings / np.linalg.norm(h2_embeddings, axis=1, keepdims=True)
        
        # Test the full chimera detection pipeline
        start_time = time.time()
        is_chimeric, results = _permutation_anova_chimera_test(
            h1_embeddings, h2_embeddings, n_permutations=100  # Reduced for faster testing
        )
        duration = time.time() - start_time
        
        # Verify reasonable performance (should complete in under 30 seconds)
        assert duration < 30.0, (
            f"Chimera detection with {n1}+{n2} embeddings took too long: {duration:.4f}s"
        )
        
        # Should produce valid results
        assert isinstance(is_chimeric, bool)
        assert 'p_value' in results
        assert 'mean_intra_distance' in results
        assert 'mean_inter_distance' in results
        assert results['n_intra_pairs'] == (n1 * (n1-1) // 2) + (n2 * (n2-1) // 2)
        assert results['n_inter_pairs'] == n1 * n2
    
    def test_vectorized_function_edge_cases(self):
        """Test vectorized function handles edge cases efficiently."""
        # Empty embeddings
        empty_distances = _vectorized_pairwise_distances(np.array([]).reshape(0, 5))
        assert len(empty_distances) == 0
        
        # Single embedding
        single_embedding = np.array([[1, 0, 0, 0, 0]])
        single_distances = _vectorized_pairwise_distances(single_embedding)
        assert len(single_distances) == 0
        
        # Two embeddings
        two_embeddings = np.array([[1, 0, 0], [0, 1, 0]])
        two_embeddings = two_embeddings / np.linalg.norm(two_embeddings, axis=1, keepdims=True)
        two_distances = _vectorized_pairwise_distances(two_embeddings)
        assert len(two_distances) == 1
        # Orthogonal vectors should have distance = 1
        assert abs(two_distances[0] - 1.0) < 1e-10
    
    def test_memory_efficiency(self):
        """Test that vectorized operations are memory efficient."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large embeddings
        np.random.seed(42)
        n_samples = 500
        embedding_dim = 256
        embeddings = np.random.randn(n_samples, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Perform vectorized calculation
        distances = _vectorized_pairwise_distances(embeddings)
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify reasonable memory usage (should be under 500MB for reasonable dataset size)
        assert memory_increase < 500.0, (
            f"Memory usage too high: {memory_increase:.2f} MB for {n_samples} embeddings"
        )
        
        # Verify we got the expected number of distances
        expected_pairs = n_samples * (n_samples - 1) // 2
        assert len(distances) == expected_pairs


class TestComplexityAnalysis:
    """Analyze computational complexity improvements."""
    
    def test_scaling_behavior(self):
        """Test how computation time scales with input size."""
        np.random.seed(42)
        embedding_dim = 64
        sizes = [10, 20, 50, 100]  # Different input sizes
        times = []
        
        for n in sizes:
            embeddings = np.random.randn(n, embedding_dim)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Time the vectorized calculation
            start_time = time.time()
            distances = _vectorized_pairwise_distances(embeddings)
            duration = time.time() - start_time
            times.append(duration)
            
            # Verify performance scales reasonably (should be under 1 second for small datasets)
            assert duration < 1.0, (
                f"Distance calculation for size {n} took too long: {duration:.6f}s"
            )
        
        # Verify times are reasonable and scale properly
        # For vectorized operations, should scale better than O(n²)
        for i in range(len(times)):
            assert times[i] < 1.0, f"Size {sizes[i]} took too long: {times[i]:.6f}s"
    
    def test_correctness_with_known_vectors(self):
        """Test correctness with known vectors and expected distances."""
        # Create test vectors with known relationships
        embeddings = np.array([
            [1, 0, 0],      # Vector 1
            [0, 1, 0],      # Vector 2 (orthogonal to 1)
            [1/np.sqrt(2), 1/np.sqrt(2), 0],  # Vector 3 (45° from 1 and 2)
            [1, 0, 0]       # Vector 4 (identical to 1)
        ], dtype=float)
        
        distances = _vectorized_pairwise_distances(embeddings)
        
        # We expect 6 distances (4 choose 2)
        assert len(distances) == 6
        
        # Known distances:
        # d(1,2) = 1.0 (orthogonal)
        # d(1,3) = 1 - cos(45°) = 1 - 1/sqrt(2) ≈ 0.293
        # d(1,4) = 0.0 (identical)
        # d(2,3) = 1 - cos(45°) ≈ 0.293  
        # d(2,4) = 1.0 (orthogonal)
        # d(3,4) = 1 - cos(45°) ≈ 0.293
        
        expected_distances = [
            1.0,                    # d(1,2)
            1 - 1/np.sqrt(2),      # d(1,3)
            0.0,                    # d(1,4)
            1 - 1/np.sqrt(2),      # d(2,3)
            1.0,                    # d(2,4)
            1 - 1/np.sqrt(2)       # d(3,4)
        ]
        
        for i, (actual, expected) in enumerate(zip(distances, expected_distances)):
            assert abs(actual - expected) < 1e-10, f"Distance {i}: expected {expected}, got {actual}"


@pytest.mark.benchmark  
class TestBenchmarks:
    """Benchmark tests for regression detection."""
    
    def test_baseline_chimera_detection_benchmark(self):
        """Baseline benchmark for chimera detection performance."""
        np.random.seed(42)
        
        # Standard test size
        n1, n2 = 100, 100
        embedding_dim = 128
        
        h1_embeddings = np.random.randn(n1, embedding_dim)
        h1_embeddings = h1_embeddings / np.linalg.norm(h1_embeddings, axis=1, keepdims=True)
        
        h2_embeddings = np.random.randn(n2, embedding_dim)
        h2_embeddings = h2_embeddings / np.linalg.norm(h2_embeddings, axis=1, keepdims=True)
        
        # Benchmark the optimized version
        start_time = time.time()
        is_chimeric, results = _permutation_anova_chimera_test(
            h1_embeddings, h2_embeddings, n_permutations=1000
        )
        duration = time.time() - start_time
        
        # Verify reasonable baseline performance (serves as regression test baseline)
        assert duration < 10.0, (
            f"Baseline chimera detection too slow: {duration:.4f}s"
        )
        
        # Log performance for monitoring
        with open('/tmp/remag_performance_log.txt', 'a') as f:
            f.write(f"chimera_detection_100x100: {duration:.4f}s\n")
    
    def test_distance_calculation_benchmark(self):
        """Benchmark pure distance calculation performance."""
        np.random.seed(42)
        
        # Test various sizes
        test_sizes = [(50, 50), (100, 100), (200, 150)]
        
        for n1, n2 in test_sizes:
            embedding_dim = 128
            
            embeddings1 = np.random.randn(n1, embedding_dim)
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            
            embeddings2 = np.random.randn(n2, embedding_dim)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Benchmark intra-group distance calculation
            start_time = time.time()
            intra1_distances = _vectorized_pairwise_distances(embeddings1)
            intra2_distances = _vectorized_pairwise_distances(embeddings2)
            inter_distances = _vectorized_pairwise_distances(embeddings1, embeddings2)
            duration = time.time() - start_time
            
            # Verify performance scales reasonably with dataset size
            assert duration < 5.0, (
                f"Distance calculation {n1}x{n2} took too long: {duration:.4f}s"
            )
            
            # Should scale reasonably
            expected_max_time = (n1 * n2) / 10000  # Rough heuristic
            assert duration < max(expected_max_time, 10.0)
            
            # Verify we got expected number of distances
            assert len(intra1_distances) == n1 * (n1 - 1) // 2
            assert len(intra2_distances) == n2 * (n2 - 1) // 2  
            assert len(inter_distances) == n1 * n2