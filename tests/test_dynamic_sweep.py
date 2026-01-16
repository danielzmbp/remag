
import pytest
from unittest.mock import patch, MagicMock
from remag.clustering import cluster_contigs
import pandas as pd
import numpy as np

def test_dynamic_resolution_sweep_generation():
    """Test that low starting resolution triggers dynamic sweep generation."""
    
    # Mock data
    embeddings_df = pd.DataFrame(
        np.random.rand(10, 5),
        index=[f"contig_{i}" for i in range(10)]
    )
    fragments_dict = {}
    
    args = MagicMock()
    args.output = "dummy_output"
    args.leiden_resolution = 0.001  # Very low start
    args.skip_chimera_detection = True
    
    # Mock clustering manager
    with patch('remag.clustering.ClusteringManager') as MockManager:
        mock_graph = MagicMock()
        mock_graph.ecount.return_value = 100
        mock_graph.vcount.return_value = 10
        mock_graph.connected_components.return_value = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] # One component
        
        MockManager.return_value.graph_manager.construct_graph.return_value = mock_graph
        MockManager.return_value.graph_manager.k = 15
        MockManager.return_value.graph_manager.similarity_threshold = 0.1

        # Mock leiden clustering to:
        # 1. Return 1 cluster initially (trigger reclustering)
        # 2. Return 1 cluster for first few attempts
        # 3. Return 2 clusters eventually
        
        # Initial call + sweep calls
        # We expect calls with: 0.001 (initial), then 0.002, 0.004, 0.008...
        
        side_effects = []
        # Initial call
        side_effects.append(np.zeros(10, dtype=int)) 
        
        # Dynamic sweep calls (doubling)
        # 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128
        for _ in range(7): 
            side_effects.append(np.zeros(10, dtype=int))
            
        # Success at 0.15 (first base sweep value)
        success_labels = np.zeros(10, dtype=int)
        success_labels[5:] = 1
        side_effects.append(success_labels)
        
        with patch('remag.clustering._leiden_clustering_on_graph', side_effect=side_effects) as mock_leiden:
            with patch('os.makedirs'):
                with patch('pandas.DataFrame.to_csv'):
                     cluster_contigs(embeddings_df, fragments_dict, args)
            
            # Verify calls
            # 1 initial + 7 doubling + 1 success = 9 calls
            assert mock_leiden.call_count == 9
            
            # Verify resolutions called
            calls = mock_leiden.call_args_list
            resolutions = [c[1]['resolution'] for c in calls]
            
            # Check the sequence
            expected_start = [0.001, 0.002, 0.004, 0.008]
            for i, val in enumerate(expected_start):
                 assert abs(resolutions[i] - val) < 1e-6

