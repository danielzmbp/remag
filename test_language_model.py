#!/usr/bin/env python3
"""
Test script for language model integration in REMAG
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add remag to path
sys.path.insert(0, os.path.dirname(__file__))

def create_test_fasta():
    """Create a small test FASTA file."""
    test_fasta = """
>contig_1
ATGCGATCGTAGCTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGC
GATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAG
CGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTA
>contig_2 
GCTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGA
TCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCG
ATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGC
>contig_3
CGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTA
GCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGT
AGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCG
""".strip()
    
    return test_fasta

def test_embedding_generator():
    """Test the embedding generator with the language model."""
    print("Testing DNA embedding generator...")
    
    # Check if model exists
    model_path = "model"
    if not os.path.exists(model_path):
        print(f"Model directory not found at {model_path}")
        return False
    
    try:
        from remag.embedding_generator import DNAEmbeddingGenerator, calculate_dna_embeddings
        
        # Create test sequences
        test_sequences = [
            ("seq1", "ATGCGATCGTAGCTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGC"),
            ("seq2", "GCTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGA"),
            ("seq3", "CGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTAGCGATCGTA")
        ]
        
        print("Initializing embedding generator...")
        generator = DNAEmbeddingGenerator(model_path=model_path, device="auto")
        
        print("Generating embeddings...")
        embeddings_df = generator.generate_embeddings(test_sequences, batch_size=2)
        
        print(f"Generated embeddings shape: {embeddings_df.shape}")
        print(f"Expected 768 dimensions: {embeddings_df.shape[1] == 768}")
        
        # Test convenience function
        print("Testing convenience function...")
        embeddings_df2 = calculate_dna_embeddings(
            test_sequences, 
            model_path=model_path,
            device="auto",
            batch_size=2
        )
        
        print(f"Convenience function embeddings shape: {embeddings_df2.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing embedding generator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test the updated feature extraction with language model."""
    print("\nTesting feature extraction with language model...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test FASTA
        fasta_file = os.path.join(temp_dir, "test.fasta")
        with open(fasta_file, 'w') as f:
            f.write(create_test_fasta())
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            from remag.features import get_features
            
            print("Testing language model features...")
            features_df, fragments_dict = get_features(
                fasta_file=fasta_file,
                bam_files=None,
                tsv_files=None,
                output_dir=output_dir,
                min_contig_length=100,  # Low threshold for test
                cores=1,
                num_augmentations=2,
                use_language_model=True,
                model_path="model"
            )
            
            print(f"Features shape: {features_df.shape}")
            print(f"Features have 768 dimensions: {features_df.shape[1] == 768}")
            print(f"Number of fragments: {len(fragments_dict)}")
            
            # Test traditional k-mer features for comparison
            print("\nTesting k-mer features for comparison...")
            features_df_kmer, fragments_dict_kmer = get_features(
                fasta_file=fasta_file,
                bam_files=None,
                tsv_files=None,
                output_dir=output_dir,
                min_contig_length=100,
                cores=1,
                num_augmentations=2,
                use_language_model=False,
                model_path=None
            )
            
            print(f"K-mer features shape: {features_df_kmer.shape}")
            print(f"K-mer features have 136 dimensions: {features_df_kmer.shape[1] == 136}")
            
            return True
            
        except Exception as e:
            print(f"Error testing feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_neural_network():
    """Test that the neural network can handle language model features."""
    print("\nTesting neural network with language model features...")
    
    try:
        import torch
        import pandas as pd
        import numpy as np
        from remag.models import SiameseNetwork
        
        # Create fake language model features (768 dims) + coverage (2 dims)
        n_samples = 50
        n_lm_features = 768
        n_coverage_features = 2
        
        fake_features = np.random.randn(n_samples, n_lm_features + n_coverage_features)
        features_df = pd.DataFrame(fake_features)
        features_df.index = [f"contig_{i}.original" for i in range(n_samples)]
        
        print(f"Created fake features: {features_df.shape}")
        
        # Test model creation
        model = SiameseNetwork(
            n_lm_features=n_lm_features,
            n_coverage_features=n_coverage_features,
            embedding_dim=128
        )
        
        print("Model created successfully")
        
        # Test forward pass
        x = torch.randn(4, n_lm_features + n_coverage_features)
        embedding = model.get_embedding(x)
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Expected embedding dim 128: {embedding.shape[1] == 128}")
        
        return True
        
    except Exception as e:
        print(f"Error testing neural network: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing REMAG language model integration")
    print("=" * 50)
    
    tests = [
        ("Embedding Generator", test_embedding_generator),
        ("Feature Extraction", test_feature_extraction),
        ("Neural Network", test_neural_network),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)