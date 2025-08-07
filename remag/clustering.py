"""
Clustering module for REMAG
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from loguru import logger
import os
import json
import igraph as ig
import leidenalg

from .utils import extract_base_contig_name, get_torch_device, group_contigs_by_cluster
import torch

# Try to import cuML for GPU acceleration
try:
    import cuml
    import cudf
    import cupy as cp
    CUML_AVAILABLE = True
    logger.debug("cuML GPU acceleration is available")
except ImportError:
    CUML_AVAILABLE = False
    logger.debug("cuML not available, using CPU-only HDBSCAN")


def _iterative_kmeans_filtering(embeddings, contig_names, eukaryotic_scores, 
                               small_cluster_threshold=0.1, min_eukaryotic_score=0.95, 
                               max_iterations=10):
    """
    Iteratively filter out small clusters with low eukaryotic confidence using k-means.
    
    Args:
        embeddings: Normalized embedding matrix (n_contigs x embedding_dim)
        contig_names: List of contig names corresponding to embeddings
        eukaryotic_scores: Dict mapping contig names to eukaryotic confidence scores
        small_cluster_threshold: Fraction of total data below which a cluster is considered "small"
        min_eukaryotic_score: Minimum eukaryotic score to consider a contig high-confidence
        max_iterations: Maximum number of k-means iterations to perform
        
    Returns:
        tuple: (filtered_embeddings, filtered_contig_names, filter_stats)
    """
    logger.info("Starting iterative k-means filtering to remove small, low-confidence clusters...")
    
    current_embeddings = embeddings.copy()
    current_contig_names = contig_names.copy()
    iteration = 0
    total_removed = 0
    filter_stats = []
    
    while iteration < max_iterations:
        iteration += 1
        n_contigs = len(current_contig_names)
        
        if n_contigs < 10:  # Stop if too few contigs remain
            logger.info(f"Stopping k-means filtering: only {n_contigs} contigs remain")
            break
            
        # Perform k-means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(current_embeddings)
        
        # Analyze cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Identify small cluster
        cluster_0_size = cluster_sizes.get(0, 0)
        cluster_1_size = cluster_sizes.get(1, 0)
        
        small_cluster_label = None
        small_cluster_size = 0
        large_cluster_size = 0
        
        if cluster_0_size < cluster_1_size:
            small_cluster_label = 0
            small_cluster_size = cluster_0_size
            large_cluster_size = cluster_1_size
        else:
            small_cluster_label = 1
            small_cluster_size = cluster_1_size
            large_cluster_size = cluster_0_size
        
        small_cluster_fraction = small_cluster_size / n_contigs
        
        logger.debug(f"Small cluster: label={small_cluster_label}, size={small_cluster_size} "
                    f"({small_cluster_fraction:.3f} of total)")
        
        # Check if small cluster meets removal criteria
        if small_cluster_fraction > small_cluster_threshold:
            logger.info(f"Small cluster is {small_cluster_fraction:.3f} of data "
                       f"(threshold: {small_cluster_threshold}). Stopping k-means filtering.")
            break
        
        # Get contigs in small cluster
        small_cluster_mask = cluster_labels == small_cluster_label
        small_cluster_contigs = [current_contig_names[i] for i in range(n_contigs) if small_cluster_mask[i]]
        
        # Check eukaryotic confidence in small cluster
        high_conf_eukaryotes = 0
        total_scored = 0
        
        for contig in small_cluster_contigs:
            if contig in eukaryotic_scores:
                total_scored += 1
                if eukaryotic_scores[contig] >= min_eukaryotic_score:
                    high_conf_eukaryotes += 1
        
        eukaryotic_fraction = high_conf_eukaryotes / total_scored if total_scored > 0 else 0
        
        logger.info(f"Small cluster analysis: {small_cluster_size} contigs, "
                   f"{high_conf_eukaryotes}/{total_scored} high-confidence eukaryotes "
                   f"(fraction: {eukaryotic_fraction:.3f})")
        
        # Decide whether to remove small cluster
        should_remove = high_conf_eukaryotes == 0 and total_scored > 0
        
        if should_remove:
            logger.info(f"Removing small cluster with {small_cluster_size} contigs "
                       f"(no high-confidence eukaryotes)")
            
            # Keep only large cluster
            large_cluster_mask = cluster_labels != small_cluster_label
            current_embeddings = current_embeddings[large_cluster_mask]
            current_contig_names = [current_contig_names[i] for i in range(n_contigs) if large_cluster_mask[i]]
            
            total_removed += small_cluster_size
            
            # Record filtering stats
            filter_stats.append({
                'iteration': iteration,
                'contigs_before': n_contigs,
                'small_cluster_size': small_cluster_size,
                'small_cluster_fraction': small_cluster_fraction,
                'high_conf_eukaryotes': high_conf_eukaryotes,
                'total_scored': total_scored,
                'eukaryotic_fraction': eukaryotic_fraction,
                'removed': True,
                'contigs_after': len(current_contig_names)
            })
        else:
            if high_conf_eukaryotes > 0:
                reason = f"contains {high_conf_eukaryotes} high-confidence eukaryotes"
            elif total_scored == 0:
                reason = "no eukaryotic scores available"
            else:
                reason = "unknown"
                
            logger.info(f"Keeping small cluster ({reason}). Stopping k-means filtering.")
            
            filter_stats.append({
                'iteration': iteration,
                'contigs_before': n_contigs,
                'small_cluster_size': small_cluster_size,
                'small_cluster_fraction': small_cluster_fraction,
                'high_conf_eukaryotes': high_conf_eukaryotes,
                'total_scored': total_scored,
                'eukaryotic_fraction': eukaryotic_fraction,
                'removed': False,
                'reason': reason,
                'contigs_after': len(current_contig_names)
            })
            break
    
    final_stats = {
        'iterations': iteration,
        'original_contigs': len(contig_names),
        'filtered_contigs': len(current_contig_names),
        'total_removed': total_removed,
        'removal_fraction': total_removed / len(contig_names) if len(contig_names) > 0 else 0,
        'iteration_details': filter_stats
    }
    
    logger.info(f"K-means filtering complete: {len(current_contig_names)}/{len(contig_names)} contigs remaining "
               f"({total_removed} removed in {iteration} iterations)")
    
    return current_embeddings, current_contig_names, final_stats


def _construct_knn_graph(embeddings, k=15, similarity_threshold=0.1):
    """
    Construct a k-NN graph from multidimensional embeddings using cosine similarity.
    
    Args:
        embeddings: Numpy array of L2-normalized embeddings (n_samples x embedding_dim)
        k: Number of nearest neighbors for each node
        similarity_threshold: Minimum cosine similarity to create an edge (0-1)
        
    Returns:
        igraph.Graph: Weighted graph with cosine similarity weights
    """
    logger.info(f"Constructing k-NN graph from {len(embeddings)} embeddings (k={k})")
    
    # Calculate cosine similarity matrix (since embeddings are L2-normalized)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # Create adjacency lists for efficient graph construction
    edges = []
    weights = []
    
    for i in range(len(embeddings)):
        # Get k nearest neighbors (excluding self)
        similarities = similarity_matrix[i]
        
        # Get indices of k+1 highest similarities (including self)
        top_k_indices = np.argpartition(similarities, -(k+1))[-(k+1):]
        top_k_similarities = similarities[top_k_indices]
        
        # Sort by similarity (highest first) and exclude self
        sorted_indices = np.argsort(top_k_similarities)[::-1]
        top_k_indices = top_k_indices[sorted_indices]
        top_k_similarities = top_k_similarities[sorted_indices]
        
        # Remove self-connection and apply similarity threshold
        for j_idx, similarity in zip(top_k_indices, top_k_similarities):
            if j_idx != i and similarity >= similarity_threshold:
                edges.append((i, j_idx))
                weights.append(float(similarity))
    
    logger.info(f"Created {len(edges)} edges with similarity >= {similarity_threshold}")
    
    # Create igraph from edge list
    g = ig.Graph()
    g.add_vertices(len(embeddings))
    g.add_edges(edges)
    g.es['weight'] = weights
    
    # Make graph undirected by averaging edge weights
    g = g.as_undirected(mode='mean')
    
    logger.info(f"Graph created: {g.vcount()} nodes, {g.ecount()} edges")
    
    return g


def _leiden_clustering(embeddings, k=15, similarity_threshold=0.1, resolution=1.0, random_state=42):
    """
    Perform Leiden clustering on embeddings by first constructing a k-NN graph.
    
    Args:
        embeddings: Numpy array of L2-normalized embeddings (n_samples x embedding_dim)
        k: Number of nearest neighbors for graph construction
        similarity_threshold: Minimum cosine similarity to create an edge
        resolution: Resolution parameter for Leiden algorithm (higher = more clusters)
        random_state: Random seed for reproducibility
        
    Returns:
        numpy.array: Cluster labels (-1 for isolated nodes, 0+ for clusters)
    """
    logger.info(f"Starting Leiden clustering with k={k}, resolution={resolution}")
    
    # Construct k-NN graph
    graph = _construct_knn_graph(embeddings, k=k, similarity_threshold=similarity_threshold)
    
    # Check if graph has edges
    if graph.ecount() == 0:
        logger.warning("No edges in graph - all nodes will be noise")
        return np.full(len(embeddings), -1, dtype=int)
    
    # Find connected components
    components = graph.connected_components()
    n_components = len(components)
    logger.info(f"Graph has {n_components} connected components")
    
    if n_components == 1:
        logger.info("Single connected component - running Leiden on full graph")
        # Run Leiden on the entire graph
        partition = leidenalg.find_partition(
            graph, 
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_state
        )
        cluster_labels = np.array(partition.membership)
        
    else:
        # Handle multiple components separately
        logger.info(f"Multiple components detected - running Leiden on each component")
        cluster_labels = np.full(graph.vcount(), -1, dtype=int)
        current_cluster_id = 0
        
        for component in components:
            if len(component) < 2:
                # Single-node component -> noise
                continue
                
            # Extract subgraph for this component
            subgraph = graph.induced_subgraph(component)
            
            # Run Leiden on subgraph
            sub_partition = leidenalg.find_partition(
                subgraph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=random_state
            )
            
            # Map back to original indices
            for i, cluster_id in enumerate(sub_partition.membership):
                original_idx = component[i]
                cluster_labels[original_idx] = current_cluster_id + cluster_id
            
            # Update cluster ID offset for next component
            current_cluster_id += len(set(sub_partition.membership))
    
    # Report clustering results
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    logger.info(f"Leiden clustering complete: {n_clusters} clusters, {n_noise} noise points")
    
    # Log cluster sizes
    if n_clusters > 0:
        cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0])
        logger.info(f"Cluster sizes: {cluster_sizes.tolist()}")
    
    return cluster_labels


def _create_hdbscan_clusterer(min_cluster_size, min_samples, cluster_selection_epsilon=0.3, use_gpu=True):
    """Create HDBSCAN clusterer, preferring GPU if available."""
    if use_gpu and CUML_AVAILABLE:
        try:
            clusterer = cuml.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True
            )
            return clusterer, True  # GPU clusterer
        except Exception as e:
            logger.warning(f"GPU clustering failed, falling back to CPU: {e}")
    
    # CPU fallback
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    return clusterer, False  # CPU clusterer


def _permutation_anova_chimera_test(h1_embeddings, h2_embeddings, n_permutations=1000, alpha=0.05):
    """
    Perform permutation ANOVA to test if inter-group distances are significantly
    larger than intra-group distances, indicating a possible chimeric contig.
    
    Args:
        h1_embeddings: numpy array of embeddings for h1 fragments (n_h1 x embedding_dim)
        h2_embeddings: numpy array of embeddings for h2 fragments (n_h2 x embedding_dim) 
        n_permutations: number of permutations for the test
        alpha: significance level
        
    Returns:
        tuple: (is_chimeric, results_dict)
    """
    # Calculate pairwise cosine distances within and between groups
    
    # Intra-group distances (within h1)
    h1_intra_distances = []
    if len(h1_embeddings) > 1:
        for i in range(len(h1_embeddings)):
            for j in range(i+1, len(h1_embeddings)):
                # Cosine distance = 1 - cosine_similarity
                cos_sim = cosine_similarity([h1_embeddings[i]], [h1_embeddings[j]])[0][0]
                h1_intra_distances.append(1 - cos_sim)
    
    # Intra-group distances (within h2)
    h2_intra_distances = []
    if len(h2_embeddings) > 1:
        for i in range(len(h2_embeddings)):
            for j in range(i+1, len(h2_embeddings)):
                cos_sim = cosine_similarity([h2_embeddings[i]], [h2_embeddings[j]])[0][0]
                h2_intra_distances.append(1 - cos_sim)
    
    # Inter-group distances (between h1 and h2)
    inter_distances = []
    for i in range(len(h1_embeddings)):
        for j in range(len(h2_embeddings)):
            cos_sim = cosine_similarity([h1_embeddings[i]], [h2_embeddings[j]])[0][0]
            inter_distances.append(1 - cos_sim)
    
    # Combine all distances with group labels
    all_distances = h1_intra_distances + h2_intra_distances + inter_distances
    group_labels = (['intra'] * (len(h1_intra_distances) + len(h2_intra_distances)) + 
                   ['inter'] * len(inter_distances))
    
    if not all_distances or len(set(group_labels)) < 2:
        # Not enough data for test
        return False, {
            'f_statistic': 0.0,
            'p_value': 1.0,
            'mean_intra_distance': 0.0,
            'mean_inter_distance': 0.0,
            'n_intra_pairs': len(h1_intra_distances) + len(h2_intra_distances),
            'n_inter_pairs': len(inter_distances),
            'test_performed': False
        }
    
    # Calculate observed F-statistic
    def calculate_f_statistic(distances, labels):
        intra_distances = [d for d, l in zip(distances, labels) if l == 'intra']
        inter_distances = [d for d, l in zip(distances, labels) if l == 'inter']
        
        if not intra_distances or not inter_distances:
            return 0.0
            
        mean_intra = np.mean(intra_distances)
        mean_inter = np.mean(inter_distances)
        mean_total = np.mean(distances)
        
        # Between-group sum of squares
        ss_between = (len(intra_distances) * (mean_intra - mean_total)**2 + 
                     len(inter_distances) * (mean_inter - mean_total)**2)
        
        # Within-group sum of squares
        ss_within = (sum((d - mean_intra)**2 for d in intra_distances) + 
                    sum((d - mean_inter)**2 for d in inter_distances))
        
        # Degrees of freedom
        df_between = 1  # 2 groups - 1
        df_within = len(distances) - 2
        
        if df_within <= 0 or ss_within == 0:
            return 0.0
            
        # F-statistic
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        return ms_between / ms_within if ms_within > 0 else 0.0
    
    observed_f = calculate_f_statistic(all_distances, group_labels)
    
    # Permutation test
    extreme_count = 0
    all_indices = list(range(len(all_distances)))
    
    for _ in range(n_permutations):
        # Randomly shuffle group labels
        shuffled_labels = np.random.permutation(group_labels)
        permuted_f = calculate_f_statistic(all_distances, shuffled_labels)
        
        if permuted_f >= observed_f:
            extreme_count += 1
    
    p_value = extreme_count / n_permutations
    is_chimeric = p_value < alpha
    
    # Calculate summary statistics
    intra_distances_all = [d for d, l in zip(all_distances, group_labels) if l == 'intra']
    inter_distances_all = [d for d, l in zip(all_distances, group_labels) if l == 'inter']
    
    results = {
        'f_statistic': float(observed_f),
        'p_value': float(p_value),
        'mean_intra_distance': float(np.mean(intra_distances_all)) if intra_distances_all else 0.0,
        'mean_inter_distance': float(np.mean(inter_distances_all)) if inter_distances_all else 0.0,
        'n_intra_pairs': len(intra_distances_all),
        'n_inter_pairs': len(inter_distances_all),
        'test_performed': True,
        'alpha': alpha,
        'n_permutations': n_permutations
    }
    
    return is_chimeric, results


def detect_chimeric_contigs(embeddings_df, clusters_df, args):
    """
    Detect chimeric contigs by analyzing clustering patterns and embedding similarity of contig halves.
    
    For large contigs (>50kb) that were split into halves during feature generation,
    this function checks if the two halves have divergent embeddings and cluster assignments,
    which could indicate a chimeric contig containing sequences from different organisms.
    
    Args:
        embeddings_df: DataFrame with embeddings for all fragments
        clusters_df: DataFrame with cluster assignments for contigs
        args: Command line arguments
    
    Returns:
        dict: Mapping of contig names to chimera detection results
    """
    logger.info("Starting chimera detection for large contigs...")
    
    # Find contigs that have both h1 and h2 fragments (large contigs that were split)
    split_contigs = {}
    chimera_results = {}
    
    # Load features data to find h1/h2 fragments for large contigs
    from .features import get_features_csv_path
    features_csv_path = get_features_csv_path(args.output)
    
    features_df = None
    if os.path.exists(features_csv_path):
        try:
            features_df = pd.read_csv(features_csv_path, index_col=0)
        except Exception as e:
            logger.error(f"Error loading features data from csv: {e}")
            return {}
    else:
        logger.warning(f"Features file not found at {features_csv_path}, skipping chimera detection")
        return {}
    
    # Group h1/h2 fragments by base contig name
    for fragment_name in features_df.index:
        if '.h1.' in fragment_name or '.h2.' in fragment_name:
            # Extract base contig name (everything before .h1. or .h2.)
            if '.h1.' in fragment_name:
                base_contig = fragment_name.split('.h1.')[0]
                half_id = 'h1'
            else:
                base_contig = fragment_name.split('.h2.')[0]
                half_id = 'h2'
            
            # Only process if this is a large contig with .original embedding
            original_fragment = f"{base_contig}.original"
            if original_fragment in embeddings_df.index:
                if base_contig not in split_contigs:
                    split_contigs[base_contig] = {'h1': [], 'h2': []}
                split_contigs[base_contig][half_id].append(fragment_name)
    
    logger.info(f"Found {len(split_contigs)} large contigs split into halves")
    
    if not split_contigs:
        logger.info("No large contigs found for chimera detection")
        return {}
    
    # Load the trained model for generating embeddings
    from .models import train_siamese_network, generate_embeddings_for_fragments, get_model_path
    
    # Load or train the model
    model_path = get_model_path(args)
    if os.path.exists(model_path):
        logger.info(f"Loading trained model from {model_path}")
        device = get_torch_device()
        
        from .models import SiameseNetwork
        
        # Determine feature dimensions (same logic as in train_siamese_network)
        n_kmer_features = 136
        total_features = features_df.shape[1]
        n_coverage_features = total_features - n_kmer_features
        
        # Create model instance and load state dict
        model = SiameseNetwork(
            n_kmer_features=n_kmer_features, 
            n_coverage_features=n_coverage_features,
            embedding_dim=getattr(args, 'embedding_dim', 128)
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
    else:
        logger.info("Training new model for chimera detection...")
        model = train_siamese_network(features_df, args)
    
    # Get h1/h2 fragments that need embeddings
    h1_h2_fragments = []
    for halves in split_contigs.values():
        h1_h2_fragments.extend(halves['h1'])
        h1_h2_fragments.extend(halves['h2'])
    
    # Generate embeddings for h1/h2 fragments
    try:
        logger.info(f"Generating embeddings for {len(h1_h2_fragments)} h1/h2 fragments...")
        h1_h2_embeddings_df = generate_embeddings_for_fragments(model, features_df, h1_h2_fragments, args)
        logger.info(f"Generated embeddings for {len(h1_h2_embeddings_df)} h1/h2 fragments")
        
        if h1_h2_embeddings_df.empty:
            logger.warning("No embeddings generated for h1/h2 fragments")
            return {}
    except Exception as e:
        logger.error(f"Error generating embeddings for h1/h2 fragments: {e}")
        return {}
    
    # Analyze each split contig for chimeric patterns
    for base_contig, halves in split_contigs.items():
        if not halves['h1'] or not halves['h2']:
            # Skip if we don't have both halves
            logger.debug(f"Skipping {base_contig}: missing h1 or h2 fragments")
            continue
            
        # Validate that embeddings exist for all fragments
        missing_embeddings = []
        for fragment_list in [halves['h1'], halves['h2']]:
            for fragment in fragment_list:
                if fragment not in h1_h2_embeddings_df.index:
                    missing_embeddings.append(fragment)
        
        if missing_embeddings:
            logger.warning(f"Skipping {base_contig}: missing embeddings for {len(missing_embeddings)} fragments")
            continue
        
        # Get embeddings for each half
        h1_embeddings = h1_h2_embeddings_df.loc[halves['h1']]
        h2_embeddings = h1_h2_embeddings_df.loc[halves['h2']]
        
        # Calculate mean embeddings for each half
        h1_mean = h1_embeddings.mean(axis=0)
        h2_mean = h2_embeddings.mean(axis=0)
        
        # Perform permutation ANOVA to test for significant differences between halves
        is_possible_chimera, anova_results = _permutation_anova_chimera_test(
            h1_embeddings.values, h2_embeddings.values, n_permutations=1000
        )
        
        # Find cluster assignment for this base contig
        base_contig_cluster = None
        cluster_row = clusters_df[clusters_df['contig'] == base_contig]
        if not cluster_row.empty:
            base_contig_cluster = cluster_row.iloc[0]['cluster']
        
        # Calculate fragment count balance for additional info
        fragment_ratio = min(len(halves['h1']), len(halves['h2'])) / max(len(halves['h1']), len(halves['h2']))
        
        chimera_results[base_contig] = {
            'h1_fragment_count': int(len(halves['h1'])),
            'h2_fragment_count': int(len(halves['h2'])),
            'fragment_ratio': float(fragment_ratio),
            'cluster_assignment': str(base_contig_cluster) if base_contig_cluster is not None else None,
            'is_possible_chimera': bool(is_possible_chimera),
            **anova_results  # Include all ANOVA statistics
        }
        
        if is_possible_chimera:
            logger.info(f"Possible chimeric contig detected: {base_contig} "
                       f"(p-value: {anova_results['p_value']:.4f}, "
                       f"F-stat: {anova_results['f_statistic']:.3f}, "
                       f"fragment_ratio: {fragment_ratio:.3f})")
    
    # Save results only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        results_path = os.path.join(args.output, "chimera_detection_results.json")
        with open(results_path, 'w') as f:
            json.dump(chimera_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
    
    chimeric_count = sum(1 for r in chimera_results.values() if r['is_possible_chimera'])
    logger.info(f"Chimera detection complete. Found {chimeric_count} possible chimeric contigs out of {len(chimera_results)} analyzed")
    
    return chimera_results


def plot_umap(umap_df, output_dir):
    """Create UMAP scatter plot and save as PDF."""
    plt.figure(figsize=(12, 8))

    noise_label = "noise"
    noise_mask = umap_df["cluster"] == noise_label
    if noise_mask.any():
        plt.scatter(
            umap_df[noise_mask]["UMAP1"],
            umap_df[noise_mask]["UMAP2"],
            c="lightgray",
            alpha=0.5,
            s=20,
            label=f"Noise ({noise_mask.sum()})",
        )

    actual_cluster_ids_series = umap_df["cluster"][
        ~noise_mask & ~umap_df["cluster"].isna()
    ]
    try:
        actual_cluster_ids = sorted(actual_cluster_ids_series.unique())
    except (IndexError, ValueError):
        actual_cluster_ids = list(actual_cluster_ids_series.unique())

    n_actual_clusters = len(actual_cluster_ids)

    if n_actual_clusters > 0:
        # Use tab20 colormap for better cluster distinction
        import matplotlib.cm as cm
        if n_actual_clusters <= 20:
            colors = cm.tab20(np.linspace(0, 1, n_actual_clusters))
        else:
            # Pre-generate tab20 colors and cycle efficiently
            base_colors = cm.tab20(np.linspace(0, 1, 20))
            colors = [base_colors[i % 20] for i in range(n_actual_clusters)]

        for i, cluster_id in enumerate(actual_cluster_ids):
            cluster_mask = umap_df["cluster"] == cluster_id
            cluster_count = cluster_mask.sum()
            plt.scatter(
                umap_df[cluster_mask]["UMAP1"],
                umap_df[cluster_mask]["UMAP2"],
                c=[colors[i]],
                s=30,
                label=f"{cluster_id} ({cluster_count})",
                alpha=0.7,
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    elif not noise_mask.any():
        logger.warning("No clusters or noise found in UMAP data")

    plt.title("UMAP projection of contig embeddings")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "umap_plot.pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved UMAP plot to {plot_path}")








def cluster_contigs(embeddings_df, fragments_dict, args):
    """Main clustering function that orchestrates the clustering process."""
    bins_path = os.path.join(args.output, "bins.csv")

    # Check if bins file already exists
    if os.path.exists(bins_path):
        logger.info(f"Loading existing bins from {bins_path}")
        return pd.read_csv(bins_path)

    # Load eukaryotic classification scores if available
    eukaryotic_scores = {}
    from .features import get_classification_results_path
    
    classification_results_path = get_classification_results_path(args.fasta, args.output)
    
    if os.path.exists(classification_results_path):
        try:
            classification_df = pd.read_csv(classification_results_path, sep='\t')
            eukaryotic_scores = dict(zip(classification_df['header'], classification_df['eukar_score']))
            logger.info(f"Loaded eukaryotic scores for {len(eukaryotic_scores)} contigs")
        except Exception as e:
            logger.warning(f"Could not load classification results: {e}")
            eukaryotic_scores = {}
    else:
        logger.warning(f"Eukaryotic classification file not found: {classification_results_path}")

    # Embeddings are already L2 normalized when saved to CSV
    logger.debug("Using pre-normalized embeddings for clustering...")
    norm_data = embeddings_df.values
    contig_names = list(embeddings_df.index)
    
    # Log essential data properties
    logger.info(f"Clustering {len(contig_names)} contigs with {embeddings_df.shape[1]}D embeddings")
    if eukaryotic_scores:
        scores_array = np.array(list(eukaryotic_scores.values()))
        high_conf_count = sum(1 for s in scores_array if s > 0.95)
        logger.info(f"Eukaryotic classification: {len(eukaryotic_scores)} scored, {high_conf_count} high-confidence (>0.95)")

    # Apply k-means pre-filtering if enabled and eukaryotic scores are available
    working_contig_names = contig_names
    working_embeddings_df = embeddings_df
    kmeans_filter_stats = None
    
    if not getattr(args, 'skip_kmeans_filtering', False) and eukaryotic_scores:
        logger.info("Applying k-means pre-filtering to remove small, low-confidence clusters...")
        
        # Hard-coded filtering parameters (smaller threshold as requested)
        small_cluster_threshold = 0.05  # 5% instead of 10%
        min_eukaryotic_score = 0.95
        max_iterations = 10
        
        # Extract contig names without fragment suffixes for eukaryotic score lookup
        original_contig_names = [extract_base_contig_name(name) for name in embeddings_df.index]
        
        # Apply k-means filtering
        filtered_embeddings, filtered_contig_names, kmeans_filter_stats = _iterative_kmeans_filtering(
            norm_data, 
            original_contig_names,
            eukaryotic_scores,
            small_cluster_threshold=small_cluster_threshold,
            min_eukaryotic_score=min_eukaryotic_score,
            max_iterations=max_iterations
        )
        
        # Update working data if filtering removed contigs
        if len(filtered_contig_names) < len(original_contig_names):
            # Create mapping from original names back to fragment names with .original suffix
            filtered_fragment_names = [f"{name}.original" for name in filtered_contig_names]
            
            # Filter embeddings dataframe to keep only remaining contigs
            working_embeddings_df = embeddings_df.loc[filtered_fragment_names]
            working_contig_names = list(working_embeddings_df.index)
            norm_data = working_embeddings_df.values
            
            logger.info(f"K-means filtering: {len(working_contig_names)}/{len(contig_names)} contigs selected for HDBSCAN")
        else:
            logger.info("K-means filtering: no contigs removed, proceeding with all data")
        
        # Save filtering statistics if keeping intermediate files
        if getattr(args, "keep_intermediate", False) and kmeans_filter_stats:
            kmeans_stats_path = os.path.join(args.output, "kmeans_filtering_stats.json")
            with open(kmeans_stats_path, 'w') as f:
                json.dump(kmeans_filter_stats, f, indent=2)
            logger.debug(f"Saved k-means filtering statistics to {kmeans_stats_path}")
    else:
        if not eukaryotic_scores:
            logger.info("Skipping k-means pre-filtering: no eukaryotic classification scores available")
        else:
            logger.info("K-means pre-filtering disabled by --skip-kmeans-filtering flag")
    
    # Choose clustering method based on user preference
    clustering_method = getattr(args, 'clustering_method', 'hdbscan')
    
    if clustering_method == 'leiden':
        logger.info("Using Leiden clustering")
        leiden_resolution = getattr(args, 'leiden_resolution', 1.0)
        leiden_k_neighbors = getattr(args, 'leiden_k_neighbors', 15)
        leiden_similarity_threshold = getattr(args, 'leiden_similarity_threshold', 0.1)
        
        logger.info(f"Running Leiden on {len(working_contig_names)} contigs "
                   f"(resolution={leiden_resolution}, k={leiden_k_neighbors}, "
                   f"similarity_threshold={leiden_similarity_threshold})")
        
        cluster_labels = _leiden_clustering(
            norm_data,
            k=leiden_k_neighbors,
            similarity_threshold=leiden_similarity_threshold,
            resolution=leiden_resolution,
            random_state=42
        )
        
    else:
        # Use HDBSCAN clustering on filtered data
        logger.info("Using HDBSCAN clustering")
        precluster_success = False
        
        # Create HDBSCAN clusterer (GPU-accelerated if available)
        logger.info(f"Running HDBSCAN on {len(working_contig_names)} contigs (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})")
        clusterer, is_gpu = _create_hdbscan_clusterer(
            args.min_cluster_size, 
            args.min_samples,
            getattr(args, 'cluster_selection_epsilon', 0.3),
            use_gpu=getattr(args, 'use_gpu', True)
        )
        
        if is_gpu:
            logger.info("Using GPU-accelerated HDBSCAN clustering")
            # Convert to GPU arrays for cuML
            gpu_data = cp.asarray(norm_data, dtype=cp.float32)
            cluster_labels = clusterer.fit_predict(gpu_data)
            # Convert back to CPU numpy array
            cluster_labels = cp.asnumpy(cluster_labels)
        else:
            logger.info("Using CPU HDBSCAN clustering")
            cluster_labels = clusterer.fit_predict(norm_data)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for label in cluster_labels if label == -1)
    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0]) if n_clusters > 0 else []
    formatted_labels = [
        f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
    ]

    # Create clusters dataframe with original contig names (without .original suffix)
    final_original_contig_names = [
        extract_base_contig_name(name) for name in working_embeddings_df.index
    ]
    contig_clusters_df = pd.DataFrame(
        {"contig": final_original_contig_names, "cluster": formatted_labels}
    )

    # Use contig-level clusters directly
    clusters_df = contig_clusters_df

    # Count and report final results
    final_counts = contig_clusters_df["cluster"].value_counts().to_dict()
    n_clusters = len([k for k in final_counts.keys() if k != "noise"])
    n_noise = final_counts.get("noise", 0)
    logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise contigs, sizes: {dict(sorted(final_counts.items()))}")

    # Check if only one bin was detected and perform reclustering
    if n_clusters == 1:
        if clustering_method == 'leiden':
            logger.info("Only one bin detected. Attempting reclustering with increased resolution...")
            
            # Increase resolution by 0.5
            new_resolution = leiden_resolution + 0.5
            logger.info(f"Reclustering with resolution={new_resolution} (original: {leiden_resolution})")
            
            # Perform Leiden reclustering
            recluster_labels = _leiden_clustering(
                norm_data,
                k=leiden_k_neighbors,
                similarity_threshold=leiden_similarity_threshold,
                resolution=new_resolution,
                random_state=42
            )
        else:
            logger.info("Only one bin detected. Attempting reclustering with increased min_cluster_size...")
            
            # Increase min_cluster_size by 1
            new_min_cluster_size = args.min_cluster_size + 1
            logger.info(f"Reclustering with min_cluster_size={new_min_cluster_size} (original: {args.min_cluster_size})")
            
            # Create new HDBSCAN clusterer with increased min_cluster_size
            reclusterer, is_gpu_recluster = _create_hdbscan_clusterer(
                new_min_cluster_size, 
                args.min_samples,
                getattr(args, 'cluster_selection_epsilon', 0.3),
                use_gpu=getattr(args, 'use_gpu', True)
            )
            
            # Perform reclustering
            if is_gpu_recluster:
                logger.info("Using GPU-accelerated HDBSCAN for reclustering")
                gpu_data = cp.asarray(norm_data, dtype=cp.float32)
                recluster_labels = reclusterer.fit_predict(gpu_data)
                recluster_labels = cp.asnumpy(recluster_labels)
            else:
                logger.info("Using CPU HDBSCAN for reclustering")
                recluster_labels = reclusterer.fit_predict(norm_data)
        
        n_recluster_clusters = len(set(recluster_labels)) - (1 if -1 in recluster_labels else 0)
        n_recluster_noise = sum(1 for label in recluster_labels if label == -1)
        recluster_sizes = np.bincount(recluster_labels[recluster_labels >= 0]) if n_recluster_clusters > 0 else []
        logger.info(f"Reclustering result: {n_recluster_clusters} clusters, {n_recluster_noise} noise points, sizes: {recluster_sizes.tolist() if hasattr(recluster_sizes, 'tolist') else list(recluster_sizes)}")
        
        # Only use reclustering results if we got more than one cluster
        if n_recluster_clusters > 1:
            logger.info(f"Reclustering successful: {n_recluster_clusters} clusters found. Using reclustering results.")
            
            # Update cluster labels with reclustering results
            cluster_labels = recluster_labels
            clusterer = reclusterer
            
            
            # Update formatted labels and contig clusters dataframe
            formatted_labels = [
                f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
            ]
            
            contig_clusters_df = pd.DataFrame(
                {"contig": final_original_contig_names, "cluster": formatted_labels}
            )
            
            # Update final counts
            final_counts = contig_clusters_df["cluster"].value_counts().to_dict()
            n_clusters = len([k for k in final_counts.keys() if k != "noise"])
            n_noise = final_counts.get("noise", 0)
            logger.info(f"Final clustering result after reclustering: {n_clusters} clusters, {n_noise} noise contigs, sizes: {dict(sorted(final_counts.items()))}")
            
            # Update clusters_df for consistency
            clusters_df = contig_clusters_df
        else:
            logger.info(f"Reclustering did not improve results ({n_recluster_clusters} clusters). Keeping original single bin.")
    else:
        logger.info(f"Multiple bins detected ({n_clusters}). No reclustering needed.")

    # Filter out noise contigs for final bins.csv
    final_bins_df = contig_clusters_df[contig_clusters_df["cluster"] != "noise"].copy()
    
    # Save final bins (excluding noise)
    final_bins_df.to_csv(bins_path, index=False)

    # Count contigs per cluster using utility function
    logger.debug("Counting contigs per cluster...")
    cluster_contig_counts = group_contigs_by_cluster(contig_clusters_df)

    # Count and report noise contigs
    noise_contigs = cluster_contig_counts.get("noise", set())
    logger.info(f"Contigs classified as noise: {len(noise_contigs)}")


    # Use UMAP for visualization only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        logger.debug("Performing UMAP dimensionality reduction for visualization...")
        umap_embeddings_df = working_embeddings_df
        logger.debug(f"Using filtered data for UMAP: {umap_embeddings_df.shape[0]} contigs")
        
        # Create UMAP reducer
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=42,
            n_jobs=1,  # Set to 1 to avoid warning about random_state + parallelism
        )

        # Fit and transform the embeddings
        logger.debug(f"Running UMAP on {umap_embeddings_df.shape[0]} contigs...")
        umap_embeddings = reducer.fit_transform(umap_embeddings_df.values)

        # Save UMAP embeddings for visualization
        umap_df = pd.DataFrame(
            umap_embeddings, columns=["UMAP1", "UMAP2"], index=umap_embeddings_df.index
        )
        # Map clusters to UMAP data (removing fragment suffixes for matching)
        umap_original_names = [
            extract_base_contig_name(name) for name in umap_embeddings_df.index
        ]
        umap_clusters = []
        for original_name in umap_original_names:
            cluster_assignment = contig_clusters_df[
                contig_clusters_df["contig"] == original_name
            ]["cluster"]
            if not cluster_assignment.empty:
                umap_clusters.append(cluster_assignment.iloc[0])
            else:
                umap_clusters.append("noise")

        umap_df["cluster"] = umap_clusters
        
        # Save UMAP embeddings and plot
        umap_path = os.path.join(args.output, "umap_embeddings.csv")
        umap_df.to_csv(umap_path)
        logger.debug(f"Saved UMAP embeddings to {umap_path}")

        # Create and save UMAP plot
        logger.debug("Creating UMAP visualization plot...")
        plot_umap(umap_df, args.output)

    # Perform chimera detection for large contigs
    if not getattr(args, 'skip_chimera_detection', False):
        logger.info("Running chimera detection on large contigs...")
        chimera_results = detect_chimeric_contigs(working_embeddings_df, clusters_df, args)

    logger.info(f"Saved contig-level clusters to {bins_path}")

    return clusters_df
