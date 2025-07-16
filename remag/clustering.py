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
from loguru import logger
import os
import json

from .utils import extract_base_contig_name
import torch


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
    features_parquet_path = os.path.join(args.output, "features.parquet")
    features_csv_path = os.path.join(args.output, "features.csv")
    
    features_df = None
    if os.path.exists(features_parquet_path):
        try:
            features_df = pd.read_parquet(features_parquet_path)
        except Exception as e:
            logger.error(f"Error loading features data from parquet: {e}")
            return {}
    elif os.path.exists(features_csv_path):
        try:
            features_df = pd.read_csv(features_csv_path, index_col=0)
        except Exception as e:
            logger.error(f"Error loading features data from csv: {e}")
            return {}
    else:
        logger.warning(f"Features file not found at {features_parquet_path} or {features_csv_path}, skipping chimera detection")
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
    from .models import train_siamese_network, generate_embeddings_for_fragments
    
    # Load or train the model
    model_path = os.path.join(args.output, "siamese_model.pt")
    if os.path.exists(model_path):
        logger.info(f"Loading trained model from {model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        # Import the model class
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
    
    # Save results
    results_path = os.path.join(args.output, "chimera_detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(chimera_results, f, indent=2)
    
    chimeric_count = sum(1 for r in chimera_results.values() if r['is_possible_chimera'])
    logger.info(f"Chimera detection complete. Found {chimeric_count} possible chimeric contigs out of {len(chimera_results)} analyzed")
    logger.info(f"Results saved to {results_path}")
    
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






def soft_clustering_noise_recovery(clusterer, embeddings, cluster_labels, noise_threshold=0.5, output_dir=None):
    """
    Recover noise points using HDBSCAN's soft clustering membership vectors.
    
    Uses all_points_membership_vectors to get soft cluster assignments for all points,
    then assigns noise points to clusters with highest membership probability above threshold.
    
    Args:
        clusterer: Fitted HDBSCAN clusterer with prediction_data=True
        embeddings: Normalized embedding matrix used for clustering (unused but kept for interface compatibility)
        cluster_labels: Original cluster labels from HDBSCAN
        noise_threshold: Minimum membership probability to assign noise point to cluster
        output_dir: Directory to save debugging information
    
    Returns:
        tuple: (recovered_labels, recovery_stats)
            - recovered_labels: Updated cluster labels with recovered noise points
            - recovery_stats: Dict with recovery statistics
    """
    logger.debug(f"Attempting soft clustering noise recovery with threshold={noise_threshold}")
    
    # Find noise points
    noise_mask = cluster_labels == -1
    noise_indices = np.where(noise_mask)[0]
    n_noise_original = len(noise_indices)
    
    if n_noise_original == 0:
        logger.debug("No noise points to recover")
        return cluster_labels.copy(), {"noise_original": 0, "noise_recovered": 0, "noise_remaining": 0}
    
    logger.debug(f"Found {n_noise_original} noise points to attempt recovery")
    
    # Get soft cluster membership vectors for all points
    try:
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        logger.debug(f"Generated soft clustering vectors: {soft_clusters.shape}")
    except Exception as e:
        logger.warning(f"Failed to generate soft clustering vectors: {e}")
        return cluster_labels.copy(), {"noise_original": n_noise_original, "noise_recovered": 0, "noise_remaining": n_noise_original}
    
    # Create copy of original labels for modification
    recovered_labels = cluster_labels.copy()
    
    # Track recovery statistics
    recovered_count = 0
    cluster_assignments = {}
    noise_analysis = []
    
    # Get valid cluster indices (non-noise clusters from original clustering)
    valid_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    
    # Analyze each noise point
    for noise_idx in noise_indices:
        membership_vector = soft_clusters[noise_idx]
        
        # Find the cluster with highest membership probability
        best_cluster_internal_idx = np.argmax(membership_vector)
        best_membership = membership_vector[best_cluster_internal_idx]
        
        # Map internal cluster index to actual cluster label
        # HDBSCAN's membership vectors correspond to clusters 0, 1, 2, ... in order
        if best_cluster_internal_idx < len(valid_clusters):
            best_cluster_label = valid_clusters[best_cluster_internal_idx]
        else:
            best_cluster_label = -1
        
        # Store analysis data for debugging
        noise_analysis.append({
            "noise_index": int(noise_idx),
            "membership_vector": membership_vector.tolist(),
            "best_cluster_internal_idx": int(best_cluster_internal_idx),
            "best_cluster_label": int(best_cluster_label) if best_cluster_label != -1 else None,
            "best_membership": float(best_membership),
            "assigned": bool(best_membership >= noise_threshold and best_cluster_label != -1)
        })
        
        logger.debug(f"Noise point {noise_idx}: best_cluster={best_cluster_label}, membership={best_membership:.3f}, threshold={noise_threshold:.3f}")
        
        # Assign to cluster if membership is above threshold
        if best_membership >= noise_threshold and best_cluster_label != -1:
            recovered_labels[noise_idx] = best_cluster_label
            recovered_count += 1
            
            # Track cluster assignments
            if best_cluster_label not in cluster_assignments:
                cluster_assignments[best_cluster_label] = 0
            cluster_assignments[best_cluster_label] += 1
            
            logger.debug(f"Recovered noise point {noise_idx} -> cluster {best_cluster_label} (membership: {best_membership:.3f})")
    
    n_noise_remaining = n_noise_original - recovered_count
    
    # Log recovery results
    logger.info(f"Soft clustering noise recovery: {recovered_count}/{n_noise_original} points recovered ({n_noise_remaining} remain noise)")
    
    if cluster_assignments:
        assignment_summary = ", ".join([f"cluster_{k}: {v}" for k, v in sorted(cluster_assignments.items())])
        logger.debug(f"Recovery distribution: {assignment_summary}")
    
    # Calculate membership statistics
    memberships_array = np.array([point["best_membership"] for point in noise_analysis])
    recovery_stats = {
        "method": "soft_clustering_hdbscan",
        "noise_original": n_noise_original,
        "noise_recovered": recovered_count,
        "noise_remaining": n_noise_remaining,
        "recovery_rate": recovered_count / n_noise_original if n_noise_original > 0 else 0.0,
        "cluster_assignments": {int(k): v for k, v in cluster_assignments.items()},
        "membership_threshold": noise_threshold,
        "membership_stats": {
            "min": float(memberships_array.min()) if len(memberships_array) > 0 else 0.0,
            "max": float(memberships_array.max()) if len(memberships_array) > 0 else 0.0,
            "mean": float(memberships_array.mean()) if len(memberships_array) > 0 else 0.0,
            "median": float(np.median(memberships_array)) if len(memberships_array) > 0 else 0.0,
            "above_threshold": int(np.sum(memberships_array >= noise_threshold)) if len(memberships_array) > 0 else 0
        }
    }
    
    # Save detailed debugging information
    if output_dir:
        debug_info = {
            "recovery_stats": recovery_stats,
            "noise_point_analysis": noise_analysis,
            "soft_cluster_shape": list(soft_clusters.shape),
            "valid_clusters": valid_clusters.tolist()
        }
        
        debug_path = os.path.join(output_dir, "soft_clustering_debug.json")
        try:
            with open(debug_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            logger.debug(f"Saved soft clustering debug information to {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save soft clustering debug information: {e}")
    
    return recovered_labels, recovery_stats


def cluster_contigs(embeddings_df, fragments_dict, args):
    """Main clustering function that orchestrates the clustering process."""
    clusters_contigs_path = os.path.join(args.output, "clusters_contigs.csv")

    # Check if clusters file already exists
    if os.path.exists(clusters_contigs_path):
        logger.info(f"Loading existing clusters from {clusters_contigs_path}")
        return pd.read_csv(clusters_contigs_path)

    # Load eukaryotic classification scores if available
    eukaryotic_scores = {}
    base_name = os.path.basename(args.fasta)
    name_without_ext = os.path.splitext(base_name)[0]
    if name_without_ext.endswith(".gz"):
        name_without_ext = os.path.splitext(name_without_ext)[0]
    
    classification_results_path = os.path.join(
        args.output, f"{name_without_ext}_4cac_classification.tsv"
    )
    
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

    # Normalize the embeddings data for clustering
    logger.debug("Normalizing embeddings for clustering...")
    norm_data = normalize(embeddings_df.values, norm="l2")
    contig_names = list(embeddings_df.index)
    
    # Log essential data properties
    logger.info(f"Clustering {len(contig_names)} contigs with {embeddings_df.shape[1]}D embeddings")
    if eukaryotic_scores:
        scores_array = np.array(list(eukaryotic_scores.values()))
        high_conf_count = sum(1 for s in scores_array if s > 0.95)
        logger.info(f"Eukaryotic classification: {len(eukaryotic_scores)} scored, {high_conf_count} high-confidence (>0.95)")

    # Use HDBSCAN clustering directly on all data
    logger.info("Using HDBSCAN clustering")
    working_contig_names = contig_names
    working_embeddings_df = embeddings_df
    precluster_success = False
    
    # Create HDBSCAN clusterer
    logger.info(f"Running HDBSCAN on {len(working_contig_names)} contigs (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )

    cluster_labels = clusterer.fit_predict(norm_data)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for label in cluster_labels if label == -1)
    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0]) if n_clusters > 0 else []
    logger.info(f"HDBSCAN result: {n_clusters} clusters, {n_noise} noise points, sizes: {cluster_sizes.tolist() if hasattr(cluster_sizes, 'tolist') else list(cluster_sizes)}")
    
    # Attempt to recover noise points using soft clustering
    if n_noise > 0 and not getattr(args, 'skip_noise_recovery', False):
        logger.info(f"Attempting to recover {n_noise} noise points using soft clustering...")
        
        # Get noise recovery threshold from args or use default
        noise_threshold = getattr(args, 'noise_recovery_threshold', 0.5)
        
        recovered_labels, recovery_stats = soft_clustering_noise_recovery(
            clusterer, norm_data, cluster_labels, noise_threshold, args.output
        )
        
        # Update cluster labels with recovered points
        cluster_labels = recovered_labels
        
        # Update statistics
        n_clusters_after = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise_after = sum(1 for label in cluster_labels if label == -1)
        cluster_sizes_after = np.bincount(cluster_labels[cluster_labels >= 0]) if n_clusters_after > 0 else []
        
        logger.info(f"After noise recovery: {n_clusters_after} clusters, {n_noise_after} noise points, sizes: {cluster_sizes_after.tolist() if hasattr(cluster_sizes_after, 'tolist') else list(cluster_sizes_after)}")
        
        # Save recovery statistics
        recovery_stats_path = os.path.join(args.output, "soft_clustering_recovery_stats.json")
        with open(recovery_stats_path, 'w') as f:
            json.dump(recovery_stats, f, indent=2)
        logger.debug(f"Saved noise recovery statistics to {recovery_stats_path}")
    else:
        if n_noise == 0:
            logger.debug("No noise points found - skipping noise recovery")
        else:
            logger.debug("Noise recovery disabled - skipping soft clustering")
    
    formatted_labels = [
        f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
    ]

    # Create clusters dataframe with original contig names (without .original suffix)
    original_contig_names = [
        extract_base_contig_name(name) for name in embeddings_df.index
    ]
    contig_clusters_df = pd.DataFrame(
        {"contig": original_contig_names, "cluster": formatted_labels}
    )

    # Use contig-level clusters directly
    clusters_df = contig_clusters_df

    # Count and report final results
    final_counts = contig_clusters_df["cluster"].value_counts().to_dict()
    n_clusters = len([k for k in final_counts.keys() if k != "noise"])
    n_noise = final_counts.get("noise", 0)
    logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise contigs")
    logger.debug(f"Cluster sizes: {dict(sorted(final_counts.items()))}")

    # Save contig-level cluster assignments
    contig_clusters_df.to_csv(clusters_contigs_path, index=False)

    # Count contigs per cluster
    logger.debug("Counting contigs per cluster...")
    cluster_contig_counts = {}
    for _, row in contig_clusters_df.iterrows():
        cluster_id = row["cluster"]
        contig_name = row["contig"]

        if cluster_id not in cluster_contig_counts:
            cluster_contig_counts[cluster_id] = set()
        cluster_contig_counts[cluster_id].add(contig_name)

    # Count and report noise contigs
    noise_contigs = cluster_contig_counts.get("noise", set())
    logger.info(f"Contigs classified as noise: {len(noise_contigs)}")

    logger.debug("Contigs per cluster:")
    for cluster_id, original_contigs in cluster_contig_counts.items():
        count = len(original_contigs)
        logger.debug(f"  Cluster {cluster_id}: {count} contigs")

    # Use UMAP for visualization
    logger.debug("Performing UMAP dimensionality reduction for visualization...")
    umap_embeddings_df = embeddings_df
    logger.debug(f"Using original data for UMAP: {umap_embeddings_df.shape[0]} contigs")
    
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
    umap_path = os.path.join(args.output, "umap_embeddings.csv")
    umap_df.to_csv(umap_path)
    logger.debug(f"Saved UMAP embeddings to {umap_path}")

    # Create and save UMAP plot
    logger.debug("Creating UMAP visualization plot...")
    plot_umap(umap_df, args.output)

    # Perform chimera detection for large contigs
    if not getattr(args, 'skip_chimera_detection', False):
        logger.info("Running chimera detection on large contigs...")
        chimera_results = detect_chimeric_contigs(embeddings_df, clusters_df, args)
    else:
        logger.info("Skipping chimera detection as requested")

    logger.info(f"Saved contig-level clusters to {clusters_contigs_path}")

    return clusters_df
