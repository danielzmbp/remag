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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from loguru import logger
import os
import json

from .utils import extract_base_contig_name
import torch


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
        
        # Calculate cosine similarity between half means
        similarity = cosine_similarity([h1_mean], [h2_mean])[0][0]
        
        # Calculate intra-half similarity (consistency within each half)
        h1_intra_similarity = 0.0
        h2_intra_similarity = 0.0
        
        if len(halves['h1']) > 1:
            h1_similarities = []
            for i in range(len(halves['h1'])):
                for j in range(i+1, len(halves['h1'])):
                    sim = cosine_similarity([h1_embeddings.iloc[i]], [h1_embeddings.iloc[j]])[0][0]
                    h1_similarities.append(sim)
            h1_intra_similarity = np.mean(h1_similarities) if h1_similarities else 1.0
        else:
            h1_intra_similarity = 1.0
            
        if len(halves['h2']) > 1:
            h2_similarities = []
            for i in range(len(halves['h2'])):
                for j in range(i+1, len(halves['h2'])):
                    sim = cosine_similarity([h2_embeddings.iloc[i]], [h2_embeddings.iloc[j]])[0][0]
                    h2_similarities.append(sim)
            h2_intra_similarity = np.mean(h2_similarities) if h2_similarities else 1.0
        else:
            h2_intra_similarity = 1.0
        
        # Find cluster assignment for this base contig
        base_contig_cluster = None
        cluster_row = clusters_df[clusters_df['contig'] == base_contig]
        if not cluster_row.empty:
            base_contig_cluster = cluster_row.iloc[0]['cluster']
        
        # Enhanced chimera scoring
        # Factor 1: Inter-half similarity (low similarity suggests different sequences)
        inter_half_score = 0.0
        similarity_threshold = 0.8
        if similarity < similarity_threshold:
            inter_half_score = (similarity_threshold - similarity) / similarity_threshold
        
        # Factor 2: Intra-half consistency (high consistency within halves but low between suggests chimera)
        min_intra_similarity = min(h1_intra_similarity, h2_intra_similarity)
        consistency_bonus = 0.0
        if min_intra_similarity > 0.9 and similarity < 0.7:
            consistency_bonus = 0.3
        
        # Factor 3: Fragment count imbalance (very uneven splits might indicate assembly issues)
        fragment_ratio = min(len(halves['h1']), len(halves['h2'])) / max(len(halves['h1']), len(halves['h2']))
        balance_penalty = 0.0
        if fragment_ratio < 0.3:  # Very unbalanced
            balance_penalty = 0.2
        
        # Calculate final chimera score
        chimera_score = min(1.0, inter_half_score + consistency_bonus - balance_penalty)
        
        # Determine chimera likelihood
        if chimera_score >= 0.7:
            chimera_likelihood = 'high'
        elif chimera_score >= 0.4:
            chimera_likelihood = 'medium'
        else:
            chimera_likelihood = 'low'
        
        chimera_results[base_contig] = {
            'similarity': float(similarity),
            'h1_intra_similarity': float(h1_intra_similarity),
            'h2_intra_similarity': float(h2_intra_similarity),
            'h1_fragment_count': int(len(halves['h1'])),
            'h2_fragment_count': int(len(halves['h2'])),
            'fragment_ratio': float(fragment_ratio),
            'cluster_assignment': str(base_contig_cluster) if base_contig_cluster is not None else None,
            'chimera_score': float(chimera_score),
            'chimera_likelihood': str(chimera_likelihood),
            'is_likely_chimeric': bool(chimera_score >= 0.4)
        }
        
        if chimera_score >= 0.4:
            logger.info(f"Potential chimeric contig detected: {base_contig} "
                       f"(inter-half similarity: {similarity:.3f}, score: {chimera_score:.3f}, "
                       f"h1_intra: {h1_intra_similarity:.3f}, h2_intra: {h2_intra_similarity:.3f})")
    
    # Save results
    results_path = os.path.join(args.output, "chimera_detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(chimera_results, f, indent=2)
    
    chimeric_count = sum(1 for r in chimera_results.values() if r['is_likely_chimeric'])
    logger.info(f"Chimera detection complete. Found {chimeric_count} likely chimeric contigs out of {len(chimera_results)} analyzed")
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


def detect_two_cluster_structure(embeddings, contig_names, eukaryotic_scores=None, threshold=0.3):
    """
    Returns True if data shows strong 2-cluster structure using silhouette analysis
    and eukaryotic score separation.
    
    Args:
        embeddings: Normalized embedding matrix
        contig_names: List of contig names corresponding to embeddings
        eukaryotic_scores: Dict mapping contig names to eukaryotic scores
        threshold: Minimum silhouette score to consider well-separated
    
    Returns:
        tuple: (bool, int or None) - (True if strong 2-cluster structure detected, eukaryotic cluster index)
    """
    logger.debug(f"Testing 2-cluster structure with {len(embeddings)} contigs")
    
    if len(embeddings) < 4:
        logger.debug("Insufficient data for 2-cluster detection (< 4 contigs)")
        return False, None
    
    # Test k=2 clustering
    kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_2 = kmeans_2.fit_predict(embeddings)
    
    # Calculate silhouette score
    sil_score = silhouette_score(embeddings, labels_2)
    
    # Check cluster size balance (avoid 99%/1% splits)
    cluster_sizes = np.bincount(labels_2)
    min_cluster_ratio = min(cluster_sizes) / len(labels_2)
    
    logger.debug(f"2-cluster analysis: silhouette={sil_score:.3f}, sizes={cluster_sizes.tolist()}, min_ratio={min_cluster_ratio:.3f}")
    
    # Check eukaryotic score separation if available
    eukaryotic_cluster = None
    eukaryotic_separation_good = True
    
    if eukaryotic_scores:
        cluster_0_eukar_scores = []
        cluster_1_eukar_scores = []
        
        for i, contig_name in enumerate(contig_names):
            # Remove fragment suffixes if present
            clean_contig_name = extract_base_contig_name(contig_name)
            if clean_contig_name in eukaryotic_scores:
                eukar_score = eukaryotic_scores[clean_contig_name]
                if labels_2[i] == 0:
                    cluster_0_eukar_scores.append(eukar_score)
                else:
                    cluster_1_eukar_scores.append(eukar_score)
        
        if cluster_0_eukar_scores and cluster_1_eukar_scores:
            mean_eukar_0 = np.mean(cluster_0_eukar_scores)
            mean_eukar_1 = np.mean(cluster_1_eukar_scores)
            
            # Count high-confidence eukaryotes (>0.95) in each cluster
            high_conf_eukar_0 = sum(1 for score in cluster_0_eukar_scores if score > 0.95)
            high_conf_eukar_1 = sum(1 for score in cluster_1_eukar_scores if score > 0.95)
            
            logger.debug(f"Eukaryotic separation: cluster_0={mean_eukar_0:.3f}({high_conf_eukar_0}), cluster_1={mean_eukar_1:.3f}({high_conf_eukar_1})")
            
            # Check if both clusters have high-confidence eukaryotes (bad separation)
            if high_conf_eukar_0 > 0 and high_conf_eukar_1 > 0:
                logger.debug("Both clusters contain high-confidence eukaryotes - poor separation")
                eukaryotic_separation_good = False
            else:
                # Identify which cluster has more/higher eukaryotic content
                if mean_eukar_0 > mean_eukar_1:
                    eukaryotic_cluster = 0
                else:
                    eukaryotic_cluster = 1
                
                # Require meaningful separation (>0.2 difference in mean scores)
                score_separation = abs(mean_eukar_0 - mean_eukar_1)
                if score_separation < 0.2:
                    eukaryotic_separation_good = False
                logger.debug(f"Eukaryotic cluster: {eukaryotic_cluster}, score_separation: {score_separation:.3f}")
    else:
        logger.debug("No eukaryotic scores - skipping separation check")
    
    is_good_structure = (sil_score > threshold and 
                        min_cluster_ratio > 0.1 and 
                        eukaryotic_separation_good)
    
    logger.info(f"2-cluster decision: silhouette={sil_score:.3f}, ratio={min_cluster_ratio:.3f}, eukar_sep={eukaryotic_separation_good} => use_kmeans={is_good_structure}")
    
    
    return is_good_structure, eukaryotic_cluster


def multi_k_bacterial_removal(embeddings_df, contig_names, eukaryotic_scores=None, k_values=[3, 4, 5]):
    """
    Use multi-k K-means clustering to identify and remove purely bacterial clusters.
    Only removes clusters with zero high-confidence eukaryotes, preserving all eukaryotic material.
    
    Args:
        embeddings_df: DataFrame with contig embeddings
        contig_names: List of contig names
        eukaryotic_scores: Dict mapping contig names to eukaryotic scores
        k_values: List of k-values to try
    
    Returns:
        tuple: (filtered_embeddings_df, filtered_contig_names, success_flag)
    """
    logger.info(f"Starting multi-k pre-clustering with k={k_values}...")
    logger.debug(f"Input: {len(embeddings_df)} embeddings, {len(contig_names)} contig names")
    
    # Check if we have eukaryotic scores
    if not eukaryotic_scores:
        logger.warning("No eukaryotic scores available - skipping pre-clustering")
        return embeddings_df, contig_names, False
    
    # Debug: Analyze eukaryotic score distribution
    logger.debug(f"Eukaryotic scores loaded for {len(eukaryotic_scores)} contigs")
    scores_array = np.array(list(eukaryotic_scores.values()))
    logger.debug(f"Eukaryotic score distribution: min={scores_array.min():.3f}, max={scores_array.max():.3f}, mean={scores_array.mean():.3f}")
    
    # Count contigs by score thresholds
    high_conf_count = sum(1 for s in scores_array if s > 0.95)
    medium_conf_count = sum(1 for s in scores_array if 0.7 <= s <= 0.95)
    low_conf_count = sum(1 for s in scores_array if 0.3 <= s < 0.7)
    bacterial_count = sum(1 for s in scores_array if s < 0.3)
    
    logger.debug(f"Score distribution: high_conf(>0.95)={high_conf_count}, medium_conf(0.7-0.95)={medium_conf_count}, low_conf(0.3-0.7)={low_conf_count}, bacterial(<0.3)={bacterial_count}")
    
    # Check how many contig names have corresponding eukaryotic scores
    matched_scores = 0
    for contig_name in contig_names:
        clean_name = extract_base_contig_name(contig_name)
        if clean_name in eukaryotic_scores:
            matched_scores += 1
    
    logger.debug(f"Contig name matching: {matched_scores}/{len(contig_names)} contigs have eukaryotic scores")
    
    # Normalize embeddings for clustering
    norm_data = normalize(embeddings_df.values, norm="l2")
    
    best_removal_ratio = 0
    best_kept_indices = None
    best_k = None
    max_bacterial_content = 0
    
    for k in k_values:
        if len(embeddings_df) <= k:
            logger.debug(f"Skipping k={k} (insufficient data: {len(embeddings_df)} contigs)")
            continue
            
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(norm_data)
        
        cluster_sizes = np.bincount(labels)
        
        # Analyze each cluster for high-confidence eukaryotes
        clusters_to_keep = set()
        cluster_analysis = {}
        
        logger.debug(f"K={k}: Analyzing {k} clusters from {len(contig_names)} contigs")
        
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)
            
            # Count high-confidence eukaryotes in this cluster
            high_conf_eukaryotes = 0
            medium_conf_eukaryotes = 0
            low_conf_eukaryotes = 0
            bacterial_contigs = 0
            total_with_scores = 0
            score_sum = 0.0
            
            for idx in cluster_indices:
                contig_name = extract_base_contig_name(contig_names[idx])
                if contig_name in eukaryotic_scores:
                    total_with_scores += 1
                    score = eukaryotic_scores[contig_name]
                    score_sum += score
                    
                    if score > 0.95:
                        high_conf_eukaryotes += 1
                    elif score >= 0.7:
                        medium_conf_eukaryotes += 1
                    elif score >= 0.3:
                        low_conf_eukaryotes += 1
                    else:
                        bacterial_contigs += 1
            
            cluster_size_ratio = cluster_size / len(contig_names)
            avg_score = score_sum / total_with_scores if total_with_scores > 0 else 0.0
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'size_ratio': cluster_size_ratio,
                'high_conf_eukaryotes': high_conf_eukaryotes,
                'medium_conf_eukaryotes': medium_conf_eukaryotes,
                'low_conf_eukaryotes': low_conf_eukaryotes,
                'bacterial_contigs': bacterial_contigs,
                'total_with_scores': total_with_scores,
                'avg_score': avg_score
            }
            
            # Log detailed cluster analysis
            logger.debug(f"K={k}, cluster {cluster_id}: size={cluster_size} ({cluster_size_ratio:.1%}), "
                        f"high_conf={high_conf_eukaryotes}, medium_conf={medium_conf_eukaryotes}, "
                        f"low_conf={low_conf_eukaryotes}, bacterial={bacterial_contigs}, "
                        f"avg_score={avg_score:.3f}, with_scores={total_with_scores}/{cluster_size}")
            
            # Keep clusters with ANY high-confidence eukaryotes
            # Also keep reasonably sized clusters (>5% of total) even without high-conf eukaryotes
            # (they might contain lower-scoring eukaryotes)
            keep_high_conf = high_conf_eukaryotes > 0
            keep_size = cluster_size_ratio > 0.05
            
            if keep_high_conf or keep_size:
                clusters_to_keep.add(cluster_id)
                reason = []
                if keep_high_conf:
                    reason.append(f"{high_conf_eukaryotes} high-conf eukaryotes")
                if keep_size:
                    reason.append(f"{cluster_size_ratio:.1%} size")
                logger.debug(f"K={k}, cluster {cluster_id}: KEEPING ({', '.join(reason)})")
            else:
                logger.debug(f"K={k}, cluster {cluster_id}: REMOVING (0 high-conf eukaryotes, {cluster_size_ratio:.1%} size)")
        
        # Calculate how much we can safely remove
        kept_indices = np.where(np.isin(labels, list(clusters_to_keep)))[0]
        removal_ratio = 1 - (len(kept_indices) / len(contig_names))
        
        removed_clusters = set(range(k)) - clusters_to_keep
        logger.debug(f"K={k}: keeping clusters {sorted(clusters_to_keep)}, removing clusters {sorted(removed_clusters)}")
        logger.debug(f"K={k}: keeping {len(clusters_to_keep)}/{k} clusters, removal={removal_ratio:.1%} ({len(contig_names) - len(kept_indices)}/{len(contig_names)} contigs)")
        
        # Show what we're removing
        if removed_clusters:
            for cluster_id in removed_clusters:
                analysis = cluster_analysis[cluster_id]
                logger.debug(f"K={k}: removing cluster {cluster_id} with {analysis['bacterial_contigs']} bacterial contigs, "
                            f"avg_score={analysis['avg_score']:.3f}, size={analysis['size']}")
        
        # Track the best k-value (one that removes the most purely bacterial content)
        if removal_ratio > best_removal_ratio:
            best_removal_ratio = removal_ratio
            best_kept_indices = kept_indices
            best_k = k
            logger.debug(f"New best k={k} with removal={removal_ratio:.1%}")
        
        # Track maximum bacterial content found across all k-values
        max_bacterial_content = max(max_bacterial_content, removal_ratio)
    
    # Final decision logging
    logger.debug(f"Final decision: best_k={best_k}, best_removal_ratio={best_removal_ratio:.1%}, "
                f"max_bacterial_content={max_bacterial_content:.1%}")
    logger.debug(f"Decision threshold: removal_ratio > 0.02 (2%)")
    
    # Apply the best clustering result if it removes a meaningful amount
    if best_kept_indices is not None and best_removal_ratio > 0.02:  # At least 2% removal
        filtered_embeddings_df = embeddings_df.iloc[best_kept_indices]
        filtered_contig_names = [contig_names[i] for i in best_kept_indices]
        
        bacteria_removed = len(contig_names) - len(filtered_contig_names)
        logger.info(f"Pre-clustering successful: k={best_k}, kept {len(filtered_contig_names)}/{len(contig_names)} contigs ({best_removal_ratio:.1%} bacterial removal)")
        
        # Show final statistics of what was removed
        if best_k:
            # Re-run k-means with best_k to get final cluster assignments for logging
            kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = kmeans_final.fit_predict(norm_data)
            final_clusters_to_keep = set()
            
            # Determine which clusters were kept
            for i, kept_idx in enumerate(best_kept_indices):
                final_clusters_to_keep.add(final_labels[kept_idx])
            
            removed_clusters_final = set(range(best_k)) - final_clusters_to_keep
            logger.debug(f"Final removal: removed clusters {sorted(removed_clusters_final)} from k={best_k} clustering")
        
        return filtered_embeddings_df, filtered_contig_names, True
    else:
        logger.info(f"Pre-clustering: minimal bacterial content found ({max_bacterial_content:.1%})")
        logger.debug(f"Reason: best_removal_ratio={best_removal_ratio:.1%} <= 0.02 threshold")
        return embeddings_df, contig_names, False


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

    # Check if data shows strong 2-cluster structure for pre-filtering
    is_good_structure, eukaryotic_cluster = detect_two_cluster_structure(
        norm_data, contig_names, eukaryotic_scores
    )
    
    # Always use HDBSCAN clustering, but pre-filter using 2-cluster structure if detected
    logger.info("Using HDBSCAN clustering")
    
    # Apply pre-clustering - either the existing multi-k method or 2-cluster filtering
    if is_good_structure and eukaryotic_cluster is not None:
        logger.info("Detected well-separated eukaryotic/bacterial structure - filtering small cluster before HDBSCAN")
        
        # Use K-means with 2 clusters to identify and filter out the small cluster
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(norm_data)
        cluster_dist = np.bincount(cluster_labels)
        logger.debug(f"K-means pre-filtering result: {cluster_dist.tolist()}")
        
        # Keep only the eukaryotic cluster for HDBSCAN
        eukaryotic_indices = [i for i, label in enumerate(cluster_labels) if label == eukaryotic_cluster]
        logger.info(f"Keeping eukaryotic cluster {eukaryotic_cluster} ({len(eukaryotic_indices)} contigs) for HDBSCAN, filtering out cluster {1-eukaryotic_cluster} ({cluster_dist[1-eukaryotic_cluster]} contigs)")
        
        # Filter data for HDBSCAN
        filtered_embeddings_df = embeddings_df.iloc[eukaryotic_indices]
        filtered_contig_names = [contig_names[i] for i in eukaryotic_indices]
        norm_data = normalize(filtered_embeddings_df.values, norm="l2")
        working_contig_names = filtered_contig_names
        working_embeddings_df = filtered_embeddings_df
        precluster_success = True
        
    elif args.enable_preclustering:
        logger.info("Pre-clustering enabled - attempting bacterial removal...")
        filtered_embeddings_df, filtered_contig_names, precluster_success = multi_k_bacterial_removal(
            embeddings_df, contig_names, eukaryotic_scores
        )
        
        if precluster_success:
            # Update data for HDBSCAN to use only the filtered (eukaryotic) contigs
            logger.info("Using pre-filtered eukaryotic contigs for HDBSCAN")
            norm_data = normalize(filtered_embeddings_df.values, norm="l2")
            working_contig_names = filtered_contig_names
            working_embeddings_df = filtered_embeddings_df
        else:
            logger.info("Pre-clustering had minimal effect, using original data for HDBSCAN")
            working_contig_names = contig_names
            working_embeddings_df = embeddings_df
            precluster_success = False
    else:
        logger.debug("Pre-clustering disabled")
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
        cluster_selection_epsilon=0.5,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )

    cluster_labels = clusterer.fit_predict(norm_data)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for label in cluster_labels if label == -1)
    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0]) if n_clusters > 0 else []
    logger.info(f"HDBSCAN result: {n_clusters} clusters, {n_noise} noise points, sizes: {cluster_sizes.tolist() if hasattr(cluster_sizes, 'tolist') else list(cluster_sizes)}")
    
    formatted_labels = [
        f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
    ]

    # Create clusters dataframe with original contig names (without .original suffix)
    if precluster_success and 'working_contig_names' in locals() and working_contig_names != contig_names:
        # Pre-clustering was used, need to map back to all original contigs
        logger.debug("Mapping pre-clustered results back to all contigs")
        
        # Create mapping for filtered contigs
        working_original_names = [extract_base_contig_name(name) for name in working_contig_names]
        filtered_clusters_df = pd.DataFrame(
            {"contig": working_original_names, "cluster": formatted_labels}
        )
        
        # Create full mapping including contigs that were filtered out as noise
        all_original_names = [extract_base_contig_name(name) for name in embeddings_df.index]
        all_clusters = []
        
        for original_name in all_original_names:
            cluster_assignment = filtered_clusters_df[
                filtered_clusters_df["contig"] == original_name
            ]["cluster"]
            if not cluster_assignment.empty:
                all_clusters.append(cluster_assignment.iloc[0])
            else:
                all_clusters.append("noise")  # Contigs removed by pre-clustering
        
        contig_clusters_df = pd.DataFrame(
            {"contig": all_original_names, "cluster": all_clusters}
        )
        
        removed_by_precluster = sum(1 for cluster in all_clusters if cluster == "noise") - sum(1 for label in formatted_labels if label == "noise")
        if removed_by_precluster > 0:
            logger.debug(f"Additional {removed_by_precluster} contigs marked as noise by pre-clustering")
    else:
        # No pre-clustering or pre-clustering failed, use original approach
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

    # Use UMAP for visualization - use the same data that was actually clustered
    logger.debug("Performing UMAP dimensionality reduction for visualization...")
    
    # Determine which embeddings to use for UMAP based on what was actually clustered
    if precluster_success and 'working_embeddings_df' in locals() and working_embeddings_df is not embeddings_df:
        # Pre-clustering was used and successful, use filtered data for UMAP
        umap_embeddings_df = working_embeddings_df
        logger.debug(f"Using pre-filtered data for UMAP: {umap_embeddings_df.shape[0]} contigs")
    else:
        # No pre-clustering or pre-clustering failed, use original data
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


def cluster_contigs_kmeans_refinement(
    embeddings_df, fragments_dict, args, bin_id, duplication_results
):
    """
    Cluster contigs using K-means where the number of clusters is based on
    the number of duplicated single copy core genes found in the original bin.

    Args:
        embeddings_df: DataFrame with embeddings for contigs
        fragments_dict: Dictionary containing fragment sequences
        args: Command line arguments
        bin_id: Original bin ID being refined
        duplication_results: Results from core gene duplication analysis

    Returns:
        DataFrame with cluster assignments
    """
    logger.info(f"Performing K-means clustering for {bin_id} refinement...")

    # Get the number of duplicated core genes for this bin
    if bin_id in duplication_results:
        duplicated_genes_count = len(duplication_results[bin_id]["duplicated_genes"])
        total_genes_found = duplication_results[bin_id]["total_genes_found"]
        logger.info(
            f"Bin {bin_id} has {duplicated_genes_count} duplicated core genes out of {total_genes_found} total genes"
        )
    else:
        logger.warning(f"No duplication results found for {bin_id}, using default k=2")
        duplicated_genes_count = 2

    # Determine number of clusters for K-means
    # Use the number of duplicated core genes as the number of clusters
    # This assumes each duplicated gene represents a different species/strain
    n_clusters = max(2, duplicated_genes_count)  # Minimum of 2 clusters

    # If we have more contigs than duplicated genes, cap the clusters
    n_contigs = len(embeddings_df)
    if n_contigs < n_clusters:
        logger.warning(
            f"Number of contigs ({n_contigs}) is less than number of duplicated genes ({duplicated_genes_count}), reducing clusters to {n_contigs}"
        )
        n_clusters = max(2, n_contigs - 1)  # Keep at least 2 clusters if possible

    logger.info(f"Using K-means with {n_clusters} clusters for {bin_id} refinement")

    # Normalize the embeddings data for clustering
    logger.debug("Normalizing embeddings for clustering...")
    norm_data = normalize(embeddings_df.values, norm="l2")

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(norm_data)

    # Format cluster labels
    formatted_labels = [f"bin_{label}" for label in cluster_labels]

    # Create clusters dataframe with original contig names (without fragment suffixes)
    original_contig_names = [
        extract_base_contig_name(name) for name in embeddings_df.index
    ]
    contig_clusters_df = pd.DataFrame(
        {"contig": original_contig_names, "cluster": formatted_labels}
    )

    # Count number of clusters and contigs per cluster
    n_clusters_found = len(contig_clusters_df["cluster"].unique())
    logger.info(f"K-means refinement found {n_clusters_found} clusters")

    # Count contigs per cluster
    logger.debug("Counting contigs per cluster...")
    cluster_contig_counts = {}
    for _, row in contig_clusters_df.iterrows():
        cluster_id = row["cluster"]
        contig_name = row["contig"]

        if cluster_id not in cluster_contig_counts:
            cluster_contig_counts[cluster_id] = set()
        cluster_contig_counts[cluster_id].add(contig_name)

    logger.debug("Contigs per cluster:")
    for cluster_id, original_contigs in cluster_contig_counts.items():
        count = len(original_contigs)
        logger.debug(f"  Cluster {cluster_id}: {count} contigs")

    return contig_clusters_df
