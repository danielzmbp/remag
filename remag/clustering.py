"""
Clustering module for REMAG
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  # Only import needed for k-NN graph construction
from loguru import logger
import os
import json
import igraph as ig
import leidenalg

from .utils import extract_base_contig_name, get_torch_device, group_contigs_by_cluster
# Removed torch import - device detection handled in utils


class GraphManager:
    """Handles k-NN graph construction and caching."""
    
    def __init__(self, k=15, similarity_threshold=0.1, n_jobs=-1):
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.n_jobs = n_jobs
    
    def construct_graph(self, embeddings, args=None):
        """Construct k-NN graph from embeddings."""
        return _construct_knn_graph(
            embeddings, self.k, self.similarity_threshold, args, self.n_jobs
        )


class ClusteringManager:
    """Main clustering orchestrator."""
    
    def __init__(self, args):
        self.args = args
        self.graph_manager = GraphManager(
            k=getattr(args, 'leiden_k_neighbors', 15),
            similarity_threshold=getattr(args, 'leiden_similarity_threshold', 0.1)
        )
    
    def load_eukaryotic_scores(self):
        """Load eukaryotic classification scores."""
        eukaryotic_scores = {}
        from .features import get_classification_results_path
        
        classification_results_path = get_classification_results_path(self.args.fasta, self.args.output)
        
        if os.path.exists(classification_results_path):
            try:
                classification_df = pd.read_csv(classification_results_path, sep='\t')
                eukaryotic_scores = dict(zip(classification_df['header'], classification_df['eukar_score']))
                logger.info(f"Loaded eukaryotic scores for {len(eukaryotic_scores)} contigs")
            except Exception as e:
                logger.warning(f"Could not load classification results: {e}")
        else:
            logger.warning(f"Eukaryotic classification file not found: {classification_results_path}")
        
        return eukaryotic_scores



# Removed k-means pre-filtering - simplified clustering approach


def _construct_knn_graph(embeddings, k=15, similarity_threshold=0.1, n_jobs=1, args=None):
    """
    Construct a k-NN graph from multidimensional embeddings using cosine similarity.
    Optimized for memory efficiency and parallelization.
    
    Args:
        embeddings: Numpy array of L2-normalized embeddings (n_samples x embedding_dim)
        k: Number of nearest neighbors for each node
        similarity_threshold: Minimum cosine similarity to create an edge (0-1)
        n_jobs: Number of parallel jobs for k-NN search
        args: Arguments object containing output directory and keep_intermediate flag
        
    Returns:
        igraph.Graph: Weighted graph with cosine similarity weights
    """
    # Check if graph already exists and can be loaded
    if args and args.output:
        edge_list_path = os.path.join(args.output, "knn_graph_edges.csv")
        graph_stats_path = os.path.join(args.output, "knn_graph_stats.json")
        
        if os.path.exists(edge_list_path) and os.path.exists(graph_stats_path):
            try:
                # Load graph statistics to verify compatibility
                with open(graph_stats_path, 'r') as f:
                    saved_stats = json.load(f)
                
                # Check if parameters match
                if (saved_stats.get('n_vertices') == len(embeddings) and
                    saved_stats.get('k') == k and
                    saved_stats.get('similarity_threshold') == similarity_threshold):
                    
                    logger.info(f"Loading existing k-NN graph from {edge_list_path}")
                    
                    # Load edge list
                    edges = []
                    weights = []
                    with open(edge_list_path, 'r') as f:
                        for line in f:
                            if line.startswith('#') or line.startswith('source'):
                                continue  # Skip comments and header
                            source, target, weight = line.strip().split(',')
                            edges.append((int(source), int(target)))
                            weights.append(float(weight))
                    
                    # Reconstruct graph
                    g = ig.Graph()
                    g.add_vertices(len(embeddings))
                    g.add_edges(edges)
                    g.es['weight'] = weights
                    
                    logger.info(f"Successfully loaded k-NN graph: {g.vcount()} nodes, {g.ecount()} edges")
                    return g
                else:
                    logger.info("Existing k-NN graph has different parameters, constructing new graph")
                    logger.debug(f"Saved: vertices={saved_stats.get('n_vertices')}, k={saved_stats.get('k')}, "
                               f"threshold={saved_stats.get('similarity_threshold')}")
                    logger.debug(f"Current: vertices={len(embeddings)}, k={k}, threshold={similarity_threshold}")
            except Exception as e:
                logger.warning(f"Could not load existing k-NN graph: {e}")
                logger.info("Constructing new k-NN graph")
    
    logger.info(f"Constructing k-NN graph from {len(embeddings)} embeddings (k={k}, n_jobs={n_jobs})")
    
    # Use sklearn's NearestNeighbors for efficient, parallelized k-NN search
    # Since embeddings are L2-normalized, cosine similarity = dot product
    nbrs = NearestNeighbors(
        n_neighbors=k+1,  # +1 because it includes self
        metric='cosine',
        algorithm='brute',  # brute force is often fastest for high-dimensional data
        n_jobs=n_jobs
    )
    nbrs.fit(embeddings)
    
    # Find k-NN for all points efficiently
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Convert distances to similarities (cosine distance = 1 - cosine similarity)
    similarities = 1 - distances
    
    # Build edge list efficiently
    edges = []
    weights = []
    
    for i in range(len(embeddings)):
        # Skip self (first neighbor) and apply similarity threshold
        for j in range(1, k+1):  # Skip index 0 (self)
            neighbor_idx = indices[i, j]
            similarity = similarities[i, j]
            
            if similarity >= similarity_threshold:
                edges.append((i, neighbor_idx))
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
    
    # Save graph if keep_intermediate is enabled
    if args and getattr(args, "keep_intermediate", False):
        # Save as edge list with weights
        edge_list_path = os.path.join(args.output, "knn_graph_edges.csv")
        with open(edge_list_path, 'w') as f:
            f.write("# Node IDs correspond to row indices in embeddings.csv\n")
            f.write("source,target,weight\n")
            for edge in g.es:
                source = edge.source
                target = edge.target
                weight = edge['weight']
                f.write(f"{source},{target},{weight:.6f}\n")
        logger.info(f"Saved k-NN graph edge list to {edge_list_path}")
        
        # Also save graph statistics
        graph_stats = {
            "n_vertices": g.vcount(),
            "n_edges": g.ecount(),
            "k": k,
            "similarity_threshold": similarity_threshold,
            "density": g.density(),
            "n_connected_components": len(g.connected_components()),
            "average_degree": np.mean(g.degree()),
            "max_degree": max(g.degree()),
            "min_degree": min(g.degree())
        }
        
        graph_stats_path = os.path.join(args.output, "knn_graph_stats.json")
        with open(graph_stats_path, 'w') as f:
            json.dump(graph_stats, f, indent=2)
        logger.info(f"Saved k-NN graph statistics to {graph_stats_path}")
    
    return g


def _leiden_clustering(embeddings, k=15, similarity_threshold=0.1, resolution=1.0, random_state=42, n_jobs=1, args=None):
    """
    Perform Leiden clustering on embeddings by first constructing a k-NN graph.
    
    Args:
        embeddings: Numpy array of L2-normalized embeddings (n_samples x embedding_dim)
        k: Number of nearest neighbors for graph construction
        similarity_threshold: Minimum cosine similarity to create an edge
        resolution: Resolution parameter for Leiden algorithm (higher = more clusters)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs for k-NN graph construction
        args: Arguments object containing output directory and keep_intermediate flag
        
    Returns:
        numpy.array: Cluster labels (-1 for isolated nodes, 0+ for clusters)
    """
    logger.info(f"Starting Leiden clustering with k={k}, resolution={resolution}, n_jobs={n_jobs}")
    
    # Construct k-NN graph with parallelization
    graph = _construct_knn_graph(embeddings, k=k, similarity_threshold=similarity_threshold, n_jobs=n_jobs, args=args)
    
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


def cluster_contigs(embeddings_df, fragments_dict, args):
    """Main clustering function that orchestrates the clustering process."""
    bins_path = os.path.join(args.output, "bins.csv")

    # Check if bins file already exists
    if os.path.exists(bins_path):
        logger.info(f"Loading existing bins from {bins_path}")
        return pd.read_csv(bins_path)

    # Initialize clustering manager
    clustering_manager = ClusteringManager(args)
    
    # Load eukaryotic classification scores for logging purposes
    eukaryotic_scores = clustering_manager.load_eukaryotic_scores()

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
    
    # Use Leiden clustering directly
    logger.info("Using Leiden clustering")
    leiden_resolution = getattr(args, 'leiden_resolution', 1.0)
    
    logger.info(f"Running Leiden on {len(contig_names)} contigs "
               f"(resolution={leiden_resolution}, k={clustering_manager.graph_manager.k}, "
               f"similarity_threshold={clustering_manager.graph_manager.similarity_threshold})")
    
    cluster_labels = _leiden_clustering(
        norm_data,
        k=clustering_manager.graph_manager.k,
        similarity_threshold=clustering_manager.graph_manager.similarity_threshold,
        resolution=leiden_resolution,
        random_state=42,
        n_jobs=getattr(args, 'cores', 1),
        args=args
    )
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for label in cluster_labels if label == -1)
    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0]) if n_clusters > 0 else []
    formatted_labels = [
        f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
    ]

    # Create clusters dataframe with original contig names (without .original suffix)
    final_original_contig_names = [
        extract_base_contig_name(name) for name in embeddings_df.index
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
            random_state=42,
            n_jobs=getattr(args, 'cores', 1),
            args=args
        )
        
        n_recluster_clusters = len(set(recluster_labels)) - (1 if -1 in recluster_labels else 0)
        n_recluster_noise = sum(1 for label in recluster_labels if label == -1)
        recluster_sizes = np.bincount(recluster_labels[recluster_labels >= 0]) if n_recluster_clusters > 0 else []
        logger.info(f"Reclustering result: {n_recluster_clusters} clusters, {n_recluster_noise} noise points, sizes: {recluster_sizes.tolist() if hasattr(recluster_sizes, 'tolist') else list(recluster_sizes)}")
        
        # Only use reclustering results if we got more than one cluster
        if n_recluster_clusters > 1:
            logger.info(f"Reclustering successful: {n_recluster_clusters} clusters found. Using reclustering results.")
            
            # Update cluster labels with reclustering results
            cluster_labels = recluster_labels
            
            
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
    
    # Save final bins (excluding noise) - keep only first two columns
    final_bins_df = final_bins_df[["contig", "cluster"]]
    final_bins_df.to_csv(bins_path, index=False)

    # Count contigs per cluster using utility function
    logger.debug("Counting contigs per cluster...")
    cluster_contig_counts = group_contigs_by_cluster(contig_clusters_df)



    # Note about visualization
    if getattr(args, "keep_intermediate", False):
        logger.info("Embeddings saved to embeddings.csv. Use scripts/plot_features.py for UMAP visualization with plotting dependencies.")

    # Perform chimera detection for large contigs
    if not getattr(args, 'skip_chimera_detection', False):
        logger.info("Running chimera detection on large contigs...")
        chimera_results = detect_chimeric_contigs(embeddings_df, clusters_df, args)

    logger.info(f"Saved contig-level clusters to {bins_path}")

    return clusters_df
