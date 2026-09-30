"""
Clustering module for REMAG
"""

import json
import os

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import NearestNeighbors

from .output import (
    clustering_identity,
    embedding_identity,
    file_sha256,
    incompatible_cache,
    validate_cached_binning_settings,
)
from .utils import contig_lengths_for_embeddings


def _calculate_bin_quality(contig_names, gene_mappings):
    """
    Calculate quality score using F1-score inspired by SemiBin2.

    Formulas:
        completeness = N / 133
        contamination = (G - N) / G
        F1-score = (2 * completeness * (1 - contamination)) / (completeness + (1 - contamination))

    Where:
        N = Number of nonredundant genes (unique gene families)
        G = Total genes found (sum of all gene counts)
        133 = Total number of single-copy genes (user specified)

    Args:
        contig_names: List of contig names in the bin
        gene_mappings: Dict mapping contig_name -> gene_family -> details

    Returns:
        tuple: (score, scg_count, dup_count)
    """
    gene_counts = {}
    for c in contig_names:
        # Check if contig has gene mappings
        if c in gene_mappings:
            for gene in gene_mappings[c]:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1

    if not gene_counts:
        # F1 score is 0.0 for empty bins
        return 0.0, 0, 0

    scg = sum(1 for v in gene_counts.values() if v == 1)
    dups = sum(1 for v in gene_counts.values() if v > 1)

    # N: nonredundant genes (unique gene families)
    N = len(gene_counts)

    # G: total genes found
    G = sum(gene_counts.values())

    # Calculate metrics based on provided formulas
    completeness = float(N) / 133.0

    if G > 0:
        contamination = float(G - N) / float(G)
    else:
        contamination = 0.0

    precision_term = 1.0 - contamination

    # Calculate F1-score (harmonic mean)
    if (completeness + precision_term) > 0:
        score = (2.0 * completeness * precision_term) / (completeness + precision_term)
    else:
        score = 0.0

    return score, scg, dups


def _greedy_leiden_clustering(
    embeddings,
    contig_names,
    gene_mappings,
    k=15,
    similarity_threshold=0.1,
    resolutions=[0.1, 0.5, 1.0, 2.0, 5.0],
    max_contamination=0.10,
    random_state=42,
    n_jobs=1,
    args=None,
):
    """
    Perform greedy Leiden clustering.

    Iteratively:
    1. Cluster active graph at multiple resolutions.
    2. Pick best cluster based on quality score (F1-score of completion and contamination).
    3. Remove best cluster nodes and repeat.

    Args:
        embeddings: Numpy array of L2-normalized embeddings
        contig_names: List of contig names corresponding to embeddings
        gene_mappings: Dict of gene mappings for quality calculation
        k: k for KNN graph
        similarity_threshold: threshold for edges
        resolutions: list of resolutions to try
        max_contamination: maximum contamination allowed (dups / 133)
        random_state: random seed
        n_jobs: number of cores
        args: args object

    Returns:
        list: Cluster labels matching embeddings order (-1 for noise/unbinned)
    """
    logger.info(
        f"Starting Greedy Leiden clustering with k={k}, resolutions={resolutions}, max_contamination={max_contamination}"
    )

    # 1. Construct initial graph
    graph = _construct_knn_graph(
        embeddings,
        contig_names=contig_names,
        k=k,
        similarity_threshold=similarity_threshold,
        n_jobs=n_jobs,
        args=args,
    )

    # Add names to graph vertices for easy retrieval
    graph.vs["name"] = contig_names

    # Keep track of original indices to assign final labels correctly
    n_samples = len(embeddings)
    graph.vs["original_index"] = range(n_samples)

    cluster_labels = np.full(n_samples, -1, dtype=int)

    # Current active indices in the ORIGINAL embeddings array
    active_indices = list(range(n_samples))

    bin_counter = 0
    iteration = 0

    while len(active_indices) > 0:
        iteration += 1

        # Stop if very few nodes left
        if len(active_indices) < 2:
            logger.info("Fewer than 2 nodes remaining, stopping greedy loop.")
            break

        # Extract subgraph for active nodes
        if iteration == 1:
            current_graph = graph
        else:
            current_graph = graph.induced_subgraph(active_indices)

        # If no edges, can't cluster effectively
        if current_graph.ecount() == 0:
            logger.info(f"Iter {iteration}: No edges in remaining graph. Stopping.")
            break

        best_candidate = None  # (score, scg, dups, resolution, node_indices_subgraph)
        best_score = -float("inf")

        # Try all resolutions
        for res in resolutions:
            partition = leidenalg.find_partition(
                current_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=random_state,
            )

            # Group nodes by cluster membership
            clusters = {}
            for i, cluster_id in enumerate(partition.membership):
                clusters.setdefault(cluster_id, []).append(i)

            # Evaluate each cluster
            for cid, node_indices_in_subgraph in clusters.items():
                # Get contig names
                c_names = [
                    current_graph.vs[ix]["name"] for ix in node_indices_in_subgraph
                ]

                # Basic size filter (min 2 contigs)
                if len(c_names) < 2:
                    continue

                score, scg, dups = _calculate_bin_quality(c_names, gene_mappings)

                # Check contamination constraint (dups < 10% of 133)
                if (float(dups) / 133.0) > max_contamination:
                    continue

                # Maximize score
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "score": score,
                        "scg": scg,
                        "dups": dups,
                        "res": res,
                        "node_indices_subgraph": node_indices_in_subgraph,
                        "contigs": c_names,
                    }

        # Check if best candidate meets criteria
        if best_candidate:
            # We found a valid bin
            bin_id = bin_counter
            bin_counter += 1

            subgraph_indices = best_candidate["node_indices_subgraph"]
            # Map back to original indices
            original_indices = [
                current_graph.vs[ix]["original_index"] for ix in subgraph_indices
            ]

            # Assign label
            cluster_labels[original_indices] = bin_id

            logger.debug(
                f"Iter {iteration}: Picked bin_{bin_id} (res={best_candidate['res']:.1f}, "
                f"scg={best_candidate['scg']}, dups={best_candidate['dups']}, "
                f"score={best_candidate['score']:.1f}, size={len(original_indices)})"
            )

            # Update active indices (remove binned nodes)
            to_remove = set(original_indices)
            active_indices = [idx for idx in active_indices if idx not in to_remove]

        else:
            logger.info(f"Iter {iteration}: No valid candidate found. Stopping.")
            break

        # Log progress occasionally
        if iteration % 10 == 0:
            logger.info(
                f"Greedy clustering: {len(active_indices)} contigs remaining..."
            )

    n_clustered = np.sum(cluster_labels >= 0)
    logger.info(
        f"Greedy clustering complete. {bin_counter} bins formed, {n_clustered} contigs binned out of {n_samples}."
    )

    return cluster_labels


def _construct_knn_graph(
    embeddings, k=15, similarity_threshold=0.1, n_jobs=1, args=None, contig_names=None
):
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
    # Handle edge cases
    n_samples = len(embeddings)
    if n_samples == 0:
        # Empty embeddings - return empty graph
        return ig.Graph()

    if n_samples == 1:
        # Single node - return graph with one node and no edges
        graph = ig.Graph(n=1)
        return graph

    # Adjust k if we have fewer samples than k+1
    if n_samples <= k:
        original_k = k
        k = n_samples - 1
        logger.warning(
            f"Adjusted k from {original_k} to {k} due to limited samples ({n_samples})"
        )

    graph_identity = embedding_identity(
        embeddings, list(range(n_samples)) if contig_names is None else contig_names
    )
    # Check if graph already exists and can be loaded
    if args and args.output:
        edge_list_path = os.path.join(args.output, "knn_graph_edges.csv")
        graph_stats_path = os.path.join(args.output, "knn_graph_stats.json")

        if os.path.exists(edge_list_path) != os.path.exists(graph_stats_path):
            raise incompatible_cache("incomplete graph cache")
        if os.path.exists(edge_list_path) and os.path.exists(graph_stats_path):
            try:
                # Load graph statistics to verify compatibility
                with open(graph_stats_path, "r") as f:
                    saved_stats = json.load(f)

                # Check if parameters match
                if (
                    saved_stats.get("n_vertices") == len(embeddings)
                    and saved_stats.get("k") == k
                    and saved_stats.get("similarity_threshold") == similarity_threshold
                    and saved_stats.get("ordered_embeddings_sha256") == graph_identity
                    and saved_stats.get("edges_sha256") == file_sha256(edge_list_path)
                ):
                    logger.info(f"Loading existing k-NN graph from {edge_list_path}")

                    # Load edge list
                    edges = []
                    weights = []
                    with open(edge_list_path, "r") as f:
                        for line in f:
                            if line.startswith("#") or line.startswith("source"):
                                continue  # Skip comments and header
                            source, target, weight = line.strip().split(",")
                            edges.append((int(source), int(target)))
                            weights.append(float(weight))

                    # Reconstruct graph
                    g = ig.Graph()
                    g.add_vertices(len(embeddings))
                    g.add_edges(edges)
                    g.es["weight"] = weights

                    logger.info(
                        f"Successfully loaded k-NN graph: {g.vcount()} nodes, {g.ecount()} edges"
                    )
                    return g
                raise incompatible_cache(
                    "graph vertices, values, order or parameters changed"
                )
            except (OSError, ValueError, KeyError) as error:
                raise incompatible_cache(
                    "graph cache is missing, changed or unreadable"
                ) from error

    logger.info(
        f"Constructing k-NN graph from {len(embeddings)} embeddings (k={k}, n_jobs={n_jobs})"
    )

    # Use sklearn's NearestNeighbors for efficient, parallelized k-NN search
    # Since embeddings are L2-normalized, cosine similarity = dot product
    nbrs = NearestNeighbors(
        n_neighbors=k or 1,  # Zero-neighbor graphs are sliced below.
        metric="cosine",
        algorithm="brute",  # brute force is often fastest for high-dimensional data
        n_jobs=n_jobs,
    )
    nbrs.fit(embeddings)

    # Query fitted points so sklearn excludes each point by identity.
    distances, indices = nbrs.kneighbors()

    # Convert distances to similarities (cosine distance = 1 - cosine similarity)
    similarities = 1 - distances

    # Build edge list using vectorized operations
    # Keep the requested count, including the zero-neighbor case.
    neighbor_indices = indices[:, :k]
    neighbor_similarities = similarities[:, :k]

    # Create source indices array [0, 0... 1, 1...] matching the shape
    # Use broadcasting/repeating to align with flattened neighbor arrays
    source_indices = np.repeat(np.arange(len(embeddings)), neighbor_indices.shape[1])

    # Flatten arrays
    flat_sources = source_indices
    flat_targets = neighbor_indices.flatten()
    flat_weights = neighbor_similarities.flatten()

    # Apply threshold mask
    mask = flat_weights >= similarity_threshold

    # Create edges and weights
    # igraph expects list of tuples for edges
    valid_sources = flat_sources[mask]
    valid_targets = flat_targets[mask]
    edges = list(zip(valid_sources, valid_targets))
    weights = flat_weights[mask].tolist()

    logger.info(f"Created {len(edges)} edges with similarity >= {similarity_threshold}")

    # Create igraph from edge list
    g = ig.Graph()
    g.add_vertices(len(embeddings))
    g.add_edges(edges)
    g.es["weight"] = weights

    # Save graph if keep_intermediate is enabled
    if args and getattr(args, "keep_intermediate", False):
        # Save as edge list with weights
        edge_list_path = os.path.join(args.output, "knn_graph_edges.csv")
        with open(edge_list_path, "w") as f:
            f.write(
                "# Node IDs correspond to row indices in knn_graph_contigs.csv (the graph subset)\n"
            )
            f.write("source,target,weight\n")
            for edge in g.es:
                source = edge.source
                target = edge.target
                weight = edge["weight"]
                f.write(f"{source},{target},{weight:.6f}\n")
        logger.info(f"Saved k-NN graph edge list to {edge_list_path}")

        # Also save graph statistics
        graph_stats = {
            "ordered_embeddings_sha256": graph_identity,
            "edges_sha256": file_sha256(edge_list_path),
            "n_vertices": g.vcount(),
            "n_edges": g.ecount(),
            "k": k,
            "similarity_threshold": similarity_threshold,
            "density": g.density(),
            "n_connected_components": len(g.connected_components()),
            "average_degree": np.mean(g.degree()),
            "max_degree": max(g.degree()),
            "min_degree": min(g.degree()),
        }

        graph_stats_path = os.path.join(args.output, "knn_graph_stats.json")
        with open(graph_stats_path, "w") as f:
            json.dump(graph_stats, f, indent=2)
        logger.info(f"Saved k-NN graph statistics to {graph_stats_path}")

    return g


def cluster_contigs(embeddings_df, fragments_dict, gene_mappings, args):
    """Main clustering function that orchestrates the clustering process using Greedy Leiden."""
    # Ensure output directory exists for all code paths
    os.makedirs(args.output, exist_ok=True)

    lengths = contig_lengths_for_embeddings(embeddings_df, fragments_dict)
    minimum = getattr(args, "min_contig_length", 1)
    cutoff = getattr(args, "graph_min_contig_length", None) or minimum
    if cutoff < minimum:
        raise ValueError("Graph minimum must be at least the admission minimum.")
    admitted = [c for c in embeddings_df.index if lengths[c] >= minimum]
    embeddings_df = embeddings_df.loc[admitted]
    identity = clustering_identity(embeddings_df, fragments_dict, gene_mappings, args)
    pre_path = os.path.join(args.output, "pre_rescue.csv")
    metadata_path = os.path.join(args.output, "clustering_provenance.json")
    validate_cached_binning_settings(args)
    if os.path.exists(pre_path):
        with open(metadata_path) as handle:
            saved = json.load(handle)
        if any(saved.get(k) != v for k, v in identity.items()) or saved.get(
            "pre_rescue_sha256"
        ) != file_sha256(pre_path):
            raise incompatible_cache(
                "candidate population/order, embeddings, sequences or markers changed"
            )
        clusters = pd.read_csv(
            pre_path, dtype={"contig": str, "cluster": str}, keep_default_na=False
        )
        if clusters["contig"].tolist() != admitted or clusters["cluster"].isna().any():
            raise incompatible_cache("invalid pre-rescue assignments")
        logger.info(
            "Loading compatible pre-rescue assignments; unified rescue will run again."
        )
        return clusters
    if os.path.exists(metadata_path) or os.path.exists(
        os.path.join(args.output, "bins.csv")
    ):
        raise incompatible_cache("missing pre-rescue assignments")

    # Only the graph core enters Leiden; all admitted embeddings share one model.
    contig_names = [c for c in admitted if lengths[c] >= cutoff]
    core_embeddings = embeddings_df.loc[contig_names]
    norm_data = core_embeddings.values
    # Log essential data properties
    logger.info(
        f"Clustering {len(contig_names)} contigs with {embeddings_df.shape[1]}D embeddings"
    )

    # Use Greedy Leiden clustering
    logger.info("Using Greedy Leiden clustering strategy")

    # Get parameters from args or defaults
    greedy_resolutions = getattr(args, "greedy_resolutions", [0.5, 1.0, 2.0, 5.0])
    greedy_max_contamination = getattr(args, "greedy_max_contamination", 0.10)

    cluster_labels = _greedy_leiden_clustering(
        norm_data,
        contig_names=contig_names,
        gene_mappings=gene_mappings,
        k=getattr(args, "leiden_k_neighbors", 15),
        similarity_threshold=getattr(args, "leiden_similarity_threshold", 0.1),
        resolutions=greedy_resolutions,
        max_contamination=greedy_max_contamination,
        random_state=42,
        n_jobs=getattr(args, "cores", 1),
        args=args,
    )

    formatted_labels = [
        f"bin_{label}" if label != -1 else "noise" for label in cluster_labels
    ]

    labels = dict(zip(contig_names, formatted_labels))
    contig_clusters_df = pd.DataFrame(
        {
            "contig": admitted,
            "cluster": [labels.get(c, "noise") for c in admitted],
        }
    )
    clusters_df = contig_clusters_df
    # Count and report final results
    final_counts = contig_clusters_df["cluster"].value_counts().to_dict()
    n_noise = final_counts.get("noise", 0)

    # Filter out singleton bins for reporting noise-free sizes
    singleton_bins = {k: v for k, v in final_counts.items() if k != "noise" and v == 1}
    filtered_counts = {k: v for k, v in final_counts.items() if k == "noise" or v > 1}
    n_clusters = len([k for k in filtered_counts.keys() if k != "noise"])

    logger.info(
        f"Clustering complete: {n_clusters} clusters "
        f"(excluding {len(singleton_bins)} singletons), {n_noise} noise contigs, "
        f"sizes: {dict(sorted(filtered_counts.items()))}"
    )

    # Keep noise and sub-threshold seeds: the size filter runs only after rescue.
    clusters_df.to_csv(pre_path, index=False)
    identity["pre_rescue_sha256"] = file_sha256(pre_path)
    with open(metadata_path, "w") as handle:
        json.dump(identity, handle, indent=2)
    if getattr(args, "keep_intermediate", False):
        pd.DataFrame({"contig": contig_names}).to_csv(
            os.path.join(args.output, "knn_graph_contigs.csv"), index=False
        )
    logger.info(f"Saved initial assignments to {pre_path}")
    return clusters_df
