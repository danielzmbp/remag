"""
Refinement module for REMAG (k-means based).
"""

import os
import json
import numpy as np
import pandas as pd
from loguru import logger

from .clustering import _construct_knn_graph, _leiden_clustering_on_graph
from .miniprot_utils import (
    check_core_gene_duplications,
    check_core_gene_duplications_from_cache,
    get_core_gene_duplication_results_path,
    get_gene_mappings_cache_path,
)


def _log_and_return(original_clusters_df, bin_id):
    logger.info(f"Bin {bin_id} refinement could not improve the split; keeping original bin")
    return original_clusters_df


def refine_bin(
    clusters_df,
    embeddings_df,
    bin_id,
    gene_mappings_cache,
    duplication_results,
    args,
):
    bin_mask = clusters_df["cluster"].isin([bin_id])
    contigs = clusters_df.loc[bin_mask, "contig"].tolist()
    if len(contigs) < 2:
        logger.warning(f"Bin {bin_id} has insufficient contigs to refine")
        return None

    available = [c for c in contigs if c in embeddings_df.index]
    if len(available) < 2:
        logger.warning(f"Bin {bin_id} lacks embeddings for refinement")
        return None

    # Calculate original SCG count (total gene instances) to use as baseline
    original_scg_count = 0
    for contig in available:
        genes = gene_mappings_cache.get(contig, {})
        original_scg_count += len(genes)

    # Extract embeddings for this bin
    emb = embeddings_df.loc[available].values.astype(np.float32)
    # Embeddings should already be normalized from the model/CSV, but re-normalizing is safe
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized = emb / norm

    original_dup = len(duplication_results.get(bin_id, {}).get("duplicated_genes", {}))
    if original_dup == 0:
        return None

    # Graph-based Refinement (Leiden)
    # 1. Construct local k-NN graph
    # Use a slightly smaller k for refinement to detect finer structures, but respect user args
    default_k = getattr(args, 'leiden_k_neighbors', 15)
    k = min(default_k, len(available) - 1)
    
    if k < 1:
        return None

    # We construct the graph once
    graph = _construct_knn_graph(
        normalized, 
        k=k, 
        similarity_threshold=getattr(args, 'leiden_similarity_threshold', 0.1),
        n_jobs=1, # Small task, keep single threaded
        args=None # Don't save intermediate files for sub-tasks
    )

    try:
        n_components = len(graph.components())
        logger.debug(f"Bin {bin_id} refinement graph has {n_components} connected components")
    except Exception as e:
        logger.debug(f"Could not count graph components: {e}")

    best = None
    
    # 2. Try increasing resolutions to break up the cluster, using the same set as initial Leiden
    resolutions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.2, 1.5, 2.0]
    
    for res in resolutions:
        labels = _leiden_clustering_on_graph(graph, resolution=res, random_state=42)
        
        # Skip if it didn't split anything (all one cluster or all noise)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            continue
            
        # Score the split
        total_dup, retained_scg = _score_split(labels, available, gene_mappings_cache)

        # Determine retention threshold based on bin size
        # For very large bins (>10k contigs), we relax the threshold to allow disentangling 
        # complex mixtures (e.g. extracting 1 genome from a pool of many, which results in ~10% retention)
        retention_threshold = 0.75
        if len(contigs) > 10000:
            retention_threshold = 0.10
            
        # Skip if the best sub-bin has less than the threshold of the original SCGs (oversplitting)
        if original_scg_count > 0 and retained_scg < retention_threshold * original_scg_count:
            logger.debug(f"Bin {bin_id} res={res} skipped: retained SCG {retained_scg} < {int(retention_threshold*100)}% of original {original_scg_count}")
            continue
        
        dup_scg_ratio = (total_dup / retained_scg * 100) if retained_scg > 0 else 0.0
        logger.debug(
            f"Bin {bin_id} Leiden res={res}: total_dup={total_dup}, retained_scg={retained_scg} ({dup_scg_ratio:.1f}%), sub_bins={len(unique_labels)}"
        )

        # We want to reduce duplications
        # If duplications are equal, we prefer retaining more SCGs
        if total_dup < original_dup:
            score = (total_dup, -retained_scg)
            if best is None or score < (best[0], best[1]):
                best = (total_dup, -retained_scg, labels.copy(), res)

    if best is None:
        logger.info(f"Bin {bin_id} refinement did not reduce duplicated genes")
        return None

    best_dup, _, best_labels, best_res = best
    
    logger.debug(f"Bin {bin_id} resolved {original_dup - best_dup} duplications (from {original_dup} to {best_dup})")
    
    # Format new labels
    refined = pd.DataFrame(
        {
            "contig": available,
            "cluster": [f"{bin_id}_{label}" if label != -1 else f"{bin_id}_noise" for label in best_labels],
        }
    )
    
    # Filter out noise from refinement
    refined = refined[~refined["cluster"].str.endswith("_noise")]
    
    logger.info(f"Bin {bin_id} refinement succeeded (Leiden res={best_res}): split into {refined['cluster'].nunique()} sub-bins")
    return refined


def _score_split(labels, contig_names, gene_mappings_cache):
    clusters = {}
    for contig, label in zip(contig_names, labels):
        if label == -1: continue # Skip noise
        clusters.setdefault(label, []).append(contig)

    total_dup = 0
    scg_retained = 0
    for contigs in clusters.values():
        gene_counts = {}
        scg_count = 0
        for contig in contigs:
            genes = gene_mappings_cache.get(contig, {})
            scg_count += len(genes)
            for g in genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1
        
        # For this sub-bin, track max SCGs (as a proxy for the main genome quality)
        # Ideally we want one big clean bin, not 10 tiny clean bins
        scg_retained = max(scg_retained, scg_count)
        total_dup += sum(1 for cnt in gene_counts.values() if cnt > 1)

    return total_dup, scg_retained


def refine_contaminated_bins(
    clusters_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=1
):
    logger.info("Refining contaminated bins (Leiden sub-clustering)...")

    # Load embeddings once
    embeddings_path = os.path.join(args.output, "embeddings.csv")
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings not found at {embeddings_path}")
        return clusters_df, fragments_dict, {}
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    # Ensure index is string to match clusters_df contig names
    embeddings_df.index = embeddings_df.index.astype(str)

    duplication_results = args._duplication_results if hasattr(args, "_duplication_results") else {}
    gene_mappings_cache = getattr(args, "_gene_mappings_cache", None)
    if gene_mappings_cache is None:
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                gene_mappings_cache = json.load(f)

    if not duplication_results:
        results_path = get_core_gene_duplication_results_path(args)
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                duplication_results = json.load(f)

    if not duplication_results:
        logger.warning("No duplication data available; skipping refinement")
        return clusters_df, fragments_dict, {}

    contaminated_bins = []
    
    is_single_cell = getattr(args, "mode", "metagenomics").lower() == "single-cell"
    
    for bin_id, info in duplication_results.items():
        dups_count = len(info.get("duplicated_genes", {}))
        
        # Check minimum duplications threshold
        if dups_count < getattr(args, "min_duplications_for_refinement", 1):
            continue
            
        # For single-cell mode: only refine if duplications are significant relative to SCGs
        if is_single_cell:
            scg_count = info.get("single_copy_genes_count", 0)
            if scg_count > 0:
                dup_ratio = dups_count / scg_count
                if dup_ratio < 0.05:
                    logger.debug(
                        f"Skipping refinement for bin {bin_id} in single-cell mode: "
                        f"{dups_count} dups / {scg_count} SCGs = {dup_ratio:.1%} < 5%"
                    )
                    continue
        
        contaminated_bins.append(bin_id)

    if not contaminated_bins:
        logger.info("No contaminated bins found; refinement skipped")
        return clusters_df, fragments_dict, {}

    refinement_summary = {}
    new_bins_dfs = []
    processed_bins = set()

    for bin_id in contaminated_bins:
        # Log duplication status before refinement
        info = duplication_results.get(bin_id, {})
        n_dups = len(info.get("duplicated_genes", {}))
        n_scgs = info.get("single_copy_genes_count", 0)
        dup_scg_ratio = (n_dups / n_scgs * 100) if n_scgs > 0 else 0.0
        logger.info(f"Refining bin {bin_id}: {n_dups} duplicated genes, {n_scgs} single-copy genes ({dup_scg_ratio:.1f}%)")

        refined_df = refine_bin(
            clusters_df,
            embeddings_df,
            bin_id,
            gene_mappings_cache,
            duplication_results,
            args, # Passed args
        )
        if refined_df is None:
            refinement_summary[bin_id] = {
                "status": "failed",
                "reason": "no_improvement",
                "sub_bins": 0,
            }
            continue

        processed_bins.add(bin_id)
        new_bins_dfs.append(refined_df)
        refinement_summary[bin_id] = {
            "status": "success",
            "reason": "leiden_split",
            "sub_bins": refined_df["cluster"].nunique(),
        }

    # Update clusters_df only if any bins were successfully refined
    if new_bins_dfs:
        # Remove old versions of refined bins
        clusters_df = clusters_df[~clusters_df["cluster"].isin(processed_bins)]
        # Add new refined bins
        refined_bins_concat = pd.concat(new_bins_dfs, ignore_index=True)
        clusters_df = pd.concat([clusters_df, refined_bins_concat], ignore_index=True)

    return clusters_df, fragments_dict, refinement_summary
