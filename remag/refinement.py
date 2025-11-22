"""
Refinement module for REMAG (k-means based).
"""

import os
import json
import numpy as np
import pandas as pd
from loguru import logger

from .miniprot_utils import (
    check_core_gene_duplications,
    check_core_gene_duplications_from_cache,
    get_core_gene_duplication_results_path,
    get_gene_mappings_cache_path,
)


def _log_and_return(original_clusters_df, bin_id):
    logger.info(f"Bin {bin_id} k-means refinement could not improve the split; keeping original bin")
    return original_clusters_df


def refine_bin(
    clusters_df,
    embeddings_df,
    bin_id,
    gene_mappings_cache,
    duplication_results,
    max_k=6,
    restarts=3,
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

    emb = embeddings_df.loc[available].values.astype(np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized = emb / norm

    original_dup = len(duplication_results.get(bin_id, {}).get("duplicated_genes", {}))
    if original_dup == 0:
        return None

    best = None
    rng_seed = 42

    for k in range(2, min(max_k + 1, len(normalized))):
        for restart in range(restarts):
            rng = np.random.default_rng(rng_seed + k * 100 + restart)
            labels = _run_kmeans(normalized, k, rng)
            if labels is None:
                continue

            total_dup, retained_scg = _score_split(labels, available, gene_mappings_cache)
            logger.debug(
                f"Bin {bin_id} k-means k={k} restart={restart}: total_dup={total_dup}, retained_scg={retained_scg}"
            )

            if total_dup < original_dup:
                score = (total_dup, -retained_scg)
                if best is None or score < (best[0], best[1]):
                    best = (total_dup, -retained_scg, labels.copy(), k)

    if best is None:
        logger.info(f"Bin {bin_id} k-means refinement did not reduce duplicated genes")
        return None

    _, _, best_labels, best_k = best
    refined = pd.DataFrame(
        {
            "contig": available,
            "cluster": [f"{bin_id}_{label}" for label in best_labels],
        }
    )
    logger.info(f"Bin {bin_id} k-means refinement succeeded: k={best_k}, sub-bins={len(np.unique(best_labels))}")
    return refined


def _run_kmeans(data, k, rng, max_iter=20):
    n = len(data)
    if k >= n:
        return None

    centers_idx = [rng.integers(n)]
    centers = [data[centers_idx[0]]]
    distances = np.full(n, np.inf, dtype=np.float32)

    for _ in range(1, k):
        distances = np.minimum(distances, np.sum((data - centers[-1]) ** 2, axis=1))
        total = distances.sum()
        if total == 0 or not np.isfinite(total):
            idx = rng.integers(n)
        else:
            probs = distances / total
            idx = rng.choice(n, p=probs)
        centers.append(data[idx])

    centers = np.stack(centers)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centers[idx] = data[mask].mean(axis=0)

    if len(np.unique(labels)) < 2:
        return None
    return labels


def _score_split(labels, contig_names, gene_mappings_cache):
    clusters = {}
    for contig, label in zip(contig_names, labels):
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
        scg_retained = max(scg_retained, scg_count)
        total_dup += sum(1 for cnt in gene_counts.values() if cnt > 1)

    return total_dup, scg_retained


def refine_contaminated_bins(
    clusters_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=1
):
    logger.info("Refining contaminated bins (k-means only)...")

    # Load embeddings once
    embeddings_path = os.path.join(args.output, "embeddings.csv")
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings not found at {embeddings_path}")
        return clusters_df, fragments_dict, {}
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)

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

    contaminated_bins = [
        bin_id
        for bin_id, info in duplication_results.items()
        if len(info.get("duplicated_genes", {})) >= getattr(args, "min_duplications_for_refinement", 1)
    ]

    if not contaminated_bins:
        logger.info("No contaminated bins found; refinement skipped")
        return clusters_df, fragments_dict, {}

    refinement_summary = {}
    new_bins_dfs = []
    processed_bins = set()

    for bin_id in contaminated_bins:
        refined_df = refine_bin(
            clusters_df,
            embeddings_df,
            bin_id,
            gene_mappings_cache,
            duplication_results,
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
            "reason": "kmeans_split",
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
