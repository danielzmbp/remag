"""Merge fragmented core bins, then recruit eligible unassigned contigs once."""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from .miniprot_utils import get_gene_mappings_cache_path
from .utils import contig_lengths_for_embeddings

RESCUE_ALGORITHM = "unified-v1"
MAX_MERGE_DUPLICATION_PERCENT = 10.0
NONWORSENING_MERGE_SIMILARITY = 0.95


def _weighted_centroid(members, embeddings_df, contig_lengths):
    """Return the length-weighted centroid of the given contigs' embeddings.

    All ``members`` must be present in ``embeddings_df.index``.
    """
    vecs = embeddings_df.loc[members].values
    weights = np.array(
        [contig_lengths.get(c, 1000) for c in members], dtype=float
    ).reshape(-1, 1)
    weights = weights / weights.sum()
    return np.sum(vecs * weights, axis=0)


def _marker_counts(members, gene_mappings):
    # Count a marker once per contig, regardless of its number of loci there.
    return Counter(gene for c in members for gene in gene_mappings.get(c, {}))


def _duplication(counts):
    return sum(n > 1 for n in counts.values()) / len(counts) * 100.0 if counts else 0.0


def get_bin_scg_stats(bin_contigs, gene_mappings_cache):
    """Return marker duplication percentage and distinct marker count."""
    counts = _marker_counts(bin_contigs, gene_mappings_cache)
    return _duplication(counts), len(counts)


def _bin_members(clusters):
    """Preserve first-bin occurrence and member row order, including small bins."""
    members = {}
    for contig, cluster in clusters[["contig", "cluster"]].itertuples(
        index=False, name=None
    ):
        if cluster != "noise":
            members.setdefault(cluster, []).append(contig)
    return members


def _merge_fragmented_bins(
    clusters,
    embeddings,
    lengths,
    genes,
    similarity_threshold,
    max_duplication_increase,
    max_total_duplication,
):
    """One stable smallest-first merge pass with fixed original centroids."""
    clusters = clusters.copy()
    members = _bin_members(clusters)
    centers = {}
    for name, contigs in members.items():
        available = [c for c in contigs if c in embeddings.index]
        if available:
            centers[name] = _weighted_centroid(available, embeddings, lengths)
    sizes = {b: sum(lengths[c] for c in members[b]) for b in centers}
    order = sorted(centers, key=sizes.get)  # Stable; no alphabetical tie-break.
    removed = set()
    cap = min(max_total_duplication, MAX_MERGE_DUPLICATION_PERCENT)
    for source in order:
        if source in removed:
            continue
        target, score = None, -1.0
        for candidate in order:
            if (
                candidate == source
                or candidate in removed
                or sizes[candidate] < sizes[source]
            ):
                continue
            sim = cosine_similarity(
                centers[source].reshape(1, -1), centers[candidate].reshape(1, -1)
            )[0, 0]
            if sim > score:
                target, score = candidate, sim
        if target is None or score < similarity_threshold:
            continue
        before, _ = get_bin_scg_stats(members[target], genes)
        after, _ = get_bin_scg_stats(members[target] + members[source], genes)
        nonworsening = (
            before > cap and after <= before and score >= NONWORSENING_MERGE_SIMILARITY
        )
        if after - before < max_duplication_increase and (after <= cap or nonworsening):
            logger.debug(
                f"Merging {source} into {target}: cosine={score:.3f}, duplication={before:.2f}%->{after:.2f}%"
            )
            clusters.loc[clusters["cluster"] == source, "cluster"] = target
            members[target].extend(members[source])
            sizes[target] += sizes[source]
            removed.add(source)
    return clusters


def _recruit_contigs(
    members,
    candidates,
    embeddings,
    genes,
    lengths,
    similarity_threshold=0.70,
    max_duplication_increase=5.0,
    max_total_duplication=5.0,
):
    """Fixed post-merge centers, first-target ties, cumulative marker safeguards."""
    members = {b: list(contigs) for b, contigs in members.items()}
    names = [
        b
        for b, contigs in members.items()
        if any(c in embeddings.index for c in contigs)
    ]
    if not names or not candidates:
        return members
    centers = np.vstack(
        [
            _weighted_centroid(
                [c for c in members[b] if c in embeddings.index], embeddings, lengths
            )
            for b in names
        ]
    )
    counts = {b: _marker_counts(members[b], genes) for b in names}
    # Bound the similarity matrix while preserving the original candidate order.
    for start in range(0, len(candidates), 4096):
        part = candidates[start : start + 4096]
        similarities = cosine_similarity(embeddings.loc[part].values, centers)
        for i, contig in enumerate(part):
            scores = similarities[i]
            best = int(np.argmax(scores))
            score = float(scores[best])
            # Match the native pairwise calculation at ties and the cutoff boundary.
            near_tie = (
                len(scores) > 1 and np.sort(scores)[-1] - np.sort(scores)[-2] <= 1e-12
            )
            if near_tie or abs(score - similarity_threshold) <= 1e-12:
                scores = [
                    float(
                        cosine_similarity(
                            embeddings.loc[contig].values.reshape(1, -1),
                            center.reshape(1, -1),
                        )[0, 0]
                    )
                    for center in centers
                ]
                best = max(range(len(scores)), key=scores.__getitem__)
                score = scores[best]
            if score < similarity_threshold:
                continue
            target = names[best]
            proposed = counts[target].copy()
            proposed.update(genes.get(contig, {}).keys())
            before, after = _duplication(counts[target]), _duplication(proposed)
            if genes.get(contig) and not (
                after <= max_total_duplication
                and after - before < max_duplication_increase
            ):
                continue  # Do not try a second-best target or the whole-bin exception.
            members[target].append(contig)
            counts[target] = proposed
    return members


def rescue_fragmented_bins(
    clusters_df,
    embeddings_df,
    fragments_dict,
    args,
    similarity_threshold=0.70,
    max_duplication_increase=5.0,
    max_total_duplication=5.0,
    min_contig_length=1,
):
    """Merge core bins and recruit all eligible noise/short contigs in one pass.

    Embeddings must share one trained space and contain only admitted contigs.
    The final minimum-bin-size filter belongs to export, after this entry point.
    An explicitly empty marker mapping is valid: marker-free candidates have no
    marker veto. Missing annotations are not treated as a successful empty map.
    """
    lengths = contig_lengths_for_embeddings(embeddings_df, fragments_dict)
    if (
        clusters_df["contig"].duplicated().any()
        or clusters_df[["contig", "cluster"]].isna().any().any()
    ):
        raise ValueError(
            "Rescue assignments contain duplicate or missing contig/cluster IDs."
        )
    if any(c not in lengths for c in clusters_df["contig"]):
        raise ValueError("Rescue assignments contain contigs without sequences.")
    genes = getattr(args, "_gene_mappings_cache", None)
    if genes is None:
        path = get_gene_mappings_cache_path(args)
        if not os.path.exists(path):
            raise ValueError("Rescue requires completed marker annotations.")
        with open(path) as handle:
            genes = json.load(handle)
    logger.info("Merging fragmented bins and recruiting unassigned contigs...")
    merged = _merge_fragmented_bins(
        clusters_df,
        embeddings_df,
        lengths,
        genes,
        similarity_threshold,
        max_duplication_increase,
        max_total_duplication,
    )
    seeds = _bin_members(merged)
    assigned = {c for contigs in seeds.values() for c in contigs}
    candidates = [
        c
        for c in embeddings_df.index
        if lengths[c] >= min_contig_length and c not in assigned
    ]
    recruited = _recruit_contigs(
        seeds,
        candidates,
        embeddings_df,
        genes,
        lengths,
        similarity_threshold,
        max_duplication_increase,
        max_total_duplication,
    )
    labels = {c: b for b, contigs in recruited.items() for c in contigs}
    # Preserve existing rows and append the short pool for complete noise provenance.
    present = set(merged["contig"])
    missing = [c for c in candidates if c not in present]
    if missing:
        merged = pd.concat(
            [merged, pd.DataFrame({"contig": missing, "cluster": "noise"})],
            ignore_index=True,
        )
    merged["cluster"] = [labels.get(c, "noise") for c in merged["contig"]]
    logger.info(
        f"Recruited {len(labels) - len(assigned)} of {len(candidates)} eligible unassigned contigs."
    )
    return merged
