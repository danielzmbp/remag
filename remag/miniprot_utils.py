"""
Miniprot utilities for core gene duplication checking.

This module consolidates the core gene duplication checking functionality
that was previously duplicated between quality.py and refinement.py.
"""

import json
import os
import shutil
from tqdm import tqdm
from loguru import logger

from .utils import extract_base_contig_name, ContigHeaderMapper


def get_core_gene_duplication_results_path(args):
    """Get the path for the core gene duplication results file."""
    return os.path.join(args.output, "core_gene_duplication_results.json")


def check_core_gene_duplications(clusters_df, fragments_dict, args, 
                                target_coverage_threshold=0.50, 
                                identity_threshold=0.50,
                                use_header_cache=False):
    """
    Check for duplicated core genes using miniprot.
    
    This function consolidates the logic previously duplicated between
    quality.py and refinement.py with configurable thresholds.
    
    Args:
        clusters_df: DataFrame with cluster assignments
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object containing output directory, cores, etc.
        target_coverage_threshold: Minimum target coverage (0.50 for quality, 0.60 for refinement)
        identity_threshold: Minimum identity (0.50 for quality, 0.40 for refinement)
        use_header_cache: Whether to use function-level caching for header lookup
    
    Returns:
        DataFrame: Updated clusters_df with duplication information
    """
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "db", "refseq_db.faa.gz"
    )
    if not os.path.exists(db_path):
        logger.warning(
            "Eukaryotic database not found, skipping core gene duplication check"
        )
        clusters_df["has_duplicated_core_genes"] = False
        return clusters_df

    logger.info("Checking for duplicated core genes using miniprot...")

    # Create temporary directory
    temp_dir = os.path.join(args.output, "temp_miniprot")
    os.makedirs(temp_dir, exist_ok=True)

    # Group contigs by cluster (clusters_df is now contig-level)
    # Use ContigHeaderMapper for efficient lookups
    if use_header_cache:
        # Use cached mapper if available
        if not hasattr(check_core_gene_duplications, '_mapper_cache'):
            check_core_gene_duplications._mapper_cache = ContigHeaderMapper(fragments_dict)
        mapper = check_core_gene_duplications._mapper_cache
    else:
        # Create new mapper
        mapper = ContigHeaderMapper(fragments_dict)
    
    cluster_contig_dict = {}
    for _, row in clusters_df.iterrows():
        contig_name = row["contig"]
        cluster_id = row["cluster"]

        original_header = mapper.get_header(contig_name)
        if original_header:
            if cluster_id not in cluster_contig_dict:
                cluster_contig_dict[cluster_id] = set()
            cluster_contig_dict[cluster_id].add(original_header)

    # Filter clusters by size and exclude noise
    filtered_clusters = {}
    for cluster_id, contig_headers in cluster_contig_dict.items():
        if cluster_id == "noise":
            continue
        total_size = sum(len(fragments_dict[h]["sequence"]) for h in contig_headers)
        if total_size >= args.min_bin_size:
            filtered_clusters[cluster_id] = contig_headers

    duplication_results = {}

    try:
        for cluster_id, contig_headers in tqdm(
            filtered_clusters.items(), desc="Checking core gene duplications"
        ):
            # Create temporary FASTA file
            bin_fasta = os.path.join(temp_dir, f"{cluster_id}.fa")
            with open(bin_fasta, "w") as f:
                for header in contig_headers:
                    seq = fragments_dict[header]["sequence"]
                    f.write(f">{header}\n")
                    for i in range(0, len(seq), 60):
                        f.write(f"{seq[i: i+60]}\n")

            # Run miniprot
            miniprot_output = os.path.join(temp_dir, f"{cluster_id}.paf")
            db_to_use = db_path  # Use the compressed file directly
            cmd = f'miniprot -t {args.cores} "{bin_fasta}" "{db_to_use}" > "{miniprot_output}" 2>/dev/null'

            if args.verbose:
                logger.debug(f"Running miniprot command: {cmd}")

            try:
                result = os.system(cmd)
                if result == 0:
                    # Parse miniprot output
                    best_alignments = (
                        {}
                    )  # {(contig, gene_family): {score, coverage, identity}}

                    if (
                        os.path.exists(miniprot_output)
                        and os.path.getsize(miniprot_output) > 0
                    ):
                        if args.verbose:
                            logger.debug(
                                f"Miniprot output file exists and has size: {os.path.getsize(miniprot_output)} bytes"
                            )
                        with open(miniprot_output, "r") as paf_file:
                            for line in paf_file:
                                if line.startswith("#") or not line.strip():
                                    continue

                                parts = line.strip().split("\t")
                                if len(parts) >= 11:
                                    try:
                                        query_name = parts[0]  # Protein name
                                        target_name = parts[5]  # Contig name
                                        target_length = int(parts[6])
                                        target_start = int(parts[7])
                                        target_end = int(parts[8])
                                        matching_bases = int(parts[9])
                                        alignment_length = int(parts[10])

                                        # Extract gene family code from protein name (query)
                                        full_gene_id = query_name.split()[0]
                                        if ":" in full_gene_id:
                                            gene_family_code = full_gene_id.split(":")[
                                                -1
                                            ]
                                        else:
                                            gene_family_code = full_gene_id

                                        # Calculate quality metrics
                                        target_coverage = (
                                            (target_end - target_start) / target_length
                                            if target_length > 0
                                            else 0
                                        )
                                        identity = (
                                            matching_bases / alignment_length
                                            if alignment_length > 0
                                            else 0
                                        )

                                        # Only consider high-quality alignments with configurable thresholds
                                        if target_coverage >= target_coverage_threshold and identity >= identity_threshold:
                                            score = target_coverage * identity
                                            key = (
                                                target_name,
                                                gene_family_code,
                                            )  # Use contig name as key

                                            if (
                                                key not in best_alignments
                                                or score > best_alignments[key]["score"]
                                            ):
                                                best_alignments[key] = {
                                                    "score": score,
                                                    "coverage": target_coverage,
                                                    "identity": identity,
                                                    "gene_family": gene_family_code,
                                                }

                                    except (ValueError, IndexError):
                                        continue

                    # Count gene families present in each contig
                    contig_genes = {}
                    for (contig, gene_family), alignment in best_alignments.items():
                        if contig not in contig_genes:
                            contig_genes[contig] = set()
                        contig_genes[contig].add(gene_family)

                    # Count total occurrences of each gene family
                    gene_counts = {}
                    for contig, gene_families in contig_genes.items():
                        for gene_family in gene_families:
                            if gene_family not in gene_counts:
                                gene_counts[gene_family] = 0
                            gene_counts[gene_family] += 1

                    # Check for duplications
                    duplicated_genes = {
                        gene: count for gene, count in gene_counts.items() if count > 1
                    }
                    has_duplications = len(duplicated_genes) > 0

                    duplication_results[cluster_id] = {
                        "has_duplications": has_duplications,
                        "duplicated_genes": duplicated_genes,
                        "total_genes_found": len(gene_counts),
                    }

                else:
                    duplication_results[cluster_id] = {
                        "has_duplications": False,
                        "duplicated_genes": {},
                        "total_genes_found": 0,
                    }

            except Exception as e:
                logger.warning(f"Error running miniprot for {cluster_id}: {e}")
                duplication_results[cluster_id] = {
                    "has_duplications": False,
                    "duplicated_genes": {},
                    "total_genes_found": 0,
                }

    finally:
        # Clean up temp_miniprot folder unless keeping intermediate files
        if not getattr(args, "keep_intermediate", False):
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary miniprot files at: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary miniprot files: {e}")
        else:
            logger.info(f"Miniprot files preserved at: {temp_dir}")

    # Add duplication information to clusters_df
    clusters_df = clusters_df.copy()
    clusters_df["has_duplicated_core_genes"] = False
    clusters_df["duplicated_core_genes_count"] = 0
    clusters_df["total_core_genes_found"] = 0

    for cluster_id, result in duplication_results.items():
        mask = clusters_df["cluster"] == cluster_id
        clusters_df.loc[mask, "has_duplicated_core_genes"] = result["has_duplications"]
        clusters_df.loc[mask, "duplicated_core_genes_count"] = len(
            result["duplicated_genes"]
        )
        clusters_df.loc[mask, "total_core_genes_found"] = result["total_genes_found"]

    # Log summary
    bins_with_duplications = sum(
        1 for r in duplication_results.values() if r["has_duplications"]
    )
    total_bins_checked = len(duplication_results)
    logger.info(
        f"Checked {total_bins_checked} bins: {bins_with_duplications} have duplicated core genes"
    )

    # Save results only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        results_path = get_core_gene_duplication_results_path(args)
        with open(results_path, "w") as f:
            json.dump(duplication_results, f, indent=2)

    return clusters_df