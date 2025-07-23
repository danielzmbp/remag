"""
Refinement module for REMAG
"""

import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from .utils import extract_base_contig_name, ContigHeaderMapper, group_contigs_by_cluster


from .miniprot_utils import check_core_gene_duplications, get_core_gene_duplication_results_path


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

    # Count contigs per cluster using utility function
    logger.debug("Counting contigs per cluster...")
    cluster_contig_counts = group_contigs_by_cluster(contig_clusters_df)

    total_clusters = len(cluster_contig_counts)
    if total_clusters > 0:
        logger.debug(f"K-means created {total_clusters} clusters")

    return contig_clusters_df


def _refine_single_bin_worker(args_tuple):
    """
    Worker function to refine a single contaminated bin in parallel.

    Args:
        args_tuple: Tuple containing (bin_id, bin_fragments, fragments_dict, args,
                   duplication_results, refinement_round, max_refinement_rounds,
                   refinement_dir, cores_per_bin)

    Returns:
        dict: Results of the bin refinement including status and refined clusters
    """
    (
        bin_id,
        bin_fragments,
        fragments_dict,
        args,
        duplication_results,
        refinement_round,
        max_refinement_rounds,
        refinement_dir,
        cores_per_bin,
    ) = args_tuple

    try:
        logger.info(f"Worker: Starting refinement of {bin_id}...")

        # Create bin-specific directory
        bin_refinement_dir = os.path.join(refinement_dir, f"bin_{bin_id}")
        os.makedirs(bin_refinement_dir, exist_ok=True)

        # Get original contigs for this bin (bin_fragments is now contig-level)
        # Use ContigHeaderMapper for efficient O(1) lookups
        mapper = ContigHeaderMapper(fragments_dict)
        bin_contigs = set()
        for _, row in bin_fragments.iterrows():
            contig_name = row["contig"]
            header = mapper.get_header(contig_name)
            if header:
                bin_contigs.add(header)

        if not bin_contigs:
            return {
                "bin_id": bin_id,
                "status": "failed",
                "reason": "no_contigs",
                "sub_bins": 0,
                "clusters_df": None,
            }

        # Write bin FASTA
        bin_fasta = os.path.join(bin_refinement_dir, f"{bin_id}.fa")
        with open(bin_fasta, "w") as f:
            for contig_header in bin_contigs:
                seq = fragments_dict[contig_header]["sequence"]
                f.write(f">{contig_header}\n")
                for i in range(0, len(seq), 60):
                    f.write(f"{seq[i: i+60]}\n")

        # Import the required functions here to avoid circular imports
        from .features import get_features
        from .models import train_siamese_network, generate_embeddings

        # Create refined args for this bin (with adjusted parameters)
        refined_args = argparse.Namespace(**vars(args))
        refined_args.fasta = bin_fasta
        refined_args.output = bin_refinement_dir
        refined_args.cores = cores_per_bin  # Use allocated cores for this worker
        # Set refinement-specific flags
        refined_args.skip_bacterial_filter = True  # Already filtered in main pipeline
        refined_args.skip_refinement = refinement_round >= max_refinement_rounds
        # Use same parameters as main pipeline for consistency
        refined_args.epochs = args.epochs
        refined_args.min_cluster_size = args.min_cluster_size
        refined_args.min_samples = args.min_samples
        refined_args.max_positive_pairs = args.max_positive_pairs
        # Use half the batch size for refinement step
        refined_args.batch_size = args.batch_size // 2

        logger.debug(
            f"Worker: Running refinement pipeline for {bin_id} with {len(bin_contigs)} contigs using {cores_per_bin} cores"
        )

        # Run the refinement pipeline
        refined_features_df, refined_fragments_dict = get_features(
            refined_args.fasta,
            refined_args.bam,
            refined_args.tsv,
            refined_args.output,
            refined_args.min_contig_length,
            refined_args.cores,
            getattr(
                refined_args, "num_augmentations", 0
            ),  # No augmentations for refinement
        )

        if refined_features_df.empty:
            return {
                "bin_id": bin_id,
                "status": "failed",
                "reason": "no_features",
                "sub_bins": 0,
                "clusters_df": None,
            }

        # Train siamese network for this bin
        logger.debug(f"Worker: Training siamese network for {bin_id} refinement...")
        refined_model = train_siamese_network(refined_features_df, refined_args)

        # Generate embeddings
        logger.debug(f"Worker: Generating embeddings for {bin_id} refinement...")
        refined_embeddings_df = generate_embeddings(
            refined_model, refined_features_df, refined_args
        )

        # Cluster the refined embeddings using K-means based on duplicated core genes
        logger.debug(f"Worker: Clustering {bin_id} refinement using K-means...")
        refined_clusters_df = cluster_contigs_kmeans_refinement(
            refined_embeddings_df,
            refined_fragments_dict,
            refined_args,
            bin_id,
            duplication_results,
        )

        # Check for duplicated core genes in refined bins
        logger.debug(
            f"Worker: Checking core gene duplications in {bin_id} refined sub-bins..."
        )
        refined_clusters_df = check_core_gene_duplications(
            refined_clusters_df, 
            refined_fragments_dict, 
            refined_args,
            target_coverage_threshold=0.60,
            identity_threshold=0.40,
            use_header_cache=True
        )

        # Rename clusters to avoid conflicts (prefix with original bin name)
        refined_clusters_df = refined_clusters_df.copy()
        refined_clusters_df["original_bin"] = bin_id
        refined_clusters_df["cluster"] = refined_clusters_df["cluster"].apply(
            lambda x: f"{bin_id}_refined_{x}" if x != "noise" else "noise"
        )

        # Count successful sub-bins (exclude noise)
        sub_bins = refined_clusters_df[refined_clusters_df["cluster"] != "noise"][
            "cluster"
        ].nunique()

        if sub_bins > 1:
            logger.info(
                f"Worker: Successfully refined {bin_id} into {sub_bins} sub-bins"
            )
            return {
                "bin_id": bin_id,
                "status": "success",
                "sub_bins": sub_bins,
                "clusters_df": refined_clusters_df,
            }
        else:
            logger.warning(
                f"Worker: Refinement of {bin_id} produced only {sub_bins} sub-bins"
            )
            return {
                "bin_id": bin_id,
                "status": "insufficient_split",
                "sub_bins": sub_bins,
                "clusters_df": None,
            }

    except Exception as e:
        logger.error(f"Worker: Error during refinement of {bin_id}: {e}")
        return {
            "bin_id": bin_id,
            "status": "error",
            "reason": str(e),
            "sub_bins": 0,
            "clusters_df": None,
        }


def refine_contaminated_bins(
    clusters_df, fragments_dict, args, refinement_round=1, max_refinement_rounds=2
):
    """
    Refine bins that have duplicated core genes by re-running the entire pipeline
    on each contaminated bin to split it into cleaner sub-bins. This function performs
    iterative refinement:

    1. First round: Refines bins with duplicated core genes using K-means clustering
       based on the number of duplicated core genes
    2. Checks for duplications in refined sub-bins
    3. Second round: Refines still-contaminated sub-bins (if max_refinement_rounds >= 2)
    4. Final check: Marks any remaining contaminated bins without further refinement

    This approach allows for progressive improvement of bin quality while preventing
    infinite refinement loops.

    Args:
        clusters_df: DataFrame with cluster assignments and duplication flags
        fragments_dict: Dictionary containing fragment sequences
        args: Command line arguments
        refinement_round: Current refinement round (1-indexed)
        max_refinement_rounds: Maximum number of refinement rounds to perform

    Returns:
        tuple: (refined_clusters_df, refined_fragments_dict, refinement_summary)
    """
    # Identify contaminated bins
    contaminated_bins = []
    if "has_duplicated_core_genes" in clusters_df.columns:
        contaminated_clusters = clusters_df[
            clusters_df["has_duplicated_core_genes"] == True
        ]["cluster"].unique()
        contaminated_bins = [c for c in contaminated_clusters if c != "noise"]

    if not contaminated_bins:
        logger.info("No contaminated bins found, skipping refinement")
        return clusters_df, fragments_dict, {}

    if refinement_round > max_refinement_rounds:
        logger.info(
            f"Maximum refinement rounds ({max_refinement_rounds}) reached, marking remaining contaminated bins without further refinement"
        )
        return clusters_df, fragments_dict, {}

    logger.info(
        f"Starting refinement round {refinement_round} of {len(contaminated_bins)} contaminated bins..."
    )
    if refinement_round == 1:
        logger.info(
            f"Maximum {max_refinement_rounds} refinement rounds will be performed"
        )
        logger.info(
            "Refinement approach: bins with duplicated core genes will be split into sub-bins using K-means clustering"
        )
        logger.info("Number of clusters = number of duplicated single copy core genes")
        logger.info("After refinement, sub-bins will be checked again for duplications")
        if max_refinement_rounds > 1:
            logger.info(
                "Still-contaminated sub-bins will undergo additional refinement rounds"
            )
        logger.info(
            "After maximum rounds, any remaining contaminated bins will be marked but not refined further"
        )

    # Create refinement output directory
    refinement_dir = os.path.join(args.output, "refinement")
    os.makedirs(refinement_dir, exist_ok=True)

    # Load duplication results for K-means clustering
    duplication_results = {}
    results_path = get_core_gene_duplication_results_path(args)
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                duplication_results = json.load(f)
            logger.info(
                f"Loaded duplication results for {len(duplication_results)} bins"
            )
        except Exception as e:
            logger.warning(f"Failed to load duplication results: {e}")
    else:
        logger.warning(
            "No duplication results file found, will use default k=2 for K-means"
        )

    # No fragment mapping needed since clusters_df is already contig-level

    all_refined_clusters = []
    refinement_summary = {}

    # Calculate core allocation for parallel processing
    num_contaminated_bins = len(contaminated_bins)
    if num_contaminated_bins == 0:
        return clusters_df, fragments_dict, {}

    # Determine optimal parallelization strategy
    max_parallel_bins = (
        max(1, min(args.cores // 2, num_contaminated_bins))
        if num_contaminated_bins > 0
        else 1
    )
    cores_per_bin = max(1, args.cores // max_parallel_bins)

    logger.info(
        f"Processing {num_contaminated_bins} contaminated bins using {max_parallel_bins} parallel workers"
    )
    logger.info(f"Each worker will use {cores_per_bin} cores")

    # Filter bins with insufficient contigs before parallel processing
    valid_bins = []
    for bin_id in contaminated_bins:
        # Get contigs belonging to this bin (clusters_df is now contig-level)
        bin_contigs_df = clusters_df[clusters_df["cluster"] == bin_id]
        if bin_contigs_df.empty:
            continue

        # Get original contigs for this bin - use ContigHeaderMapper for O(1) lookup
        mapper = ContigHeaderMapper(fragments_dict)
        bin_contigs = set()
        for _, row in bin_contigs_df.iterrows():
            contig_name = row["contig"]
            header = mapper.get_header(contig_name)
            if header:
                bin_contigs.add(header)

        if len(bin_contigs) < 2:
            logger.warning(
                f"Bin {bin_id} has fewer than 2 contigs, skipping refinement"
            )
            refinement_summary[bin_id] = {
                "status": "skipped",
                "reason": "too_few_contigs",
                "sub_bins": 0,
            }
            continue

        valid_bins.append((bin_id, bin_contigs_df))

    if not valid_bins:
        logger.info("No valid bins found for refinement")
        return clusters_df, fragments_dict, refinement_summary

    # Prepare worker arguments for parallel processing
    worker_args = []
    for bin_id, bin_fragments in valid_bins:
        worker_args.append(
            (
                bin_id,
                bin_fragments,
                fragments_dict,
                args,
                duplication_results,
                refinement_round,
                max_refinement_rounds,
                refinement_dir,
                cores_per_bin,
            )
        )

    # Process bins sequentially
    logger.info(f"Starting sequential refinement of {len(valid_bins)} bins...")
    results = []
    for worker_arg in tqdm(worker_args, desc="Refining contaminated bins (sequential)"):
        results.append(_refine_single_bin_worker(worker_arg))

    # Process results
    for result in results:
        bin_id = result["bin_id"]
        refinement_summary[bin_id] = {
            "status": result["status"],
            "sub_bins": result["sub_bins"],
        }

        if result["status"] == "success" and result["clusters_df"] is not None:
            all_refined_clusters.append(result["clusters_df"])
            logger.info(
                f"Successfully refined {bin_id} into {result['sub_bins']} sub-bins"
            )
        elif result["status"] == "error":
            refinement_summary[bin_id]["reason"] = result["reason"]
            logger.error(f"Error during refinement of {bin_id}: {result['reason']}")
        elif result["status"] == "insufficient_split":
            logger.warning(
                f"Refinement of {bin_id} produced only {result['sub_bins']} sub-bins, keeping original"
            )
        elif result["status"] == "failed":
            refinement_summary[bin_id]["reason"] = result["reason"]
            logger.warning(f"Refinement of {bin_id} failed: {result['reason']}")

    logger.info(f"Parallel refinement completed. Processed {len(results)} bins.")

    # Combine all refined clusters
    if all_refined_clusters:
        logger.info("Integrating refined bins into final results...")

        # Remove contaminated bins from original results
        clean_original_clusters = clusters_df[
            ~clusters_df["cluster"].isin(contaminated_bins)
        ].copy()

        # Add refined clusters
        all_refined_df = pd.concat(all_refined_clusters, ignore_index=True)

        # Combine clean original + refined clusters
        final_clusters_df = pd.concat(
            [clean_original_clusters, all_refined_df], ignore_index=True
        )

        # Update fragments_dict to include refined fragments (they should already be there)
        # No need to modify fragments_dict as it contains original sequences

        logger.info(f"Refinement round {refinement_round} complete!")
        success_count = sum(1 for s in refinement_summary.values() if s["status"] == "success")
        logger.info(f"Refinement summary: {success_count}/{len(refinement_summary)} bins successfully refined")

        # Check if we should perform another round of refinement
        if refinement_round < max_refinement_rounds:
            logger.info(
                f"Checking for contaminated bins requiring round {refinement_round+1} refinement..."
            )

            # Check for contaminated bins in the current result
            still_contaminated_bins = []
            if "has_duplicated_core_genes" in final_clusters_df.columns:
                still_contaminated_clusters = final_clusters_df[
                    final_clusters_df["has_duplicated_core_genes"] == True
                ]["cluster"].unique()
                still_contaminated_bins = [
                    c for c in still_contaminated_clusters if c != "noise"
                ]

            if still_contaminated_bins:
                logger.info(
                    f"Found {len(still_contaminated_bins)} bins still needing refinement, starting round {refinement_round+1}"
                )

                # Recursively refine the still-contaminated bins
                final_clusters_df, fragments_dict, additional_refinement_summary = (
                    refine_contaminated_bins(
                        final_clusters_df,
                        fragments_dict,
                        args,
                        refinement_round=refinement_round + 1,
                        max_refinement_rounds=max_refinement_rounds,
                    )
                )

                # Merge refinement summaries
                refinement_summary.update(additional_refinement_summary)
            else:
                logger.info("No more contaminated bins found, refinement complete!")

        return final_clusters_df, fragments_dict, refinement_summary
    else:
        logger.warning("No bins were successfully refined, keeping original results")

        # Check if we should perform another round anyway (in case original bins are still contaminated)
        if refinement_round < max_refinement_rounds:
            logger.info(
                "Checking if original contaminated bins should undergo another refinement round..."
            )
            return refine_contaminated_bins(
                clusters_df,
                fragments_dict,
                args,
                refinement_round=refinement_round + 1,
                max_refinement_rounds=max_refinement_rounds,
            )

        return clusters_df, fragments_dict, refinement_summary
