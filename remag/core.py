"""
Core module for REMAG - Main execution logic
"""

import json
import os
import sys
from importlib.metadata import version

from loguru import logger

from .clustering import cluster_contigs
from .features import filter_bacterial_contigs, get_features
from .miniprot_utils import (
    check_core_gene_duplications,
    check_core_gene_duplications_from_cache,
    get_gene_mappings_cache_path,
)
from .models import generate_embeddings, train_siamese_network
from .output import save_clusters_as_fasta
from .rescue import rescue_fragmented_bins  # Import the new rescue function
from .utils import setup_logging


def main(args):
    try:
        setup_logging(args.output, verbose=args.verbose)
        os.makedirs(args.output, exist_ok=True)
        # Log the exact command used
        logger.info(f"Command: {' '.join(sys.argv)}")
    except Exception as e:
        logger.error(f"Failed to initialize output directory: {e}")
        sys.exit(1)

    if getattr(args, "keep_intermediate", False):
        params_path = os.path.join(args.output, "params.json")
        params = {"version": version("remag"), **vars(args)}
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        logger.debug(f"Run parameters saved to {params_path}")

    # Apply eukaryotic filtering if not skipped
    input_fasta = args.fasta
    skip_bacterial_filter = getattr(args, "skip_bacterial_filter", False)
    if not skip_bacterial_filter:
        logger.info("Filtering non-eukaryotic contigs using HyenaDNA classifier...")
        hyenadna_batch_size = getattr(args, "hyenadna_batch_size", 1024)
        input_fasta = filter_bacterial_contigs(
            args.fasta,
            args.output,
            min_contig_length=args.min_contig_length,
            hyenadna_batch_size=hyenadna_batch_size,
        )
    else:
        logger.info("Skipping eukaryotic filtering as requested")

    # If user only wants filtering, exit early
    if getattr(args, "filter_only", False):
        logger.info(
            "Filter-only mode enabled; skipping feature generation and binning."
        )
        return

    # Generate all features with full augmentations upfront
    logger.info(
        f"Generating features with {args.num_augmentations} augmentations per contig..."
    )
    try:
        features_df, fragments_dict = get_features(
            input_fasta,  # Use filtered FASTA if bacterial filtering was applied
            args.bam,
            args.tsv,
            args.output,
            args.min_contig_length,
            args.cores,
            args.num_augmentations,
            args,  # Pass args for keep_intermediate check
        )
    except Exception as e:
        logger.error(f"Failed to generate features: {e}")
        sys.exit(1)

    if features_df.empty:
        logger.error("No features generated. Exiting.")
        sys.exit(1)

    logger.info("Training neural network and generating embeddings...")
    try:
        model = train_siamese_network(features_df, args)
        embeddings_df = generate_embeddings(model, features_df, args)
    except Exception as e:
        logger.error(f"Failed to train model or generate embeddings: {e}")
        sys.exit(1)

    # Generate gene mappings for greedy clustering
    logger.info("Generating gene mappings for greedy clustering...")
    try:
        from .miniprot_utils import estimate_organisms_from_all_contigs

        # Check if cache exists
        cache_path = get_gene_mappings_cache_path(args)
        gene_mappings = {}

        if os.path.exists(cache_path):
            logger.info(f"Loading existing gene mappings from {cache_path}")
            with open(cache_path, "r") as f:
                gene_mappings = json.load(f)
        else:
            logger.info("Running miniprot to generate gene mappings...")
            _ = estimate_organisms_from_all_contigs(fragments_dict, args)

            # Reload from the newly created file
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    gene_mappings = json.load(f)
            else:
                logger.warning("Gene mappings cache not found after miniprot run.")

        logger.info(f"Loaded {len(gene_mappings)} contig gene mappings")

        # Store in args for later use
        args._gene_mappings_cache = gene_mappings

    except Exception as e:
        logger.error(f"Failed to generate gene mappings: {e}")
        # Initialize empty mappings to allow clustering to proceed (though quality scores will be -100)
        gene_mappings = {}

    try:
        clusters_df = cluster_contigs(
            embeddings_df, fragments_dict, gene_mappings, args
        )
    except Exception as e:
        logger.error(f"Failed to cluster contigs: {e}")
        sys.exit(1)

    # Check for duplicated core genes using miniprot (using compleasm-style thresholds)
    logger.info("Checking for duplicated core genes...")

    # Use the gene mappings we already loaded for greedy clustering
    gene_mappings_cache = (
        args._gene_mappings_cache if hasattr(args, "_gene_mappings_cache") else None
    )

    # Use cached approach if available, otherwise run miniprot
    if gene_mappings_cache is not None:
        try:
            clusters_df = check_core_gene_duplications_from_cache(
                clusters_df, gene_mappings_cache, args
            )
        except Exception as e:
            logger.warning(f"Cache-based duplication check failed: {e}")
            logger.warning("Falling back to full miniprot run")
            clusters_df = check_core_gene_duplications(
                clusters_df,
                fragments_dict,
                args,
                target_coverage_threshold=0.55,
                identity_threshold=0.35,
            )
    else:
        clusters_df = check_core_gene_duplications(
            clusters_df,
            fragments_dict,
            args,
            target_coverage_threshold=0.55,
            identity_threshold=0.35,
        )

    # --- Run rescue step ---
    if not getattr(args, "skip_rescue", False):
        logger.info("Applying bin rescue strategy...")
        clusters_df = rescue_fragmented_bins(
            clusters_df,
            embeddings_df,
            fragments_dict,
            args,
            similarity_threshold=0.70,  # Based on our proof of concept
            max_duplication_increase=getattr(
                args, "rescue_max_duplication_increase", 5.0
            ),
            max_total_duplication=getattr(args, "rescue_max_total_duplication", 5.0),
        )
    else:
        logger.info("Skipping rescue step")

    # Save updated bins.csv with refined cluster assignments (excluding noise)
    logger.info("Saving final bins.csv with refined cluster assignments...")
    bins_csv_path = os.path.join(args.output, "bins.csv")
    final_bins_df = clusters_df[clusters_df["cluster"] != "noise"].copy()
    # Keep only the first two columns: contig and cluster
    final_bins_df = final_bins_df[["contig", "cluster"]]

    valid_bins = save_clusters_as_fasta(clusters_df, fragments_dict, args)

    # Filter bins to only include contigs from valid bins (those that meet minimum size)
    logger.info("Filtering bins.csv to match saved bins...")
    filtered_bins_df = final_bins_df[final_bins_df["cluster"].isin(valid_bins)]
    filtered_bins_df.to_csv(bins_csv_path, index=False)
    logger.info(
        f"bins.csv saved with {len(filtered_bins_df)} contigs from {len(valid_bins)} valid bins"
    )

    logger.info("REMAG analysis completed successfully!")
