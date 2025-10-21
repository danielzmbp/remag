"""
Core module for REMAG - Main execution logic
"""

import json
import os
import sys
from loguru import logger

from .utils import setup_logging
from .features import filter_bacterial_contigs, get_features
from .models import train_siamese_network, generate_embeddings
from .clustering import cluster_contigs
from .miniprot_utils import check_core_gene_duplications, check_core_gene_duplications_from_cache, get_gene_mappings_cache_path
from .refinement import refine_contaminated_bins
from .output import save_clusters_as_fasta


def main(args):
    try:
        setup_logging(args.output, verbose=args.verbose)
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to initialize output directory: {e}")
        sys.exit(1)
    
    if getattr(args, "keep_intermediate", False):
        params_path = os.path.join(args.output, "params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)
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
            cores=args.cores,
            hyenadna_batch_size=hyenadna_batch_size,
        )
        logger.info(f"Using filtered FASTA file: {input_fasta}")
    else:
        logger.info("Skipping eukaryotic filtering as requested")

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

    # Optionally determine optimal resolution automatically
    auto_resolution_enabled = False
    if getattr(args, "auto_resolution", False):
        logger.info("Auto-resolution enabled - determining optimal Leiden resolution...")
        try:
            from .adaptive_resolution import determine_optimal_resolution
            optimal_resolution = determine_optimal_resolution(embeddings_df, fragments_dict, args)
            # Update args with optimal resolution
            args.leiden_resolution = optimal_resolution
            logger.info(f"Using automatically determined resolution: {optimal_resolution:.2f}")
            auto_resolution_enabled = True
        except Exception as e:
            logger.warning(f"Adaptive resolution determination failed: {e}")
            logger.warning(f"Falling back to manual resolution: {args.leiden_resolution}")

    try:
        clusters_df = cluster_contigs(embeddings_df, fragments_dict, args)
    except Exception as e:
        logger.error(f"Failed to cluster contigs: {e}")
        sys.exit(1)

    # Check for duplicated core genes using miniprot (using compleasm-style thresholds)
    logger.info("Checking for duplicated core genes...")

    # If auto-resolution was enabled, try to reuse the gene mappings cache
    gene_mappings_cache = None
    if auto_resolution_enabled:
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    gene_mappings_cache = json.load(f)
                logger.info(f"Loaded gene mappings cache from auto-resolution ({len(gene_mappings_cache)} contigs)")
                logger.info("Using cached gene mappings instead of re-running miniprot")
            except Exception as e:
                logger.warning(f"Failed to load gene mappings cache: {e}")
                gene_mappings_cache = None

    # Use cached approach if available, otherwise run miniprot
    if gene_mappings_cache is not None:
        try:
            clusters_df = check_core_gene_duplications_from_cache(
                clusters_df,
                gene_mappings_cache,
                args
            )
            # Store cache in args for refinement
            args._gene_mappings_cache = gene_mappings_cache
        except Exception as e:
            logger.warning(f"Cache-based duplication check failed: {e}")
            logger.warning("Falling back to full miniprot run")
            clusters_df = check_core_gene_duplications(
                clusters_df,
                fragments_dict,
                args,
                target_coverage_threshold=0.55,
                identity_threshold=0.35,
                use_header_cache=False
            )
    else:
        clusters_df = check_core_gene_duplications(
            clusters_df,
            fragments_dict,
            args,
            target_coverage_threshold=0.55,
            identity_threshold=0.35,
            use_header_cache=False
        )

    skip_refinement = getattr(args, "skip_refinement", False)
    if not skip_refinement:
        logger.info("Refining contaminated bins...")
        clusters_df, fragments_dict, refinement_summary = refine_contaminated_bins(
            clusters_df,
            fragments_dict,
            args,
            refinement_round=1,
            max_refinement_rounds=args.max_refinement_rounds,
        )
    else:
        logger.info("Skipping refinement")
        refinement_summary = {}

    if refinement_summary and getattr(args, "keep_intermediate", False):
        refinement_summary_path = os.path.join(args.output, "refinement_summary.json")
        with open(refinement_summary_path, "w", encoding="utf-8") as f:
            json.dump(refinement_summary, f, indent=2)

    # Save updated bins.csv with refined cluster assignments (excluding noise)
    logger.info("Saving final bins.csv with refined cluster assignments...")
    bins_csv_path = os.path.join(args.output, "bins.csv")
    final_bins_df = clusters_df[clusters_df["cluster"] != "noise"].copy()
    # Keep only the first two columns: contig and cluster
    final_bins_df = final_bins_df[["contig", "cluster"]]
    final_bins_df.to_csv(bins_csv_path, index=False)
    logger.info(f"bins.csv saved with {len(final_bins_df)} contigs from refined clusters")

    logger.info("Saving bins as FASTA files...")
    valid_bins = save_clusters_as_fasta(clusters_df, fragments_dict, args)
    
    # Filter bins.csv to only include contigs from valid bins (those that meet minimum size)
    logger.info("Filtering bins.csv to match saved bins...")
    if os.path.exists(bins_csv_path):
        import pandas as pd
        bins_df = pd.read_csv(bins_csv_path)
        filtered_bins_df = bins_df[bins_df["cluster"].isin(valid_bins)]
        # Ensure only the first two columns are kept
        filtered_bins_df = filtered_bins_df[["contig", "cluster"]]
        filtered_bins_df.to_csv(bins_csv_path, index=False)
        logger.info(f"bins.csv now contains {len(filtered_bins_df)} contigs from {len(valid_bins)} valid bins")

    logger.info("REMAG analysis completed successfully!")
