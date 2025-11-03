"""
Core module for REMAG - Main execution logic
"""

import json
import os
import random
import sys
from importlib.metadata import version
from loguru import logger
import numpy as np
import torch

from .utils import setup_logging
from .features import filter_bacterial_contigs, get_features
from .models import train_siamese_network, generate_embeddings
from .clustering import cluster_contigs
from .miniprot_utils import (
    check_core_gene_duplications,
    check_core_gene_duplications_from_cache,
    get_gene_mappings_cache_path,
    estimate_organisms_from_all_contigs,
    check_miniprot_available
)
from .refinement import refine_contaminated_bins
from .output import save_clusters_as_fasta


def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Note: We don't set torch.backends.cudnn.deterministic=True
    # to preserve performance (best-effort determinism)
    logger.debug(f"Random seeds set to {seed} for reproducibility")


def main(args):
    try:
        setup_logging(args.output, verbose=args.verbose)
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to initialize output directory: {e}")
        sys.exit(1)

    # Set random seeds for reproducibility
    set_random_seeds(seed=42)

    if getattr(args, "keep_intermediate", False):
        params_path = os.path.join(args.output, "params.json")
        params = {
            "version": version("remag"),
            **vars(args)
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        logger.debug(f"Run parameters saved to {params_path}")

    # Apply eukaryotic filtering if not skipped
    input_fasta = args.fasta
    skip_bacterial_filter = getattr(args, "skip_bacterial_filter", False)
    if not skip_bacterial_filter:
        logger.info("Filtering non-eukaryotic contigs using HyenaDNA classifier...")
        hyenadna_batch_size = getattr(args, "hyenadna_batch_size", 1024)
        save_filtered_contigs = getattr(args, "save_filtered_contigs", False)
        input_fasta = filter_bacterial_contigs(
            args.fasta,
            args.output,
            min_contig_length=args.min_contig_length,
            cores=args.cores,
            hyenadna_batch_size=hyenadna_batch_size,
            save_filtered_contigs=save_filtered_contigs,
        )
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

    # Filter out zero-coverage contigs from training (but keep them for embeddings)
    coverage_columns = [col for col in features_df.columns if isinstance(col, str) and "coverage" in col.lower()]

    if coverage_columns:
        logger.info("Filtering zero-coverage contigs from training data...")

        # Identify rows with zero coverage across all samples
        zero_coverage_mask = (features_df[coverage_columns] == 0).all(axis=1)

        # Create training dataframe excluding zero-coverage rows
        features_df_training = features_df[~zero_coverage_mask].copy()

        zero_count = int(zero_coverage_mask.sum())
        logger.info(f"Excluded {zero_count} zero-coverage fragments from training")
        logger.info(f"Training with {len(features_df_training)} features, embeddings for all {len(features_df)} features")

        if features_df_training.empty:
            logger.error("No features with non-zero coverage found for training. Exiting.")
            sys.exit(1)
    else:
        logger.info("No coverage data - using all features for training")
        features_df_training = features_df

    # ====================================================================
    # CONSOLIDATED MINIPROT RUN: Run once and reuse for all downstream steps
    # ====================================================================
    # Determine if we need to run miniprot at all
    needs_miniprot = (
        getattr(args, 'auto_resolution', False)  # Auto-resolution needs it
        # Core gene duplication check will also use it if available
    )

    gene_mappings = None
    if needs_miniprot and check_miniprot_available():
        # Run miniprot on all contigs to create gene mappings
        gene_counts = estimate_organisms_from_all_contigs(fragments_dict, args)

        # Load the gene mappings cache that was just created
        cache_path = get_gene_mappings_cache_path(args)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    gene_mappings = json.load(f)
                logger.debug(f"Loaded gene mappings for {len(gene_mappings)} contigs - will reuse for all steps")
            except Exception as e:
                logger.warning(f"Failed to load gene mappings cache: {e}")
                gene_mappings = None
        else:
            logger.warning("Miniprot completed but no cache file was created")
    elif needs_miniprot and not check_miniprot_available():
        logger.warning("miniprot not available - some features will be disabled")

    logger.info("Training neural network and generating embeddings...")
    try:
        # Train on features excluding zero-coverage contigs
        model = train_siamese_network(features_df_training, args)
        # Generate embeddings for ALL contigs (including zero-coverage ones)
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
            optimal_resolution = determine_optimal_resolution(
                embeddings_df, fragments_dict, args, gene_mappings=gene_mappings
            )
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

    # Reuse the gene_mappings from the consolidated miniprot run if available
    # This avoids redundant miniprot execution
    if gene_mappings is not None:
        try:
            logger.debug("Using pre-computed gene mappings from consolidated miniprot run (no redundant execution)")
            clusters_df = check_core_gene_duplications_from_cache(
                clusters_df,
                gene_mappings,
                args
            )
            # Store cache in args for refinement
            args._gene_mappings_cache = gene_mappings
        except Exception as e:
            logger.warning(f"Cache-based duplication check failed: {e}")
            logger.warning("Falling back to full miniprot run")
            clusters_df = check_core_gene_duplications(
                clusters_df,
                fragments_dict,
                args,
                target_coverage_threshold=0.50,
                identity_threshold=0.35,
                use_header_cache=False
            )
    else:
        # No gene mappings available - run miniprot now
        # This happens when auto-resolution is disabled
        clusters_df = check_core_gene_duplications(
            clusters_df,
            fragments_dict,
            args,
            target_coverage_threshold=0.50,
            identity_threshold=0.35,
            use_header_cache=False
        )

    # Save bins before refinement if requested
    save_bins_before_refinement = getattr(args, "save_bins_before_refinement", False)
    if save_bins_before_refinement:
        logger.info("Saving bins before refinement...")

        # Save bins.csv with '_before_refinement' suffix
        bins_csv_before_path = os.path.join(args.output, "bins_before_refinement.csv")
        bins_before_df = clusters_df[clusters_df["cluster"] != "noise"].copy()
        bins_before_df = bins_before_df[["contig", "cluster"]]
        bins_before_df.to_csv(bins_csv_before_path, index=False)
        logger.info(f"Saved bins_before_refinement.csv with {len(bins_before_df)} contigs")

        # Save FASTA files with '_before_refinement' in directory name
        bins_before_dir = os.path.join(args.output, "bins_before_refinement")
        os.makedirs(bins_before_dir, exist_ok=True)

        # Temporarily change output directory for FASTA saving
        original_output = args.output
        args.output = bins_before_dir

        try:
            valid_bins_before = save_clusters_as_fasta(clusters_df, fragments_dict, args)
            logger.info(f"Saved {len(valid_bins_before)} bins to {bins_before_dir}/")
        finally:
            # Restore original output directory
            args.output = original_output

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

    # Prepare refined bin assignments (excluding noise)
    logger.info("Preparing final bins.csv with refined cluster assignments...")
    bins_csv_path = os.path.join(args.output, "bins.csv")
    final_bins_df = clusters_df[clusters_df["cluster"] != "noise"].copy()
    # Keep only the first two columns: contig and cluster
    final_bins_df = final_bins_df[["contig", "cluster"]]

    valid_bins = save_clusters_as_fasta(clusters_df, fragments_dict, args)
    
    # Keep only contigs from valid bins (those that meet minimum size)
    filtered_bins_df = final_bins_df[final_bins_df["cluster"].isin(valid_bins)]
    filtered_bins_df.to_csv(bins_csv_path, index=False)
    logger.info(
        f"bins.csv saved with {len(filtered_bins_df)} contigs from {len(valid_bins)} valid bins"
    )

    logger.info("REMAG analysis completed successfully!")
