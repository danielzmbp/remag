"""
Core module for REMAG - Main execution logic
"""

import json
import os
import sys
from loguru import logger

from .utils import setup_logging
from .features import filter_bacterial_contigs, get_features, get_classification_results_path
from .models import train_siamese_network, generate_embeddings
from .clustering import cluster_contigs
from .miniprot_utils import check_core_gene_duplications
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

    # Apply bacterial filtering if not skipped
    input_fasta = args.fasta
    skip_bacterial_filter = getattr(args, "skip_bacterial_filter", False)
    if not skip_bacterial_filter:
        logger.info("Applying bacterial contig filtering using 4CAC classifier...")
        input_fasta = filter_bacterial_contigs(
            args.fasta,
            args.output,
            min_contig_length=args.min_contig_length,
            cores=args.cores,
        )
        logger.info(f"Using filtered FASTA file: {input_fasta}")
    else:
        logger.info("Skipping bacterial filtering as requested")

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

    if not skip_bacterial_filter:
        classification_results_path = get_classification_results_path(args.fasta, args.output)

    logger.info("Training neural network and generating embeddings...")
    try:
        model = train_siamese_network(features_df, args)
        embeddings_df = generate_embeddings(model, features_df, args)
    except Exception as e:
        logger.error(f"Failed to train model or generate embeddings: {e}")
        sys.exit(1)

    try:
        clusters_df = cluster_contigs(embeddings_df, fragments_dict, args)
    except Exception as e:
        logger.error(f"Failed to cluster contigs: {e}")
        sys.exit(1)

    # Check for duplicated core genes using miniprot
    logger.info("Checking for duplicated core genes...")
    clusters_df = check_core_gene_duplications(
        clusters_df, 
        fragments_dict, 
        args,
        target_coverage_threshold=0.50,
        identity_threshold=0.50,
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

    logger.info("Final bins saved to bins.csv (noise contigs excluded)")

    logger.info("Saving bins as FASTA files...")
    valid_bins = save_clusters_as_fasta(clusters_df, fragments_dict, args)
    
    # Filter bins.csv to only include contigs from valid bins
    logger.info("Filtering bins.csv to match saved bins...")
    bins_csv_path = os.path.join(args.output, "bins.csv")
    if os.path.exists(bins_csv_path):
        import pandas as pd
        bins_df = pd.read_csv(bins_csv_path)
        filtered_bins_df = bins_df[bins_df["cluster"].isin(valid_bins)]
        filtered_bins_df.to_csv(bins_csv_path, index=False)
        logger.info(f"bins.csv now contains {len(filtered_bins_df)} contigs from {len(valid_bins)} bins")

    logger.info("REMAG analysis completed successfully!")
