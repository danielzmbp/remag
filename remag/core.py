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
from .miniprot_utils import check_core_gene_duplications
from .refinement import refine_contaminated_bins
from .output import save_clusters_as_fasta


def main(args):
    setup_logging(args.output, verbose=args.verbose)
    os.makedirs(args.output, exist_ok=True)
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
    features_df, fragments_dict = get_features(
        input_fasta,  # Use filtered FASTA if bacterial filtering was applied
        args.bam,
        args.tsv,
        args.output,
        args.min_contig_length,
        args.cores,
        args.num_augmentations,
    )

    if features_df.empty:
        logger.error("No features generated. Exiting.")
        sys.exit(1)

    if not skip_bacterial_filter:
        base_name = os.path.basename(args.fasta)
        name_without_ext = os.path.splitext(base_name)[0]
        if name_without_ext.endswith(".gz"):
            name_without_ext = os.path.splitext(name_without_ext)[0]

        classification_results_path = os.path.join(
            args.output, f"{name_without_ext}_4cac_classification.tsv"
        )

    logger.info("Starting neural network training...")
    model = train_siamese_network(features_df, args)

    logger.info("Generating embeddings...")
    embeddings_df = generate_embeddings(model, features_df, args)
    embeddings_path = os.path.join(args.output, "embeddings.csv")
    if not os.path.exists(embeddings_path):
        embeddings_df.to_csv(embeddings_path)

    logger.info("Clustering contigs...")
    clusters_df = cluster_contigs(embeddings_df, fragments_dict, args)

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

    if refinement_summary:
        refinement_summary_path = os.path.join(args.output, "refinement_summary.json")
        with open(refinement_summary_path, "w", encoding="utf-8") as f:
            json.dump(refinement_summary, f, indent=2)

    logger.info("Saving final contig-to-bin mapping...")
    final_clusters_contigs_path = os.path.join(
        args.output, "final_clusters_contigs.csv"
    )

    # Save final contig-level cluster assignments (clusters_df is already contig-level)
    clusters_df.to_csv(final_clusters_contigs_path, index=False)
    logger.info(f"Saved final contig-level clusters to {final_clusters_contigs_path}")

    # Save clusters as FASTA files
    logger.info("Saving bins as FASTA files...")
    save_clusters_as_fasta(clusters_df, fragments_dict, args)

    logger.info("REMAG analysis completed successfully!")
