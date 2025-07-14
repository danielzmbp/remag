"""
Command Line Interface for REMAG
"""

import argparse
import glob
import os
import rich_click as click
from .core import main as run_remag
from .utils import setup_logging


class SpaceSeparatedPaths(click.ParamType):
    """Custom click type that accepts space-separated file paths."""
    name = "paths"
    
    def convert(self, value, param, ctx):
        if value is None:
            return value
            
        # If it's already a list (from multiple calls), flatten it
        if isinstance(value, (list, tuple)):
            all_files = []
            for item in value:
                all_files.extend(self.convert(item, param, ctx))
            return all_files
            
        # Split on spaces to handle space-separated paths
        paths = value.split()
        validated_paths = []
        
        for path in paths:
            # Handle glob patterns
            if "*" in path or "?" in path or "[" in path:
                matched_files = glob.glob(path)
                if not matched_files:
                    self.fail(f"No files match the pattern: {path}", param, ctx)
                validated_paths.extend(matched_files)
            else:
                # Validate individual file
                if not os.path.exists(path):
                    self.fail(f"File does not exist: {path}", param, ctx)
                if not os.path.isfile(path):
                    self.fail(f"Path is not a file: {path}", param, ctx)
                validated_paths.append(path)
        
        return sorted(validated_paths)


click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = (
    "Try running the '--help' flag for more information."
)


def validate_coverage_options(ctx, param, value):
    """Validate that either --bam or --tsv is provided, but not both."""
    if param.name == "tsv":
        bam = ctx.params.get("bam")
        if bam and value:
            raise click.BadParameter("--bam and --tsv are mutually exclusive.")
    elif param.name == "bam":
        tsv = ctx.params.get("tsv")
        if tsv and value:
            raise click.BadParameter("--bam and --tsv are mutually exclusive.")

        if value:
            # Flatten the list of lists that might come from multiple -b calls
            flattened_files = []
            for item in value:
                if isinstance(item, list):
                    flattened_files.extend(item)
                else:
                    flattened_files.append(item)
            return flattened_files

    return value


@click.command(name="remag")
@click.help_option("--help", "-h")
@click.version_option(version="0.1.0", prog_name="REMAG")
@click.option(
    "-f",
    "--fasta",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=str),
    help="Input FASTA file with contigs to bin. Can be gzipped.",
)
@click.option(
    "-b",
    "--bam",
    type=SpaceSeparatedPaths(),
    multiple=True,
    callback=validate_coverage_options,
    help="Input BAM file(s) for coverage calculation. Must be indexed. Each BAM represents a sample. Supports space-separated files or glob patterns (e.g., '*.bam', 'sample_*.bam').",
)
@click.option(
    "-t",
    "--tsv",
    multiple=True,
    type=click.Path(exists=True, readable=True, path_type=str),
    callback=validate_coverage_options,
    help="Input TSV file(s) with coverage information.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(path_type=str),
    help="Output directory for results.",
)
@click.option(
    "--epochs",
    type=click.IntRange(min=1, max=10000),
    default=400,
    show_default=True,
    help="Training epochs for neural network.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=32, max=8192),
    default=512,
    show_default=True,
    help="Batch size for training.",
)
@click.option(
    "--embedding-dim",
    type=click.IntRange(min=32, max=512),
    default=256,
    show_default=True,
    help="Embedding dimension for contrastive learning.",
)
@click.option(
    "--nce-temperature",
    type=click.FloatRange(min=0.01, max=1.0),
    default=0.07,
    show_default=True,
    help="Temperature for InfoNCE loss.",
)
@click.option(
    "--min-cluster-size",
    type=click.IntRange(min=1, max=1000),
    default=5,
    show_default=True,
    help="Minimum fragments per cluster.",
)
@click.option(
    "--min-samples",
    type=click.IntRange(min=1, max=1000),
    default=3,
    show_default=True,
    help="Minimum samples for HDBSCAN core points.",
)
@click.option(
    "--min-contig-length",
    type=click.IntRange(min=500, max=50000),
    default=1000,
    show_default=True,
    help="Minimum contig length in bp.",
)
@click.option(
    "--max-positive-pairs",
    type=click.IntRange(min=10000, max=50000000),
    default=5000000,
    show_default=True,
    help="Maximum positive pairs for contrastive learning.",
)
@click.option(
    "-c",
    "--cores",
    type=click.IntRange(min=1, max=128),
    default=8,
    show_default=True,
    help="Number of CPU cores.",
)
@click.option(
    "--min-bin-size",
    type=click.IntRange(min=50000, max=10000000),
    default=100000,
    show_default=True,
    help="Minimum bin size in bp.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--enable-preclustering/--disable-preclustering",
    default=True,
    help="Enable K-means pre-clustering to remove bacterial contigs before HDBSCAN.",
)
@click.option(
    "--skip-bacterial-filter",
    is_flag=True,
    help="Skip bacterial contig filtering (4CAC classifier + contrastive learning).",
)
@click.option(
    "--skip-refinement",
    is_flag=True,
    help="Skip bin refinement.",
)
@click.option(
    "--max-refinement-rounds",
    type=click.IntRange(min=1, max=5),
    default=2,
    show_default=True,
    help="Maximum refinement rounds.",
)
@click.option(
    "--num-augmentations",
    type=click.IntRange(min=0, max=64),
    default=8,
    show_default=True,
    help="Number of random fragments per contig.",
)
@click.option(
    "--skip-chimera-detection",
    is_flag=True,
    help="Skip chimera detection for large contigs.",
)
def main_cli(
    fasta,
    bam,
    tsv,
    output,
    epochs,
    batch_size,
    embedding_dim,
    nce_temperature,
    min_cluster_size,
    min_samples,
    min_contig_length,
    max_positive_pairs,
    cores,
    min_bin_size,
    verbose,
    enable_preclustering,
    skip_bacterial_filter,
    skip_refinement,
    max_refinement_rounds,
    num_augmentations,
    skip_chimera_detection,
):
    """REMAG: Recovery of eukaryotic genomes using contrastive learning."""
    args = argparse.Namespace(
        fasta=fasta,
        bam=bam if bam else None,
        tsv=list(tsv) if tsv else None,
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        nce_temperature=nce_temperature,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_contig_length=min_contig_length,
        max_positive_pairs=max_positive_pairs,
        cores=cores,
        min_bin_size=min_bin_size,
        verbose=verbose,
        enable_preclustering=enable_preclustering,
        skip_bacterial_filter=skip_bacterial_filter,
        skip_refinement=skip_refinement,
        max_refinement_rounds=max_refinement_rounds,
        num_augmentations=num_augmentations,
        skip_chimera_detection=skip_chimera_detection,
    )
    run_remag(args)


def main():
    main_cli()
