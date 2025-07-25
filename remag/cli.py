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
click.rich_click.OPTION_GROUPS = {
    "remag": [
        {
            "name": "Input/Output",
            "options": ["--fasta", "--bam", "--tsv", "--output"],
        },
        {
            "name": "Contrastive Learning",
            "options": ["--epochs", "--batch-size", "--embedding-dim", "--nce-temperature", "--base-learning-rate", "--max-positive-pairs", "--num-augmentations"],
        },
        {
            "name": "Clustering",
            "options": ["--min-cluster-size", "--min-samples", "--cluster-selection-epsilon"],
        },
        {
            "name": "Filtering & Processing",
            "options": ["--min-contig-length", "--min-bin-size", "--skip-bacterial-filter", "--skip-refinement", "--max-refinement-rounds", "--skip-chimera-detection", "--keep-intermediate"],
        },
        {
            "name": "General",
            "options": ["--cores", "--verbose"],
        },
    ]
}


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
    help="Input FASTA file containing contigs to bin into genomes. Supports gzipped files.",
)
@click.option(
    "-b",
    "--bam",
    type=SpaceSeparatedPaths(),
    multiple=True,
    callback=validate_coverage_options,
    help="Indexed BAM files for coverage calculation. Each file represents one sample. Supports space-separated paths and glob patterns (e.g., '*.bam').",
)
@click.option(
    "-t",
    "--tsv",
    multiple=True,
    type=click.Path(exists=True, readable=True, path_type=str),
    callback=validate_coverage_options,
    help="Pre-computed coverage data in TSV format. Alternative to BAM files.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(path_type=str),
    help="Output directory for binning results and intermediate files.",
)
@click.option(
    "--epochs",
    type=click.IntRange(min=50, max=2000),
    default=400,
    show_default=True,
    help="Number of training epochs for contrastive learning model.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=64, max=8192),
    default=2048,
    show_default=True,
    help="Batch size for contrastive learning training.",
)
@click.option(
    "--embedding-dim",
    type=click.IntRange(min=64, max=512),
    default=256,
    show_default=True,
    help="Dimensionality of contig embeddings in contrastive learning.",
)
@click.option(
    "--base-learning-rate",
    type=click.FloatRange(min=1e-5, max=0.1),
    default=8e-3,
    show_default=True,
    help="Base learning rate for contrastive learning training (scaled by batch size).",
)
@click.option(
    "--min-cluster-size",
    type=click.IntRange(min=2, max=100),
    default=2,
    show_default=True,
    help="Minimum number of contigs required to form a cluster/bin.",
)
@click.option(
    "--min-samples",
    type=click.IntRange(min=1, max=100),
    default=None,
    show_default=True,
    help="Minimum samples for HDBSCAN core points. If None, uses min-cluster-size.",
)
@click.option(
    "--min-contig-length",
    type=click.IntRange(min=500, max=10000),
    default=1000,
    show_default=True,
    help="Minimum contig length in base pairs for binning consideration.",
)
@click.option(
    "--max-positive-pairs",
    type=click.IntRange(min=100000, max=10000000),
    default=5000000,
    show_default=True,
    help="Maximum number of positive pairs for contrastive learning training.",
)
@click.option(
    "-c",
    "--cores",
    type=click.IntRange(min=1, max=64),
    default=8,
    show_default=True,
    help="Number of CPU cores to use for parallel processing.",
)
@click.option(
    "--min-bin-size",
    type=click.IntRange(min=50000, max=10000000),
    default=100000,
    show_default=True,
    help="Minimum total bin size in base pairs for output.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed logging output.")
@click.option(
    "--skip-bacterial-filter",
    is_flag=True,
    help="Skip bacterial contig filtering using 4CAC classifier and contrastive learning.",
)
@click.option(
    "--skip-refinement",
    is_flag=True,
    help="Skip post-clustering bin refinement and optimization.",
)
@click.option(
    "--max-refinement-rounds",
    type=click.IntRange(min=1, max=10),
    default=2,
    show_default=True,
    help="Maximum number of iterative bin refinement rounds.",
)
@click.option(
    "--num-augmentations",
    type=click.IntRange(min=1, max=32),
    default=8,
    show_default=True,
    help="Number of random fragments per contig for data augmentation.",
)
@click.option(
    "--enable-chimera-detection",
    is_flag=True,
    default=False,
    help="Enable chimeric contig detection and splitting for large contigs.",
)
@click.option(
    "--cluster-selection-epsilon",
    type=click.FloatRange(min=0, max=1.0),
    default=0,
    show_default=True,
    help="HDBSCAN cluster selection epsilon for reachability-based clustering (higher = more flexible clustering).",
)
@click.option(
    "--keep-intermediate",
    is_flag=True,
    default=False,
    help="Keep intermediate files (embeddings, features, model, etc.). By default, only bins.csv and bins/ folder are kept.",
)
@click.option(
    "--skip-kmeans-filtering",
    is_flag=True,
    default=False,
    help="Skip k-means pre-filtering to remove small, low-confidence clusters.",
)
def main_cli(
    fasta,
    bam,
    tsv,
    output,
    epochs,
    batch_size,
    embedding_dim,
    base_learning_rate,
    min_cluster_size,
    min_samples,
    min_contig_length,
    max_positive_pairs,
    cores,
    min_bin_size,
    verbose,
    skip_bacterial_filter,
    skip_refinement,
    max_refinement_rounds,
    num_augmentations,
    enable_chimera_detection,
    cluster_selection_epsilon,
    keep_intermediate,
    skip_kmeans_filtering,
):
    """REMAG: Recovery of eukaryotic genomes using contrastive learning."""
    # Validate resource parameters
    import multiprocessing
    import psutil
    
    max_cores = multiprocessing.cpu_count()
    if cores > max_cores:
        click.echo(f"Warning: Requested {cores} cores but only {max_cores} available. Using {max_cores}.", err=True)
        cores = max_cores
    
    # Check available memory for batch size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    estimated_memory_per_batch = batch_size * 0.001  # Rough estimate: 1MB per sample
    if estimated_memory_per_batch > available_memory_gb * 0.8:  # Use max 80% of available memory
        recommended_batch_size = int((available_memory_gb * 0.8) / 0.001)
        click.echo(f"Warning: Batch size {batch_size} may exceed available memory ({available_memory_gb:.1f}GB). "
                   f"Consider reducing to {recommended_batch_size}.", err=True)
    args = argparse.Namespace(
        fasta=fasta,
        bam=bam if bam else None,
        tsv=list(tsv) if tsv else None,
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        base_learning_rate=base_learning_rate,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_contig_length=min_contig_length,
        max_positive_pairs=max_positive_pairs,
        cores=cores,
        min_bin_size=min_bin_size,
        verbose=verbose,
        skip_bacterial_filter=skip_bacterial_filter,
        skip_refinement=skip_refinement,
        max_refinement_rounds=max_refinement_rounds,
        num_augmentations=num_augmentations,
        skip_chimera_detection=not enable_chimera_detection,
        cluster_selection_epsilon=cluster_selection_epsilon,
        keep_intermediate=keep_intermediate,
        skip_kmeans_filtering=skip_kmeans_filtering,
    )
    run_remag(args)


def main():
    main_cli()
