"""
Command Line Interface for REMAG
"""

import argparse
import glob
import os
import sys
import rich_click as click
from .core import main as run_remag

from importlib.metadata import version

__version__ = version("remag")


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
            "options": ["--fasta", "--coverage", "--output"],
        },
        {
            "name": "General",
            "options": ["--threads", "--verbose", "--keep-intermediate"],
        },
        {
            "name": "Contrastive Learning",
            "options": ["--epochs", "--batch-size", "--embedding-dim", "--base-learning-rate", "--max-positive-pairs", "--num-augmentations"],
        },
        {
            "name": "Clustering",
            "options": ["--min-cluster-size", "--leiden-resolution", "--auto-resolution", "--leiden-k-neighbors", "--leiden-similarity-threshold"],
        },
        {
            "name": "Filtering & Processing",
            "options": ["--min-contig-length", "--min-bin-size", "--coverage-batch-size", "--hyenadna-batch-size", "--skip-bacterial-filter", "--save-filtered-contigs", "--skip-refinement", "--save-bins-before-refinement", "--max-refinement-rounds", "--min-duplications-for-refinement", "--skip-chimera-detection"],
        },
    ]
}

# Store full option groups for restoration
_FULL_OPTION_GROUPS = click.rich_click.OPTION_GROUPS.copy()


def custom_help_callback(ctx, param, value):
    """
    Custom help callback that shows different content for -h vs --help.

    -h: Shows only Input/Output and General options (quick reference)
    --help: Shows all options (full documentation)
    """
    if not value or ctx.resilient_parsing:
        return

    # Detect which flag was used
    show_basic_help = '-h' in sys.argv and '--help' not in sys.argv

    if show_basic_help:
        # Basic options to show
        basic_option_names = {
            'fasta', 'fasta_arg', 'coverage', 'output',
            'threads', 'verbose', 'keep_intermediate'
        }

        # Store original docstring and replace with minimal version
        original_doc = ctx.command.help
        ctx.command.help = (
            "**REMAG**: Recovery of Eukaryotic Metagenome-Assembled Genomes\n\n"
            "A specialized metagenomic binning tool for recovering high-quality eukaryotic genomes "
            "from mixed prokaryotic-eukaryotic samples using contrastive learning."
        )

        # Temporarily filter parameters to only show basic ones
        original_params = ctx.command.params
        basic_params = []

        for p in original_params:
            # Keep all arguments and the help/version options
            if not isinstance(p, click.Option):
                basic_params.append(p)
            elif p.name in ['help', 'version']:
                basic_params.append(p)
            elif p.name in basic_option_names:
                basic_params.append(p)

        # Temporarily replace params with filtered list
        ctx.command.params = basic_params

        # Get help text with only basic parameters and minimal docstring
        help_text = ctx.get_help()

        # Restore original parameters and docstring
        ctx.command.params = original_params
        ctx.command.help = original_doc

        click.echo(help_text, color=ctx.color)
        click.echo("\n💡 Use 'remag --help' to see all advanced options")
    else:
        # Show full help for --help
        click.echo(ctx.get_help(), color=ctx.color)

    ctx.exit()


def validate_coverage_options(ctx, param, value):
    """Validate and categorize coverage files by extension."""
    if param.name != "coverage" or not value:
        return value

    # Flatten the list of lists that might come from multiple -c calls
    flattened_files = []
    for item in value:
        if isinstance(item, list):
            flattened_files.extend(item)
        else:
            flattened_files.append(item)

    # Categorize files by extension
    bam_cram_files = []
    tsv_files = []

    for file_path in flattened_files:
        ext = file_path.lower().split('.')[-1]
        if ext in ['bam', 'cram']:
            bam_cram_files.append(file_path)
        elif ext in ['tsv', 'txt']:
            tsv_files.append(file_path)
        else:
            raise click.BadParameter(f"Unsupported coverage file format: {file_path}. Supported formats: BAM, CRAM, TSV")

    # Don't allow mixing BAM/CRAM with TSV files
    if bam_cram_files and tsv_files:
        raise click.BadParameter("Cannot mix BAM/CRAM files with TSV files. Use either alignment files or pre-computed coverage files, not both.")

    return flattened_files


@click.command(name="remag")
@click.option("--help", "-h", is_flag=True, expose_value=False, is_eager=True, callback=custom_help_callback, help="Show this message and exit.")
@click.version_option(version=__version__, prog_name="REMAG")
@click.argument(
    "fasta_arg",
    type=click.Path(exists=True, readable=True, path_type=str),
    required=False,
)
@click.option(
    "-f",
    "--fasta",
    required=False,
    type=click.Path(exists=True, readable=True, path_type=str),
    help="Input FASTA file containing contigs to bin into genomes. Supports gzipped files.",
)
@click.option(
    "-c",
    "--coverage",
    type=SpaceSeparatedPaths(),
    multiple=True,
    callback=validate_coverage_options,
    help="Coverage files for calculation. Supports BAM, CRAM (indexed), and TSV formats. Each file represents one sample. Auto-detects format by extension. Supports space-separated paths and glob patterns (e.g., '*.bam', '*.cram', '*.tsv').",
)
@click.option(
    "-o",
    "--output",
    required=False,
    default=None,
    type=click.Path(path_type=str),
    help="Output directory for binning results and intermediate files. If not specified, creates 'remag_output' in the same directory as the input FASTA.",
)
@click.option(
    "--epochs",
    type=int,
    default=400,
    show_default=True,
    help="Number of training epochs for contrastive learning model.",
)
@click.option(
    "--batch-size",
    type=int,
    default=4096,
    show_default=True,
    help="Batch size for contrastive learning training.",
)
@click.option(
    "--embedding-dim",
    type=int,
    default=256,
    show_default=True,
    help="Dimensionality of contig embeddings in contrastive learning.",
)
@click.option(
    "--base-learning-rate",
    type=float,
    default=0.0025,
    show_default=True,
    help="Base learning rate for contrastive learning training (scaled by batch size).",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    show_default=True,
    help="Minimum number of contigs required to form a cluster/bin.",
)
@click.option(
    "--min-contig-length",
    type=int,
    default=4096,
    show_default=True,
    help="Minimum contig length in base pairs for binning consideration.",
)
@click.option(
    "--max-positive-pairs",
    type=int,
    default=5000000,
    show_default=True,
    help="Maximum number of positive pairs for contrastive learning training.",
)
@click.option(
    "-t",
    "--threads",
    type=int,
    default=8,
    show_default=True,
    help="Number of CPU cores to use for parallel processing.",
)
@click.option(
    "--min-bin-size",
    type=int,
    default=300000,
    show_default=True,
    help="Minimum total bin size in base pairs for output.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed logging output.")
@click.option(
    "--skip-bacterial-filter",
    is_flag=True,
    help="Skip eukaryotic contig filtering using HyenaDNA classifier (keeps all contigs).",
)
@click.option(
    "--save-filtered-contigs",
    is_flag=True,
    help="Save non-eukaryotic (filtered out) contigs to a separate FASTA file in the output directory.",
)
@click.option(
    "--skip-refinement",
    is_flag=True,
    help="Skip post-clustering bin refinement and optimization.",
)
@click.option(
    "--save-bins-before-refinement",
    is_flag=True,
    help="Save bins (bins.csv and FASTA files) before refinement step, with '_before_refinement' suffix.",
)
@click.option(
    "--max-refinement-rounds",
    type=int,
    default=16,
    show_default=True,
    help="Maximum number of iterative bin refinement rounds.",
)
@click.option(
    "--min-duplications-for-refinement",
    type=int,
    default=1,
    show_default=True,
    help="Minimum number of duplicated core genes required to trigger refinement.",
)
@click.option(
    "--num-augmentations",
    type=int,
    default=4,
    show_default=True,
    help="Number of random fragments per contig for data augmentation.",
)
@click.option(
    "--skip-chimera-detection",
    is_flag=True,
    default=True,
    help="Skip chimeric contig detection and splitting for large contigs (default: skip).",
)
@click.option(
    "--leiden-resolution",
    type=float,
    default=None,
    show_default=False,
    help="Resolution parameter for Leiden clustering (higher = more clusters). If not specified, auto-resolution is used to determine optimal value based on core gene duplications.",
)
@click.option(
    "--auto-resolution",
    is_flag=True,
    default=True,
    show_default=True,
    help="Automatically determine optimal Leiden resolution based on core gene duplications (enabled by default). Disabled if --leiden-resolution is specified.",
)
@click.option(
    "--leiden-k-neighbors",
    type=int,
    default=15,
    show_default=True,
    help="Number of nearest neighbors for k-NN graph construction in Leiden clustering.",
)
@click.option(
    "--leiden-similarity-threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum cosine similarity threshold for k-NN graph edges in Leiden clustering.",
)
@click.option(
    "-k",
    "--keep-intermediate",
    is_flag=True,
    default=False,
    help="Keep intermediate files (embeddings, features, model, etc.). By default, only bins.csv and bins/ folder are kept.",
)
@click.option(
    "--coverage-batch-size",
    type=int,
    default=100000,
    show_default=True,
    help="Number of contigs to process per batch when calculating coverage from alignment files. Reduce this value if running out of memory with very large datasets.",
)
@click.option(
    "--hyenadna-batch-size",
    type=int,
    default=1024,
    show_default=True,
    help="Batch size for HyenaDNA model inference. Higher values speed up GPU inference but use more VRAM. Use 2048-4096 for high-end GPUs.",
)
def main_cli(
    fasta_arg,
    fasta,
    coverage,
    output,
    epochs,
    batch_size,
    embedding_dim,
    base_learning_rate,
    min_cluster_size,
    min_contig_length,
    max_positive_pairs,
    threads,
    min_bin_size,
    verbose,
    skip_bacterial_filter,
    save_filtered_contigs,
    skip_refinement,
    save_bins_before_refinement,
    max_refinement_rounds,
    min_duplications_for_refinement,
    num_augmentations,
    skip_chimera_detection,
    auto_resolution,
    leiden_resolution,
    leiden_k_neighbors,
    leiden_similarity_threshold,
    keep_intermediate,
    coverage_batch_size,
    hyenadna_batch_size,
):
    """
    **REMAG**: Recovery of Eukaryotic Metagenome-Assembled Genomes

    A specialized metagenomic binning tool for recovering high-quality eukaryotic genomes
    from mixed prokaryotic-eukaryotic samples using contrastive learning.

    ## Basic Usage

    Single sample:
    ```
    remag contigs.fasta -c alignments.bam
    ```

    Multiple samples (glob patterns):
    ```
    remag contigs.fasta -c "samples/*.bam" -o output_dir
    ```

    ## Output

    - `bins/` - Directory containing binned FASTA files (one per genome)
    - `bins.csv` - Cluster assignments for all contigs
    - `embeddings.csv` - Learned contig embeddings (if --keep-intermediate)
    """
    # Handle fasta input: accept either positional argument or --fasta flag
    if fasta is None and fasta_arg is None:
        raise click.UsageError("Missing input FASTA file. Provide it as a positional argument or use -f/--fasta")

    # Prefer --fasta flag if both are provided
    if fasta is not None:
        fasta_path = fasta
    else:
        fasta_path = fasta_arg

    # Handle output directory: default to 'remag_output' in same directory as FASTA
    if output is None:
        fasta_dir = os.path.dirname(os.path.abspath(fasta_path))
        output = os.path.join(fasta_dir, "remag_output")
        click.echo(f"Output directory not specified, using: {output}", err=True)

    # Validate resource parameters
    import multiprocessing

    max_cores = multiprocessing.cpu_count()
    if threads > max_cores:
        click.echo(f"Warning: Requested {threads} threads but only {max_cores} available. Using {max_cores}.", err=True)
        threads = max_cores

    # Handle auto-resolution vs manual resolution logic
    if leiden_resolution is not None:
        # User explicitly specified resolution, disable auto-resolution
        auto_resolution = False
        click.echo(f"Using manual Leiden resolution: {leiden_resolution}", err=True)
    else:
        # No manual resolution specified, use auto-resolution (default)
        leiden_resolution = 1.0  # Fallback if auto-resolution fails
        if auto_resolution:
            click.echo("Auto-resolution enabled (default). Use --leiden-resolution to specify manually.", err=True)

    # Separate coverage files by type
    bam_cram_files = []
    tsv_files = []

    if coverage:
        for file_path in coverage:
            ext = file_path.lower().split('.')[-1]
            if ext in ['bam', 'cram']:
                bam_cram_files.append(file_path)
            elif ext in ['tsv', 'txt']:
                tsv_files.append(file_path)

    args = argparse.Namespace(
        fasta=fasta_path,
        bam=bam_cram_files if bam_cram_files else None,
        tsv=tsv_files if tsv_files else None,
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        base_learning_rate=base_learning_rate,
        min_cluster_size=min_cluster_size,
        min_contig_length=min_contig_length,
        max_positive_pairs=max_positive_pairs,
        cores=threads,
        min_bin_size=min_bin_size,
        verbose=verbose,
        skip_bacterial_filter=skip_bacterial_filter,
        save_filtered_contigs=save_filtered_contigs,
        skip_refinement=skip_refinement,
        save_bins_before_refinement=save_bins_before_refinement,
        max_refinement_rounds=max_refinement_rounds,
        min_duplications_for_refinement=min_duplications_for_refinement,
        num_augmentations=num_augmentations,
        skip_chimera_detection=skip_chimera_detection,
        auto_resolution=auto_resolution,
        leiden_resolution=leiden_resolution,
        leiden_k_neighbors=leiden_k_neighbors,
        leiden_similarity_threshold=leiden_similarity_threshold,
        keep_intermediate=keep_intermediate,
        coverage_batch_size=coverage_batch_size,
        hyenadna_batch_size=hyenadna_batch_size,
    )
    run_remag(args)


def main():
    main_cli()
