"""
Command Line Interface for REMAG
"""

import argparse
import glob
import os
import sys
from click.core import ParameterSource
from importlib.metadata import version

import rich_click as click

__version__ = version("remag")

PRECOMPUTED_COVERAGE_SUFFIXES = (
    ".tsv",
    ".tsv.gz",
    ".txt",
    ".txt.gz",
    ".cov",
    ".cov.gz",
    ".bedgraph",
    ".bedgraph.gz",
    ".bg",
    ".bg.gz",
)


def is_alignment_coverage_file(file_path):
    return file_path.lower().endswith((".bam", ".cram"))


def is_precomputed_coverage_file(file_path):
    return file_path.lower().endswith(PRECOMPUTED_COVERAGE_SUFFIXES)


def run_remag(args):
    from .core import main as core_main

    return core_main(args)


def normalize_coverage_args(args):
    """Expand `-c/--coverage` followed by multiple paths into repeated options."""
    normalized_args = []
    i = 0

    while i < len(args):
        token = args[i]

        if token not in {"-c", "--coverage"}:
            normalized_args.append(token)
            i += 1
            continue

        normalized_args.append(token)
        i += 1

        if i >= len(args) or args[i].startswith("-"):
            continue

        normalized_args.append(args[i])
        i += 1

        while i < len(args) and not args[i].startswith("-"):
            normalized_args.extend([token, args[i]])
            i += 1

    return normalized_args


class CoverageAwareCommand(click.RichCommand):
    """Click command that normalizes multi-value coverage arguments."""

    def parse_args(self, ctx, args):
        return super().parse_args(ctx, normalize_coverage_args(args))


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


class SpaceSeparatedFloats(click.ParamType):
    """Custom click type that accepts space-separated floats."""

    name = "floats"

    def convert(self, value, param, ctx):
        if value is None:
            return value

        if isinstance(value, (list, tuple)):
            return value

        try:
            return [float(x) for x in value.split()]
        except ValueError:
            self.fail(f"Invalid float list: {value}", param, ctx)


click.rich_click.TEXT_MARKUP = "markdown"
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
            "options": [
                "--epochs",
                "--batch-size",
                "--embedding-dim",
                "--base-learning-rate",
                "--max-positive-pairs",
                "--num-augmentations",
            ],
        },
        {
            "name": "Clustering",
            "options": [
                "--min-cluster-size",
                "--greedy-resolutions",
                "--leiden-k-neighbors",
                "--leiden-similarity-threshold",
            ],
        },
        {
            "name": "Filtering & Processing",
            "options": [
                "--min-contig-length",
                "--min-bin-size",
                "--coverage-batch-size",
                "--hyenadna-batch-size",
                "--skip-bacterial-filter",
                "--save-filtered-contigs",
                "--skip-rescue",
                "--rescue-max-duplication-increase",
                "--rescue-max-total-duplication",
            ],
        },
    ]
}


def custom_help_callback(ctx, param, value):
    """
    Custom help callback that shows different content for -h vs --help.

    -h: Shows only Input/Output and General options (quick reference)
    --help: Shows all options (full documentation)
    """
    if not value or ctx.resilient_parsing:
        return

    # Detect which flag was used
    show_basic_help = "-h" in sys.argv and "--help" not in sys.argv

    if show_basic_help:
        # Basic options to show
        basic_option_names = {
            "fasta",
            "fasta_arg",
            "coverage",
            "output",
            "threads",
            "verbose",
            "keep_intermediate",
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
            elif p.name in ["help", "version"]:
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
        if is_alignment_coverage_file(file_path):
            bam_cram_files.append(file_path)
        elif is_precomputed_coverage_file(file_path):
            tsv_files.append(file_path)
        else:
            raise click.BadParameter(
                f"Unsupported coverage file format: {file_path}. Supported formats: BAM, CRAM, TSV/TXT, COV, bedGraph"
            )

    # Don't allow mixing alignment files with precomputed coverage files
    if bam_cram_files and tsv_files:
        raise click.BadParameter(
            "Cannot mix BAM/CRAM files with precomputed coverage files. Use either alignment files or precomputed coverage files, not both."
        )

    return flattened_files


@click.command(name="remag", cls=CoverageAwareCommand)
@click.option(
    "--help",
    "-h",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=custom_help_callback,
    help="Show this message and exit.",
)
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
    help="Coverage files for calculation. Supports BAM/CRAM alignment files, contig-level TSV/TXT coverage, and interval COV/bedGraph coverage. Each file represents one sample. Supports space-separated paths and glob patterns (e.g., `*.bam`, `*.cram`, `*.tsv`, `*.cov.gz`).",
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
    default=2048,
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
    default=5e-3,
    show_default=True,
    help="Base learning rate for contrastive learning training (scaled by batch size).",
)
@click.option(
    "--barlow-lambda",
    type=float,
    default=None,
    show_default=False,
    help="Lambda parameter for Barlow Twins loss (redundancy reduction term). "
    "Default (auto): 0.003 for single/no coverage, 0.02 for multi-sample/coassembly.",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(
        ["metagenomics", "single-cell", "short-reads", "sr"], case_sensitive=False
    ),
    default="metagenomics",
    show_default=True,
    help="Preset mode adjusting defaults. 'metagenomics' (default) maximizes clusters; "
    "'single-cell' uses larger k-NN and minimizes clusters; 'short-reads'/'sr' enforces 1000bp min length.",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible training. Same seed produces identical models with same data.",
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
    default=None,
    show_default=False,
    help="Minimum contig length in base pairs for binning consideration. "
    "Default: 1000 for single-sample, 4096 for multi-sample/coassembly. "
    "User-specified values override auto-detection.",
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
    default=500000,
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
    "--skip-rescue",
    is_flag=True,
    help="Skip post-refinement bin rescue strategy (merging fragmented bins).",
)
@click.option(
    "--rescue-max-duplication-increase",
    type=float,
    default=5.0,
    show_default=True,
    help="Maximum allowed increase in duplication percentage when merging bins during rescue (e.g. 5.0 allows 0% -> 5%).",
)
@click.option(
    "--rescue-max-total-duplication",
    type=float,
    default=5.0,
    show_default=True,
    help=(
        "Maximum allowed TOTAL duplication percentage after merging bins during "
        "rescue (hard-capped at 10% for bin-to-bin merges)."
    ),
)
@click.option(
    "--num-augmentations",
    type=int,
    default=8,
    show_default=True,
    help="Number of random fragments per contig for data augmentation.",
)
@click.option(
    "--greedy-resolutions",
    type=SpaceSeparatedFloats(),
    default="0.1 0.5 1.0 2.0 5.0",
    show_default=True,
    help="Space-separated list of Leiden resolutions to try in greedy clustering.",
)
@click.option(
    "--leiden-k-neighbors",
    type=int,
    default=None,
    show_default=False,
    help="Number of nearest neighbors for k-NN graph construction in Leiden clustering. "
    "If not set, defaults to 15 (metagenomics) or 30 (single-cell mode).",
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
    help="Keep intermediate files such as features, model weights, and graph artifacts. By default, only core outputs are kept.",
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
@click.option(
    "--filter-only",
    is_flag=True,
    default=False,
    help="Only run eukaryotic filtering and write filtered FASTA; skip feature generation, training, and binning.",
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
    barlow_lambda,
    mode,
    random_seed,
    min_cluster_size,
    min_contig_length,
    max_positive_pairs,
    threads,
    min_bin_size,
    verbose,
    skip_bacterial_filter,
    save_filtered_contigs,
    skip_rescue,
    rescue_max_duplication_increase,
    rescue_max_total_duplication,
    num_augmentations,
    greedy_resolutions,
    leiden_k_neighbors,
    leiden_similarity_threshold,
    keep_intermediate,
    coverage_batch_size,
    hyenadna_batch_size,
    filter_only,
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
    - `embeddings.csv` - Learned contig embeddings
    """
    # Handle fasta input: accept either positional argument or --fasta flag
    if fasta is None and fasta_arg is None:
        raise click.UsageError(
            "Missing input FASTA file. Provide it as a positional argument or use -f/--fasta"
        )

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
        click.echo(
            f"Warning: Requested {threads} threads but only {max_cores} available. Using {max_cores}.",
            err=True,
        )
        threads = max_cores

    # Separate coverage files by type
    bam_cram_files = []
    tsv_files = []

    if coverage:
        for file_path in coverage:
            if is_alignment_coverage_file(file_path):
                bam_cram_files.append(file_path)
            elif is_precomputed_coverage_file(file_path):
                tsv_files.append(file_path)

    # Set default Barlow Twins lambda based on number of samples if not provided
    coverage_count = len(bam_cram_files) if bam_cram_files else len(tsv_files)
    if barlow_lambda is None:
        if coverage_count > 1:
            barlow_lambda = 0.02
            click.echo(
                "Auto Barlow lambda: 0.02 (multi-sample/coassembly detected)", err=True
            )
        else:
            barlow_lambda = 0.003
            click.echo(
                "Auto Barlow lambda: 0.003 (single-sample or no coverage)", err=True
            )

    # Set default base learning rate based on number of samples if not provided
    ctx = click.get_current_context()
    base_lr_source = (
        ctx.get_parameter_source("base_learning_rate")
        if ctx is not None
        else ParameterSource.DEFAULT
    )
    if coverage_count > 1 and base_lr_source == ParameterSource.DEFAULT:
        base_learning_rate = 0.0005
        click.echo(
            "Coassembly detected: Auto base learning rate set to 0.0005.", err=True
        )

    # Set default min contig length if not provided by user
    if min_contig_length is None:
        if mode.lower() in ["short-reads", "sr"]:
            min_contig_length = 1000
            click.echo(
                "Short-reads mode: Auto-setting min contig length to 1000 bp.", err=True
            )
        elif coverage_count > 1:
            min_contig_length = 4096
            click.echo(
                "Coassembly detected: Auto-setting min contig length to 4096 bp.",
                err=True,
            )
        else:
            min_contig_length = 1000

    # Mode-specific defaults
    effective_k = leiden_k_neighbors
    if effective_k is None:
        effective_k = 30 if mode.lower() == "single-cell" else 15
    skip_bacterial_filter_mode = skip_bacterial_filter or mode.lower() == "single-cell"

    if mode.lower() == "single-cell":
        if not skip_bacterial_filter:
            click.echo(
                "Single-cell mode: skipping euk filter (keeping all contigs).", err=True
            )

    args = argparse.Namespace(
        fasta=fasta_path,
        bam=bam_cram_files if bam_cram_files else None,
        tsv=tsv_files if tsv_files else None,
        output=output,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        base_learning_rate=base_learning_rate,
        barlow_lambda=barlow_lambda,
        mode=mode.lower(),
        random_seed=random_seed,
        min_cluster_size=min_cluster_size,
        min_contig_length=min_contig_length,
        max_positive_pairs=max_positive_pairs,
        cores=threads,
        min_bin_size=min_bin_size,
        verbose=verbose,
        skip_bacterial_filter=skip_bacterial_filter_mode,
        save_filtered_contigs=save_filtered_contigs,
        skip_rescue=skip_rescue,
        rescue_max_duplication_increase=rescue_max_duplication_increase,
        rescue_max_total_duplication=rescue_max_total_duplication,
        num_augmentations=num_augmentations,
        greedy_resolutions=greedy_resolutions,
        leiden_k_neighbors=effective_k,
        leiden_similarity_threshold=leiden_similarity_threshold,
        keep_intermediate=keep_intermediate,
        coverage_batch_size=coverage_batch_size,
        hyenadna_batch_size=hyenadna_batch_size,
        filter_only=filter_only,
    )
    run_remag(args)


def main():
    main_cli()
