"""Miniprot utilities for gene mapping and core-gene duplication checks."""

import json
import os
import shutil
import subprocess

from loguru import logger
from tqdm import tqdm

from .utils import (
    ContigHeaderMapper,
    _write_fasta_record,
    initialize_duplication_columns,
)


def check_miniprot_available():
    """Check if miniprot is available in PATH."""
    return shutil.which("miniprot") is not None


def _build_miniprot_cmd(fasta_path, db_path, cores):
    """Build the miniprot command list."""
    return [
        "miniprot",
        "-I",
        "-t",
        str(cores),
        "--outs=0.95",
        fasta_path,
        db_path,
    ]


def _parse_paf_gene_mappings(
    paf_path,
    target_coverage_threshold,
    identity_threshold,
    gene_mappings=None,
):
    """Parse a miniprot PAF file into gene-to-contig mappings.

    Retain the best alignment metrics and merge overlapping target intervals for
    each (contig, gene family). Separate intervals represent separate copies;
    overlapping alternatives count once, including matches on opposite strands.
    An intron-spanning alignment is one interval. Supplied mappings are updated
    in place so several PAF files can be merged without counting hits twice.

    Returns:
        dict: {contig_name: {gene_family: {score, coverage, identity, loci}}}
    """
    if gene_mappings is None:
        gene_mappings = {}

    if not os.path.exists(paf_path) or os.path.getsize(paf_path) == 0:
        return gene_mappings

    with open(paf_path, "r") as paf_file:
        for line in paf_file:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue

            try:
                query_name = parts[0]  # Protein name
                query_length = int(parts[1])
                query_start = int(parts[2])
                query_end = int(parts[3])
                target_name = parts[5]  # Contig name
                target_start = int(parts[7])
                target_end = int(parts[8])
                matching_bases = int(parts[9])
                alignment_length = int(parts[10])
            except (ValueError, IndexError):
                continue

            # Extract gene family code from BUSCO-style protein name
            # (e.g. "28947at2759" from "28947at2759_6832_0:00088a")
            gene_family_code = query_name.split()[0].split("_")[0]

            query_coverage = (
                (query_end - query_start) / query_length if query_length > 0 else 0
            )
            identity = matching_bases / alignment_length if alignment_length > 0 else 0

            if (
                not 0 <= target_start < target_end
                or query_coverage < target_coverage_threshold
                or identity < identity_threshold
            ):
                continue

            score = query_coverage * identity
            contig_genes = gene_mappings.setdefault(target_name, {})
            existing = contig_genes.setdefault(
                gene_family_code, {"score": -1, "loci": []}
            )
            intervals = sorted(existing["loci"] + [[target_start, target_end]])
            loci = []
            for start, end in intervals:
                if loci and start < loci[-1][1]:
                    loci[-1][1] = max(loci[-1][1], end)
                else:
                    loci.append([start, end])
            existing["loci"] = loci
            if score > existing["score"]:
                existing.update(score=score, coverage=query_coverage, identity=identity)

    return gene_mappings


def _get_gene_duplication_stats(contigs, gene_mappings):
    """Report locus counts without changing the presence counts used for binning."""
    counts = {}
    within_contig = {}
    for contig in contigs:
        for gene, info in gene_mappings.get(contig, {}).items():
            copies = len(info["loci"]) if "loci" in info else 1
            counts[gene] = counts.get(gene, 0) + copies
            if copies > 1:
                within_contig.setdefault(contig, {})[gene] = copies
    duplicated = {gene: count for gene, count in counts.items() if count > 1}
    return {
        "has_duplications": bool(duplicated),
        "duplicated_genes": duplicated,
        "total_genes_found": len(counts),
        "single_copy_genes_count": sum(count == 1 for count in counts.values()),
        "within_contig_duplications": within_contig,
    }


def load_or_generate_gene_mappings(
    fragments_dict, args, target_coverage_threshold=0.60, identity_threshold=0.40
):
    """Load cached gene mappings or generate them with one miniprot run.

    Args:
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object containing output directory, cores, etc.
        target_coverage_threshold: Minimum target coverage for alignments
        identity_threshold: Minimum identity for alignments

    Returns:
        dict: {contig_name: {gene_family: alignment details}}
    """
    cache_path = get_gene_mappings_cache_path(args)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                gene_mappings = json.load(f)
            if any(
                "loci" not in info
                for genes in gene_mappings.values()
                for info in genes.values()
            ):
                raise ValueError("Cached gene mappings lack gene positions")
            logger.info(
                f"Loaded cached miniprot gene mappings for {len(gene_mappings)} contigs"
            )
            return gene_mappings
        except Exception as e:
            logger.warning(f"Failed to load miniprot cache, will re-run: {e}")

    logger.info(f"Running miniprot on {len(fragments_dict)} contigs...")

    # Check if miniprot is available
    if not check_miniprot_available():
        logger.error("miniprot not found in PATH - cannot generate gene mappings")
        logger.error("Install miniprot with: conda install -c bioconda miniprot")
        raise RuntimeError("miniprot not found in PATH")

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "db", "refseq_db.faa.gz"
    )
    if not os.path.exists(db_path):
        raise RuntimeError(f"Eukaryotic database not found: {db_path}")

    # Create temporary directory
    temp_dir = os.path.join(args.output, "temp_gene_mapping")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create a single FASTA file with all contigs
        all_contigs_fasta = os.path.join(temp_dir, "all_contigs.fa")
        with open(all_contigs_fasta, "w") as f:
            for header, data in fragments_dict.items():
                _write_fasta_record(f, header, data["sequence"])

        # Run miniprot
        miniprot_output = os.path.join(temp_dir, "all_contigs.paf")
        miniprot_stderr = os.path.join(temp_dir, "all_contigs.stderr")

        cmd_list = _build_miniprot_cmd(all_contigs_fasta, db_path, args.cores)

        if args.verbose:
            logger.debug(f"Running miniprot command: {' '.join(cmd_list)}")

        with (
            open(miniprot_output, "w") as stdout_file,
            open(miniprot_stderr, "w") as stderr_file,
        ):
            process = subprocess.run(
                cmd_list,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=14400,  # 4 hour timeout
                check=False,
            )
            result = process.returncode

        if result != 0:
            logger.error(f"miniprot failed with exit code {result}")
            if os.path.exists(miniprot_stderr) and os.path.getsize(miniprot_stderr) > 0:
                with open(miniprot_stderr, "r") as f:
                    logger.error(f"miniprot error: {f.read().strip()}")
            raise RuntimeError(f"miniprot failed with exit code {result}")

        # Parse miniprot output into gene mappings
        gene_mappings = _parse_paf_gene_mappings(
            miniprot_output, target_coverage_threshold, identity_threshold
        )

        gene_families = {
            gene_family for genes in gene_mappings.values() for gene_family in genes
        }
        logger.info(
            f"Mapped {len(gene_families)} core gene families across "
            f"{len(gene_mappings)} contigs"
        )

        # Also replace an old cache when re-annotation finds no accepted matches.
        if gene_mappings or os.path.exists(cache_path):
            cache_path = get_gene_mappings_cache_path(args)
            try:
                with open(cache_path, "w") as f:
                    json.dump(gene_mappings, f, indent=2)
                logger.info(
                    f"Saved gene mappings cache for {len(gene_mappings)} contigs to {cache_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to save gene mappings cache: {e}")

        return gene_mappings

    except Exception as e:
        logger.error(f"Error generating gene mappings: {e}")
        raise

    finally:
        # Clean up temp files unless keeping intermediate
        if not getattr(args, "keep_intermediate", False):
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary gene mapping files: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary files: {e}")
        else:
            logger.info(f"Gene mapping files preserved at: {temp_dir}")


def get_core_gene_duplication_results_path(args):
    """Get the path for the core gene duplication results file."""
    return os.path.join(args.output, "core_gene_duplication_results.json")


def get_gene_mappings_cache_path(args):
    """Get the path for the gene-to-contig mappings cache file."""
    return os.path.join(args.output, "gene_contig_mappings.json")


def parse_and_cache_paf_files(
    temp_dir,
    filtered_clusters,
    args,
    target_coverage_threshold=0.60,
    identity_threshold=0.40,
):
    """
    Parse PAF files from miniprot output and cache gene-to-contig mappings.

    This function extracts all gene-to-contig mappings from PAF files and stores
    them in a format that can be reused during rescue without re-running miniprot.

    Args:
        temp_dir: Directory containing PAF files
        filtered_clusters: Dictionary of cluster_id -> contig_headers
        args: Arguments object
        target_coverage_threshold: Minimum target coverage for alignments
        identity_threshold: Minimum identity for alignments

    Returns:
        dict: {contig_name: {gene_family: {score, coverage, identity, loci}}}
    """
    logger.info("Parsing and caching gene-to-contig mappings from PAF files...")

    # Global mapping: contig -> gene_family -> alignment_info
    global_gene_mappings = {}

    for cluster_id in filtered_clusters:
        paf_file = os.path.join(temp_dir, f"{cluster_id}.paf")
        _parse_paf_gene_mappings(
            paf_file,
            target_coverage_threshold,
            identity_threshold,
            global_gene_mappings,
        )

    # Save cache if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        cache_path = get_gene_mappings_cache_path(args)
        with open(cache_path, "w") as f:
            json.dump(global_gene_mappings, f, indent=2)
        logger.info(f"Gene-to-contig mappings cached to {cache_path}")

    logger.info(f"Cached gene mappings for {len(global_gene_mappings)} contigs")
    return global_gene_mappings


def check_core_gene_duplications_from_cache(clusters_df, gene_mappings_cache, args):
    """
    Check for duplicated core genes using cached gene-to-contig mappings.

    This function reuses existing gene mappings instead of re-running miniprot.

    Args:
        clusters_df: DataFrame with cluster assignments
        gene_mappings_cache: Dict from parse_and_cache_paf_files()
        args: Arguments object

    Returns:
        DataFrame: Updated clusters_df with duplication information
    """
    logger.debug("Checking core gene duplications using cached mappings...")

    # Group contigs by cluster efficiently
    cluster_contig_dict = clusters_df.groupby("cluster")["contig"].apply(set).to_dict()

    duplication_results = {}

    for cluster_id, contig_names in cluster_contig_dict.items():
        if cluster_id == "noise":
            continue

        duplication_results[cluster_id] = _get_gene_duplication_stats(
            contig_names, gene_mappings_cache
        )

    # Add duplication information to clusters_df
    clusters_df = initialize_duplication_columns(clusters_df)

    for cluster_id, result in duplication_results.items():
        mask = clusters_df["cluster"] == cluster_id
        clusters_df.loc[mask, "has_duplicated_core_genes"] = result["has_duplications"]
        clusters_df.loc[mask, "duplicated_core_genes_count"] = len(
            result["duplicated_genes"]
        )
        clusters_df.loc[mask, "total_core_genes_found"] = result["total_genes_found"]
        clusters_df.loc[mask, "single_copy_genes_count"] = result[
            "single_copy_genes_count"
        ]

    # Log summary
    bins_with_duplications = sum(
        1 for r in duplication_results.values() if r["has_duplications"]
    )
    total_bins_checked = len(duplication_results)
    logger.info(
        f"Checked {total_bins_checked} bins using cache: {bins_with_duplications} have duplicated core genes"
    )

    # Save duplication results as a pipeline output
    results_path = get_core_gene_duplication_results_path(args)
    try:
        with open(results_path, "w") as f:
            json.dump(duplication_results, f, indent=2)
        logger.debug(f"Saved duplication results to {results_path}")
    except Exception as e:
        logger.warning(f"Failed to save duplication results: {e}")

    return clusters_df


def check_core_gene_duplications(
    clusters_df,
    fragments_dict,
    args,
    target_coverage_threshold=0.60,
    identity_threshold=0.40,
):
    """
    Check for duplicated core genes using miniprot.

    Args:
        clusters_df: DataFrame with cluster assignments
        fragments_dict: Dictionary mapping headers to sequences
        args: Arguments object containing output directory, cores, etc.
        target_coverage_threshold: Minimum target coverage (default: 0.60)
        identity_threshold: Minimum identity (default: 0.40)

    Returns:
        DataFrame: Updated clusters_df with duplication information
    """
    # Check if miniprot is available
    if not check_miniprot_available():
        logger.error("miniprot not found in PATH")
        logger.error("Install miniprot with: conda install -c bioconda miniprot")
        raise RuntimeError("miniprot not found in PATH")

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "db", "refseq_db.faa.gz"
    )
    if not os.path.exists(db_path):
        raise RuntimeError(f"Eukaryotic database not found: {db_path}")

    logger.info("Checking for duplicated core genes using miniprot...")

    # Create temporary directory
    temp_dir = os.path.join(args.output, "temp_miniprot")
    os.makedirs(temp_dir, exist_ok=True)

    # Group contigs by cluster (clusters_df is now contig-level)
    # Use ContigHeaderMapper for efficient lookups
    mapper = ContigHeaderMapper(fragments_dict)

    cluster_contig_dict = (
        clusters_df.groupby("cluster")["contig"]
        .apply(
            lambda contigs: {
                mapper.get_header(c) for c in contigs if mapper.get_header(c)
            }
        )
        .to_dict()
    )

    # Filter clusters by size and exclude noise
    filtered_clusters = {}
    for cluster_id, contig_headers in cluster_contig_dict.items():
        if cluster_id == "noise":
            continue
        total_size = sum(len(fragments_dict[h]["sequence"]) for h in contig_headers)
        if total_size >= args.min_bin_size:
            filtered_clusters[cluster_id] = contig_headers

    duplication_results = {}

    try:
        for cluster_id, contig_headers in tqdm(
            filtered_clusters.items(), desc="Checking core gene duplications"
        ):
            # Create temporary FASTA file
            bin_fasta = os.path.join(temp_dir, f"{cluster_id}.fa")
            with open(bin_fasta, "w") as f:
                for header in contig_headers:
                    _write_fasta_record(f, header, fragments_dict[header]["sequence"])

            # Run miniprot
            miniprot_output = os.path.join(temp_dir, f"{cluster_id}.paf")
            miniprot_stderr = os.path.join(temp_dir, f"{cluster_id}.stderr")
            db_to_use = db_path  # Use the compressed file directly

            # Build secure command list (no shell injection possible)
            cmd_list = _build_miniprot_cmd(bin_fasta, db_to_use, args.cores)

            if args.verbose:
                logger.debug(f"Running miniprot command: {' '.join(cmd_list)}")

            try:
                # Use secure subprocess with proper I/O redirection
                with (
                    open(miniprot_output, "w") as stdout_file,
                    open(miniprot_stderr, "w") as stderr_file,
                ):
                    process = subprocess.run(
                        cmd_list,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        timeout=14400,  # 4 hour timeout for large datasets
                        check=False,  # Don't raise exception on non-zero exit
                    )
                    result = process.returncode
                if result == 0:
                    # Parse miniprot output into per-contig gene families
                    gene_mappings = _parse_paf_gene_mappings(
                        miniprot_output,
                        target_coverage_threshold,
                        identity_threshold,
                    )
                    duplication_results[cluster_id] = _get_gene_duplication_stats(
                        gene_mappings, gene_mappings
                    )

                else:
                    # Log miniprot error if available
                    error_msg = (
                        f"miniprot failed for {cluster_id} (exit code: {result})"
                    )
                    if (
                        os.path.exists(miniprot_stderr)
                        and os.path.getsize(miniprot_stderr) > 0
                    ):
                        with open(miniprot_stderr, "r") as stderr_file:
                            stderr_content = stderr_file.read().strip()
                            if stderr_content:
                                error_msg += f" - Error: {stderr_content}"
                    raise RuntimeError(error_msg)

            except Exception as e:
                logger.error(f"Error running miniprot for {cluster_id}: {e}")
                raise

        # Parse and cache gene mappings for rescue
        gene_mappings_cache = parse_and_cache_paf_files(
            temp_dir,
            filtered_clusters,
            args,
            target_coverage_threshold,
            identity_threshold,
        )

        # Store mappings for immediate use during rescue
        args._gene_mappings_cache = gene_mappings_cache

    finally:
        # Clean up temp_miniprot folder unless keeping intermediate files
        if not getattr(args, "keep_intermediate", False):
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary miniprot files at: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary miniprot files: {e}")
        else:
            logger.info(f"Miniprot files preserved at: {temp_dir}")

    # Add duplication information to clusters_df
    clusters_df = initialize_duplication_columns(clusters_df)

    for cluster_id, result in duplication_results.items():
        mask = clusters_df["cluster"] == cluster_id
        clusters_df.loc[mask, "has_duplicated_core_genes"] = result["has_duplications"]
        clusters_df.loc[mask, "duplicated_core_genes_count"] = len(
            result["duplicated_genes"]
        )
        clusters_df.loc[mask, "total_core_genes_found"] = result["total_genes_found"]
        clusters_df.loc[mask, "single_copy_genes_count"] = result[
            "single_copy_genes_count"
        ]

    # Log summary
    bins_with_duplications = sum(
        1 for r in duplication_results.values() if r["has_duplications"]
    )
    total_bins_checked = len(duplication_results)
    logger.info(
        f"Checked {total_bins_checked} bins: {bins_with_duplications} have duplicated core genes"
    )

    # Save duplication results as a pipeline output
    results_path = get_core_gene_duplication_results_path(args)
    try:
        with open(results_path, "w") as f:
            json.dump(duplication_results, f, indent=2)
        logger.debug(f"Saved duplication results to {results_path}")
    except Exception as e:
        logger.warning(f"Failed to save duplication results: {e}")

    return clusters_df
