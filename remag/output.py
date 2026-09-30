"""
Output module for REMAG
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from .utils import ContigHeaderMapper, _write_fasta_record

REMAG_OUTPUT_PATTERNS = (
    "bins.csv",
    "pre_rescue.csv",
    "rescue_assignments.csv",
    "clustering_provenance.json",
    "knn_graph_contigs.csv",
    "embeddings.csv",
    "siamese_model.pt",
    "kmer_embeddings.csv",
    "coverage_embeddings.csv",
    "params.json",
    "features.csv",
    "fragments.pkl",
    "knn_graph_edges.csv",
    "knn_graph_stats.json",
    "gene_contig_mappings.json",
    "core_gene_duplication_results.json",
    "*_hyenadna_classification.tsv",
    "*_eukaryotic_filtered.fasta",
    "*_eukaryotic_filtered.fasta.tmp",
    "*_non_eukaryotic.fasta",
    "*_non_eukaryotic.fasta.tmp",
    "bins/bin_*.fa",
    "umap_coordinates.csv",
    "umap_plot.pdf",
    "remag.log",
    "remag.*.log",
)


def prepare_output_directory(args):
    """Find existing results and remove recognized outputs only when forced.

    Return whether any results were found. Input protection is checked for the
    complete deletion list before removing anything; no cache validation is done.
    """
    output = Path(args.output)
    if getattr(args, "force", False) and (output / "bins").is_symlink():
        raise ValueError(
            "--force cannot clean a symlinked bins directory. "
            "Use a different output directory."
        )
    paths = {
        path
        for pattern in REMAG_OUTPUT_PATTERNS
        for path in output.glob(pattern)
        if path.is_file() or path.is_symlink()
    }
    for name in ("temp_gene_mapping", "temp_miniprot"):
        path = output / name
        if path.is_dir() or path.is_symlink():
            paths.add(path)

    if not getattr(args, "force", False):
        return bool(paths)

    inputs = [args.fasta]
    inputs.extend(getattr(args, "bam", None) or [])
    inputs.extend(getattr(args, "tsv", None) or [])
    input_paths = [Path(path).resolve() for path in inputs]
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            if any(child.is_symlink() for child in path.rglob("*")):
                raise ValueError(
                    "--force cannot clear a temporary directory containing symbolic links. "
                    "Use a different output directory."
                )
        resolved = path.resolve()
        if any(
            resolved == source or resolved in source.parents for source in input_paths
        ):
            raise ValueError(
                f"--force would remove an input file at {path}. "
                "Use a different output directory or move the input first."
            )

    for path in sorted(paths):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    # Python callers may reuse an args object from a previous run.
    vars(args).pop("_gene_mappings_cache", None)
    return bool(paths)


def validate_cached_min_contig_length(output_dir, min_contig_length):
    """Refuse to reuse results with a different or unknown length cutoff."""
    previous = None
    try:
        with open(Path(output_dir) / "params.json", encoding="utf-8") as handle:
            params = json.load(handle)
        if isinstance(params, dict):
            previous = params.get("min_contig_length")
    except (OSError, ValueError):
        pass
    if type(previous) is not int or previous <= 0:
        previous = "unknown"
    if previous != min_contig_length:
        raise ValueError(
            f"Existing outputs have a minimum contig length of {previous}; "
            f"this run requests {min_contig_length} bp. "
            "Use --force to recompute or choose a new output directory."
        )


def binning_settings(args):
    """Effective settings governing core clustering, unified rescue and export."""
    from .rescue import RESCUE_ALGORITHM

    minimum = getattr(args, "min_contig_length", 1)
    return {
        "admission_min_contig_length": minimum,
        "graph_min_contig_length": getattr(args, "graph_min_contig_length", None)
        or minimum,
        "rescue_algorithm": RESCUE_ALGORITHM,
        "skip_rescue": getattr(args, "skip_rescue", False),
        "similarity_threshold": 0.70,
        "max_duplication_increase": getattr(
            args, "rescue_max_duplication_increase", 5.0
        ),
        "max_total_duplication": getattr(args, "rescue_max_total_duplication", 5.0),
        "merge_duplication_ceiling": 10.0,
        "nonworsening_merge_similarity": 0.95,
        "candidate_order": "shared embedding row order",
        "centroid_policy": "fixed during merging; recomputed once before recruitment",
        "min_bin_size": getattr(args, "min_bin_size", 500000),
        "greedy_resolutions": getattr(args, "greedy_resolutions", [0.5, 1.0, 2.0, 5.0]),
        "greedy_max_contamination": getattr(args, "greedy_max_contamination", 0.10),
        "leiden_k_neighbors": getattr(args, "leiden_k_neighbors", 15),
        "leiden_similarity_threshold": getattr(
            args, "leiden_similarity_threshold", 0.1
        ),
        "leiden_seed": 42,
        "leiden_weights": "weight",
        "skip_bacterial_filter": getattr(args, "skip_bacterial_filter", False),
    }


def incompatible_cache(detail):
    return ValueError(
        f"Existing clustering outputs are incompatible: {detail}. "
        "Choose a new output directory to preserve them, or use --force to recompute."
    )


def validate_cached_binning_settings(args):
    """Reject legacy or incompatible binning before opening logs or other outputs."""
    root = Path(args.output)
    binning_files = (
        "bins.csv",
        "pre_rescue.csv",
        "clustering_provenance.json",
        "rescue_assignments.csv",
        "knn_graph_edges.csv",
        "knn_graph_stats.json",
    )
    if not any((root / p).exists() for p in binning_files) and not list(
        root.glob("bins/bin_*.fa")
    ):
        return
    try:
        previous = json.loads((root / "clustering_provenance.json").read_text())
    except (OSError, ValueError) as error:
        raise incompatible_cache("missing unified-rescue provenance") from error
    if previous.get("settings") != binning_settings(args):
        raise incompatible_cache("graph cutoff or clustering/rescue settings changed")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_identity(values, contig_names):
    """Fingerprint ordered IDs and exact values, independent of float32/64 storage."""
    digest = hashlib.sha256(json.dumps(list(contig_names)).encode())
    digest.update(str(values.shape).encode())
    # Avoid copying the entire embedding matrix just to check cache compatibility.
    for start in range(0, len(values), 4096):
        digest.update(np.ascontiguousarray(values[start : start + 4096], dtype="<f8"))
    return digest.hexdigest()


def clustering_identity(embeddings, fragments, genes, args):
    sequences = hashlib.sha256()
    for contig in embeddings.index:
        sequences.update(
            contig.encode() + b"\0" + fragments[contig]["sequence"].encode() + b"\0"
        )
    return {
        "settings": binning_settings(args),
        "ordered_embeddings_sha256": embedding_identity(
            embeddings.values, embeddings.index
        ),
        "sequences_sha256": sequences.hexdigest(),
        "gene_mappings_sha256": hashlib.sha256(
            json.dumps(genes, sort_keys=True).encode()
        ).hexdigest(),
    }


def save_clusters_as_fasta(clusters_df, fragments_dict, args):
    """
    Write cluster bins as FASTA files under the `<output>/bins` directory.

    Parameters:
        clusters_df (pd.DataFrame): DataFrame with at least `cluster` and `contig` columns mapping contigs to cluster IDs.
        fragments_dict (dict): Mapping from FASTA header to a fragment record containing a `"sequence"` string.
        args (object): Namespace-like object with `output` (base output directory) and `min_bin_size` (minimum total bases per bin) attributes.

    Returns:
        valid_bins (set): Set of cluster IDs that were saved as FASTA files (excludes the `"noise"` cluster).
    """
    bins_dir = os.path.join(args.output, "bins")
    os.makedirs(bins_dir, exist_ok=True)

    logger.info(f"Saving clusters as FASTA files in {bins_dir}...")

    # Create mapper for efficient contig name to header lookups
    mapper = ContigHeaderMapper(fragments_dict)

    # Group contigs by cluster directly using vectorized operations
    cluster_contig_dict = (
        clusters_df.groupby("cluster")["contig"]
        .apply(
            lambda contigs: list(
                dict.fromkeys(
                    mapper.get_header(c) for c in contigs if mapper.get_header(c)
                )
            )
        )
        .to_dict()
    )

    # Filter clusters by size and exclude noise, storing size for later logging
    filtered_cluster_contigs = {}
    filtered_cluster_sizes = {}
    for cluster_id, contig_headers in cluster_contig_dict.items():
        if cluster_id == "noise":
            continue
        total_size = sum(len(fragments_dict[h]["sequence"]) for h in contig_headers)
        if total_size >= args.min_bin_size:
            filtered_cluster_contigs[cluster_id] = contig_headers
            filtered_cluster_sizes[cluster_id] = total_size

    # Write FASTA files
    logger.info("Bin composition:")
    for cluster_id, contig_headers in filtered_cluster_contigs.items():
        # Create simple filename
        bin_file = os.path.join(bins_dir, f"{cluster_id}.fa")

        logger.info(
            f"  {cluster_id}: {len(contig_headers)} contigs, {filtered_cluster_sizes[cluster_id]:,} bp"
        )

        with open(bin_file, "w") as f:
            for header in contig_headers:
                seq = fragments_dict[header]["sequence"]
                _write_fasta_record(f, header, seq)
                if not seq:
                    f.write("\n")

    total_contigs_in_bins = sum(
        len(contigs) for contigs in filtered_cluster_contigs.values()
    )
    logger.info(
        f"Saved {len(filtered_cluster_contigs)} bins with {total_contigs_in_bins} total contigs"
    )

    valid_bins = set(filtered_cluster_contigs.keys())
    return valid_bins
