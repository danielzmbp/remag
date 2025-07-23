"""
Output module for REMAG
"""

import os
from tqdm import tqdm
from loguru import logger

from .utils import extract_base_contig_name, ContigHeaderMapper


def save_clusters_as_fasta(clusters_df, fragments_dict, args):
    """Save clusters as FASTA files and return valid bin IDs."""
    bins_dir = os.path.join(args.output, "bins")
    os.makedirs(bins_dir, exist_ok=True)

    logger.info(f"Saving clusters as FASTA files in {bins_dir}...")

    # Create mapper for efficient contig name to header lookups
    mapper = ContigHeaderMapper(fragments_dict)

    # Group contigs by cluster directly
    cluster_contig_dict = {}
    for _, row in tqdm(clusters_df.iterrows(), desc="Processing contigs"):
        contig_name = row["contig"]
        cluster_id = row["cluster"]
        
        # Find the corresponding original header in fragments_dict
        original_header = mapper.get_header(contig_name)
        
        if original_header:
            if cluster_id not in cluster_contig_dict:
                cluster_contig_dict[cluster_id] = set()
            cluster_contig_dict[cluster_id].add(original_header)

    # Filter clusters by size
    filtered_cluster_contigs = {}
    for cluster_id, contig_headers in cluster_contig_dict.items():
        total_size = sum(len(fragments_dict[h]["sequence"]) for h in contig_headers)
        if total_size >= args.min_bin_size:
            filtered_cluster_contigs[cluster_id] = contig_headers

    # Write FASTA files
    logger.info("Bin composition:")
    for cluster_id, contig_headers in tqdm(
        filtered_cluster_contigs.items(), desc="Writing bin files"
    ):
        if cluster_id == "noise":
            continue

        total_length = sum(len(fragments_dict[h]["sequence"]) for h in contig_headers)

        # Check for duplicated core genes
        has_duplications = False
        if "has_duplicated_core_genes" in clusters_df.columns:
            bin_fragments = clusters_df[clusters_df["cluster"] == cluster_id]
            if not bin_fragments.empty:
                has_duplications = bin_fragments["has_duplicated_core_genes"].iloc[0]

        # Create filename with duplication tag
        if has_duplications:
            bin_file = os.path.join(bins_dir, f"{cluster_id}_duplicated_core_genes.fa")
            duplication_tag = " [DUPLICATED CORE GENES]"
        else:
            bin_file = os.path.join(bins_dir, f"{cluster_id}.fa")
            duplication_tag = ""

        logger.info(
            f"  {cluster_id}: {len(contig_headers)} contigs, {total_length:,} bp{duplication_tag}"
        )

        with open(bin_file, "w") as f:
            for header in contig_headers:
                seq = fragments_dict[header]["sequence"]
                f.write(f">{header}\n")
                for i in range(0, len(seq), 60):
                    f.write(f"{seq[i: i+60]}\n")

    total_contigs_in_bins = sum(
        len(contigs) for contigs in filtered_cluster_contigs.values()
    )
    logger.info(
        f"Saved {len(filtered_cluster_contigs)} bins with {total_contigs_in_bins} total contigs"
    )
    
    valid_bins = set(filtered_cluster_contigs.keys()) - {"noise"}
    return valid_bins
