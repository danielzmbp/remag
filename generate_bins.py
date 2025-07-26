#!/usr/bin/env python3
"""
Generate bin FASTA files from contigs and cluster assignments.

This script reads a FASTA file containing contigs and a CSV file with 
contig-to-bin assignments, then creates separate FASTA files for each bin.
"""

import argparse
import os
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_bins_csv(bins_file):
    """Read the bins CSV file and return a dictionary mapping contigs to bins."""
    logger.info(f"Reading bins assignments from {bins_file}")
    
    bins_df = pd.read_csv(bins_file)
    
    # Check if required columns exist
    if 'contig' not in bins_df.columns or 'cluster' not in bins_df.columns:
        raise ValueError("bins.csv must contain 'contig' and 'cluster' columns")
    
    # Create contig to bin mapping
    contig_to_bin = dict(zip(bins_df['contig'], bins_df['cluster']))
    
    # Count contigs per bin
    bin_counts = bins_df['cluster'].value_counts()
    
    logger.info(f"Found {len(bins_df)} contig assignments across {len(bin_counts)} bins")
    for bin_name, count in bin_counts.items():
        logger.info(f"  Bin {bin_name}: {count} contigs")
    
    return contig_to_bin


def generate_bins(fasta_file, contig_to_bin, output_dir, min_size=0):
    """Generate bin FASTA files from contigs.
    
    Args:
        fasta_file: Path to input FASTA file
        contig_to_bin: Dictionary mapping contig IDs to bin names
        output_dir: Output directory
        min_size: Minimum bin size in bp (default: 0, no filtering)
    """
    # Create bins subdirectory to match REMAG output structure
    bins_dir = os.path.join(output_dir, "bins")
    os.makedirs(bins_dir, exist_ok=True)
    
    # Dictionary to store sequences for each bin
    bin_sequences = defaultdict(list)
    
    # Statistics
    total_contigs = 0
    assigned_contigs = 0
    unassigned_contigs = []
    
    logger.info(f"Reading contigs from {fasta_file}")
    
    # Read FASTA file and distribute sequences to bins
    for record in SeqIO.parse(fasta_file, "fasta"):
        total_contigs += 1
        contig_id = record.id
        
        if contig_id in contig_to_bin:
            bin_name = contig_to_bin[contig_id]
            bin_sequences[bin_name].append(record)
            assigned_contigs += 1
        else:
            unassigned_contigs.append(contig_id)
    
    logger.info(f"Total contigs in FASTA: {total_contigs}")
    logger.info(f"Assigned contigs: {assigned_contigs}")
    logger.info(f"Unassigned contigs: {len(unassigned_contigs)}")
    
    if unassigned_contigs:
        logger.warning(f"The following contigs were not found in bins.csv: {', '.join(unassigned_contigs[:10])}")
        if len(unassigned_contigs) > 10:
            logger.warning(f"... and {len(unassigned_contigs) - 10} more")
    
    # Write bin FASTA files
    logger.info(f"Writing bin FASTA files to {bins_dir}")
    
    bins_written = 0
    bins_filtered = 0
    
    for bin_name, sequences in bin_sequences.items():
        # Calculate total bin size
        bin_size = sum(len(seq.seq) for seq in sequences)
        
        # Check if bin meets minimum size requirement
        if bin_size < min_size:
            logger.info(f"  Skipping {bin_name}: {bin_size:,} bp < {min_size:,} bp minimum")
            bins_filtered += 1
            continue
        
        output_file = os.path.join(bins_dir, f"{bin_name}.fa")
        
        with open(output_file, "w") as handle:
            SeqIO.write(sequences, handle, "fasta")
        
        logger.info(f"  Wrote {len(sequences)} sequences to {output_file} (total size: {bin_size:,} bp)")
        bins_written += 1
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total bins processed: {len(bin_sequences)}")
    logger.info(f"  Bins written: {bins_written}")
    if min_size > 0:
        logger.info(f"  Bins filtered out (< {min_size:,} bp): {bins_filtered}")
    logger.info(f"  Total sequences written: {sum(len(seqs) for bin_name, seqs in bin_sequences.items() if sum(len(seq.seq) for seq in seqs) >= min_size)}")
    
    return bin_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate bin FASTA files from contigs and cluster assignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s -f contigs.fasta -b bins.csv -o output_bins/
  
Input format:
  - FASTA file: Standard FASTA format with contig sequences
  - bins.csv: CSV file with columns 'contig' and 'cluster'
    Example:
      contig,cluster
      contig_1,bin_001
      contig_2,bin_001
      contig_3,bin_002
        """
    )
    
    parser.add_argument('-f', '--fasta', required=True,
                        help='Input FASTA file containing contigs')
    parser.add_argument('-b', '--bins', required=True,
                        help='CSV file with contig-to-bin assignments')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for bin FASTA files')
    parser.add_argument('--prefix', default='',
                        help='Prefix for output bin files (default: none)')
    parser.add_argument('--suffix', default='',
                        help='Suffix for output bin files (default: none)')
    parser.add_argument('--min-size', type=int, default=0,
                        help='Minimum bin size in bp (default: 0, no filtering)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.fasta):
        parser.error(f"FASTA file not found: {args.fasta}")
    if not os.path.exists(args.bins):
        parser.error(f"Bins CSV file not found: {args.bins}")
    
    # Read bins assignments
    contig_to_bin = read_bins_csv(args.bins)
    
    # Apply prefix/suffix to bin names if specified
    if args.prefix or args.suffix:
        contig_to_bin = {
            contig: f"{args.prefix}{bin_name}{args.suffix}" 
            for contig, bin_name in contig_to_bin.items()
        }
    
    # Generate bins
    generate_bins(args.fasta, contig_to_bin, args.output, args.min_size)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()