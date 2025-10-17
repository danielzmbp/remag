#!/usr/bin/env python3
"""
Calculate raw coverage from BAM files for contigs.

This script calculates the raw coverage (mean read depth) for each contig
from one or more BAM files. It supports parallel processing, gzipped FASTA files,
and filtering from bins.csv or contig lists.
"""

import argparse
import gzip
import logging
import os
import sys
from multiprocessing import Pool
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_contig_filter(contigs_file: str) -> Set[str]:
    """
    Load contig names from a file (bins.csv or text file).
    
    Args:
        contigs_file: Path to file containing contig names
        
    Returns:
        Set of contig names to include
    """
    contigs = set()
    
    try:
        # Try to read as CSV first (e.g., bins.csv)
        if contigs_file.endswith('.csv'):
            df = pd.read_csv(contigs_file)
            if 'contig' in df.columns:
                contigs = set(df['contig'].astype(str))
            else:
                # If no 'contig' column, use first column
                contigs = set(df.iloc[:, 0].astype(str))
        else:
            # Read as plain text file (one contig per line)
            with open(contigs_file, 'r') as f:
                contigs = set(line.strip() for line in f if line.strip())
        
        logger.info(f"Loaded {len(contigs)} contig names from {contigs_file}")
        
    except Exception as e:
        logger.error(f"Error loading contig filter from {contigs_file}: {e}")
        sys.exit(1)
    
    return contigs


def parse_fasta(fasta_file: str, contig_filter: Optional[Set[str]] = None) -> Dict[str, int]:
    """
    Parse FASTA file to get contig names and lengths.
    
    Args:
        fasta_file: Path to FASTA file (can be gzipped)
        contig_filter: Optional set of contig names to include
        
    Returns:
        Dictionary mapping contig names to their lengths
    """
    contigs = {}
    current_header = None
    current_seq_len = 0
    
    # Determine if file is gzipped
    open_func = gzip.open if fasta_file.endswith('.gz') else open
    mode = 'rt' if fasta_file.endswith('.gz') else 'r'
    
    logger.info(f"Parsing FASTA file: {fasta_file}")
    
    try:
        with open_func(fasta_file, mode) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous contig if exists and passes filter
                    if current_header is not None and (contig_filter is None or current_header in contig_filter):
                        contigs[current_header] = current_seq_len
                    
                    # Start new contig
                    current_header = line[1:].split()[0]  # Remove '>' and take first part
                    current_seq_len = 0
                else:
                    # Add to current sequence length
                    current_seq_len += len(line)
            
            # Save last contig if it passes filter
            if current_header is not None and (contig_filter is None or current_header in contig_filter):
                contigs[current_header] = current_seq_len
                
    except Exception as e:
        logger.error(f"Error parsing FASTA file {fasta_file}: {e}")
        sys.exit(1)
    
    logger.info(f"Found {len(contigs)} contigs")
    return contigs


def validate_bam_file(bam_file: str) -> bool:
    """
    Validate BAM file and check for index.
    
    Args:
        bam_file: Path to BAM file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(bam_file):
        logger.error(f"BAM file not found: {bam_file}")
        return False
    
    # Check for BAM index
    bai_filepath = bam_file + ".bai"
    alt_bai_filepath = os.path.splitext(bam_file)[0] + ".bai"
    
    if not os.path.exists(bai_filepath) and not os.path.exists(alt_bai_filepath):
        logger.info(f"Creating BAM index for {bam_file}")
        try:
            pysam.index(bam_file)
        except Exception as e:
            logger.error(f"Failed to create BAM index for {bam_file}: {e}")
            return False
    
    return True


def calculate_coverage_batch(args: Tuple[List[Tuple[str, int]], str]) -> Dict[str, float]:
    """
    Calculate coverage for a batch of contigs from a BAM file.
    Much faster than processing one contig at a time.
    
    Args:
        args: Tuple of (list of (contig_name, contig_length), bam_file)
        
    Returns:
        Dictionary mapping contig names to coverage values
    """
    contig_batch, bam_file = args
    results = {}
    
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bamfile:
            bam_references = set(bamfile.references)
            
            for contig_name, contig_length in contig_batch:
                # Check if contig exists in BAM
                if contig_name not in bam_references:
                    results[contig_name] = 0.0
                    continue
                
                try:
                    # Use count_coverage - MUCH faster than pileup
                    coverage_arrays = bamfile.count_coverage(
                        contig=contig_name,
                        start=0,
                        stop=contig_length,
                        quality_threshold=0,
                    )
                    # Sum across all nucleotides (A, C, G, T) and calculate mean
                    total_coverage_per_base = np.sum(coverage_arrays, axis=0)
                    mean_coverage = np.mean(total_coverage_per_base) if len(total_coverage_per_base) > 0 else 0.0
                    results[contig_name] = float(mean_coverage)
                    
                except Exception as e:
                    logger.warning(f"Error calculating coverage for {contig_name}: {e}")
                    results[contig_name] = 0.0
                    
    except Exception as e:
        logger.error(f"Error opening BAM file {bam_file}: {e}")
        # Return zeros for all contigs in batch
        for contig_name, _ in contig_batch:
            results[contig_name] = 0.0
    
    return results


def calculate_coverage_fast_approximate(bam_file: str, contigs: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate approximate coverage using samtools idxstats (very fast).
    
    Args:
        bam_file: Path to BAM file
        contigs: Dictionary of contig names to lengths
        
    Returns:
        Dictionary mapping contig names to approximate coverage values
    """
    results = {}
    
    try:
        # Get index stats
        stats = pysam.idxstats(bam_file)
        
        # Parse stats
        for line in stats.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                ref_name = parts[0]
                ref_length = int(parts[1])
                mapped_reads = int(parts[2])
                
                if ref_name in contigs and ref_length > 0:
                    # Approximate coverage = mapped_reads * read_length / contig_length
                    # Assuming average read length of 150bp (adjust if needed)
                    approx_coverage = (mapped_reads * 150) / ref_length
                    results[ref_name] = float(approx_coverage)
        
        # Fill missing contigs with 0
        for contig_name in contigs:
            if contig_name not in results:
                results[contig_name] = 0.0
                
    except Exception as e:
        logger.error(f"Error getting index stats for {bam_file}: {e}")
        results = {name: 0.0 for name in contigs}
    
    return results


def calculate_coverage_for_bam(bam_file: str, contigs: Dict[str, int], threads: int, batch_size: int = 50, fast_mode: bool = False) -> Dict[str, float]:
    """
    Calculate coverage for all contigs from a single BAM file.
    
    Args:
        bam_file: Path to BAM file
        contigs: Dictionary of contig names to lengths
        threads: Number of threads to use
        batch_size: Number of contigs to process per batch
        fast_mode: Use approximate coverage calculation (much faster)
        
    Returns:
        Dictionary mapping contig names to coverage values
    """
    logger.info(f"Calculating coverage from {os.path.basename(bam_file)}")
    
    if fast_mode:
        logger.info("Using fast approximate coverage calculation")
        return calculate_coverage_fast_approximate(bam_file, contigs)
    
    # Create batches of contigs
    contig_items = list(contigs.items())
    batches = []
    
    for i in range(0, len(contig_items), batch_size):
        batch = contig_items[i:i + batch_size]
        batches.append((batch, bam_file))
    
    logger.info(f"Processing {len(contig_items)} contigs in {len(batches)} batches")
    
    # Calculate coverage using multiprocessing with batches
    coverage_dict = {}
    
    if threads > 1 and len(batches) > 1:
        with Pool(processes=min(threads, len(batches))) as pool:
            batch_results = list(tqdm(
                pool.imap(calculate_coverage_batch, batches),
                total=len(batches),
                desc=f"Processing {os.path.basename(bam_file)}"
            ))
    else:
        batch_results = [
            calculate_coverage_batch(batch_args) 
            for batch_args in tqdm(batches, desc=f"Processing {os.path.basename(bam_file)}")
        ]
    
    # Merge all batch results
    for batch_result in batch_results:
        coverage_dict.update(batch_result)
    
    return coverage_dict


def main():
    parser = argparse.ArgumentParser(
        description="Calculate raw coverage from BAM files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-f", "--fasta",
        required=True,
        help="Input FASTA file (can be gzipped)"
    )
    
    parser.add_argument(
        "-b", "--bam",
        nargs='+',
        required=True,
        help="Input BAM file(s)"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TSV file"
    )
    
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        help="Number of threads for parallel processing"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum contig length to include"
    )
    
    parser.add_argument(
        "--contigs-file",
        help="File containing contig names to include (bins.csv or text file)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of contigs to process per batch (affects memory usage)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast approximate coverage calculation (much faster but less accurate)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.fasta):
        logger.error(f"FASTA file not found: {args.fasta}")
        sys.exit(1)
    
    for bam_file in args.bam:
        if not validate_bam_file(bam_file):
            sys.exit(1)
    
    # Load contig filter if provided
    contig_filter = None
    if args.contigs_file:
        if not os.path.exists(args.contigs_file):
            logger.error(f"Contigs file not found: {args.contigs_file}")
            sys.exit(1)
        contig_filter = load_contig_filter(args.contigs_file)
    
    # Parse FASTA file
    contigs = parse_fasta(args.fasta, contig_filter)
    
    # Filter contigs by minimum length
    if args.min_length > 0:
        filtered_contigs = {
            name: length for name, length in contigs.items() 
            if length >= args.min_length
        }
        logger.info(f"Filtered to {len(filtered_contigs)} contigs >= {args.min_length} bp")
        contigs = filtered_contigs
    
    # Calculate coverage for each BAM file
    coverage_data = []
    
    for bam_file in args.bam:
        coverage_dict = calculate_coverage_for_bam(
            bam_file, contigs, args.threads, args.batch_size, args.fast
        )
        
        # Get sample name from BAM filename
        sample_name = os.path.splitext(os.path.basename(bam_file))[0]
        
        # Add coverage data
        for contig_name, length in contigs.items():
            coverage_data.append({
                'contig': contig_name,
                'length': length,
                'sample': sample_name,
                'coverage': coverage_dict.get(contig_name, 0.0)
            })
    
    # Create DataFrame
    df = pd.DataFrame(coverage_data)
    
    # Pivot to get samples as columns
    result_df = df.pivot(index=['contig', 'length'], columns='sample', values='coverage')
    result_df = result_df.reset_index()
    
    # Fill NaN values with 0
    result_df = result_df.fillna(0.0)
    
    # Save to TSV
    result_df.to_csv(args.output, sep='\t', index=False)
    
    logger.info(f"Coverage results saved to {args.output}")
    logger.info(f"Processed {len(contigs)} contigs from {len(args.bam)} BAM files")


if __name__ == "__main__":
    main()