"""
Feature extraction module for REMAG
"""

import itertools
import numpy as np
import pandas as pd
import pysam
import os
import random
import re
from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import open_file, fasta_iter, FragmentDict, CoverageDict


def generate_feature_mapping(kmer_len):
    """Generate mapping of k-mers to feature indices."""
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0
    
    kmers = [''.join(kmer) for kmer in itertools.product("ATGC", repeat=kmer_len)]
    
    for kmer in kmers:
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = kmer.translate(str.maketrans(BASE_COMPLEMENT))[::-1]
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash, counter


def _calculate_kmer_composition(
    sequences_to_process: List[Tuple[str, str]],
    kmer_len: int = 4,
    pseudocount: float = 1e-5,
) -> pd.DataFrame:
    """
    Calculates normalized k-mer composition for a list of sequences.

    Args:
        sequences_to_process: A list of (header, sequence) tuples.
        kmer_len: The length of k-mers to calculate.
        pseudocount: A small value added to avoid division by zero.

    Returns:
        A pandas DataFrame with headers as index and normalized k-mer frequencies.
    """
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    composition = OrderedDict()

    for header, seq in sequences_to_process:
        norm_seq = seq.upper()
        kmers = []
        for j in range(len(norm_seq) - kmer_len + 1):
            kmer = norm_seq[j : j + kmer_len]
            if kmer in kmer_dict:
                kmers.append(kmer_dict[kmer])

        if kmers:
            composition[header] = np.bincount(np.array(kmers, dtype=np.int64), minlength=nr_features)
        else:
            composition[header] = np.zeros(nr_features, dtype=np.int64)

    if not composition:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(composition, orient="index", dtype=float)
    df.columns = [str(col) for col in df.columns]
    df += pseudocount
    row_sums = df.sum(axis=1)

    non_zero_mask = row_sums > 1e-9
    df[non_zero_mask] = df[non_zero_mask].div(row_sums[non_zero_mask], axis=0)
    df[~non_zero_mask] = 0.0

    return df


def get_classification_results_path(fasta_file, output_dir):
    """Get the path for the 4CAC classification results file."""
    base_name = os.path.basename(fasta_file)
    name_without_ext = os.path.splitext(base_name)[0]
    if name_without_ext.endswith(".gz"):
        name_without_ext = os.path.splitext(name_without_ext)[0]
    return os.path.join(output_dir, f"{name_without_ext}_4cac_classification.tsv")


def get_features_csv_path(output_dir):
    """Get the path for the features CSV file."""
    return os.path.join(output_dir, "features.csv")


def filter_bacterial_contigs(fasta_file, output_dir, min_contig_length=1000, cores=8):
    """
    Filter bacterial contigs using the 4CAC XGBoost classifier.
    Keeps contigs with prokaryotic & plasmid scores ≤ 0.5.
    
    Args:
        fasta_file: Path to input FASTA file
        output_dir: Output directory for filtered results
        min_contig_length: Minimum contig length threshold
        cores: Number of CPU cores to use
        
    Returns:
        str: Path to filtered FASTA file
    """
    base_name = os.path.basename(fasta_file)
    name_without_ext = os.path.splitext(base_name)[0]
    if name_without_ext.endswith(".gz"):
        name_without_ext = os.path.splitext(name_without_ext)[0]

    filtered_fasta = os.path.join(
        output_dir, f"{name_without_ext}_non_bacterial_filtered.fasta"
    )
    classification_results = get_classification_results_path(fasta_file, output_dir)

    if os.path.exists(filtered_fasta):
        logger.info(f"Using existing filtered FASTA: {filtered_fasta}")
        return filtered_fasta

    logger.info("Filtering bacterial contigs using 4CAC classifier...")

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Found JSON model saved before XGBoost 1.6.*"
            )
            try:
                from .xgbclass import xgb_class as classifier
                from .xgbclass import xgb_utils as utils
            except ImportError:
                # Fallback to global import if relative import fails
                from xgbclass import xgb_class as classifier
                from xgbclass import xgb_utils as utils

            c = classifier.xgbClass(n_procs=cores)
    except ImportError:
        logger.error(
            "4CAC classifier not found. Using original FASTA without filtering"
        )
        return fasta_file
    except Exception as e:
        logger.error(f"Failed to initialize 4CAC classifier: {e}")
        return fasta_file

    seq_names, seqs = [], []
    non_bacterial_sequences = []
    n_total = n_non_bacterial = n_filtered = 0

    def process_batch(seq_names, seqs, results_file):
        """Process a batch of sequences through classification."""
        nonlocal n_non_bacterial, n_filtered

        if not seqs:
            return

        try:
            probs = c.classify(seqs)
            for j, p in enumerate(probs):
                viral_score, plas_score, prokar_score, eukar_score = p
                predicted_class = [
                    "viral",
                    "plasmid",
                    "prokaryotic",
                    "eukaryotic",
                ][np.argmax(p)]

                results_file.write(
                    f"{seq_names[j]}\t{viral_score}\t{plas_score}\t{prokar_score}\t{eukar_score}\t{predicted_class}\n"
                )

                if prokar_score > 0.5 or plas_score > 0.5:
                    n_filtered += 1
                else:
                    non_bacterial_sequences.append((seq_names[j], seqs[j]))
                    n_non_bacterial += 1
        except Exception as e:
            logger.error(f"Classification error: {e}")
            for j in range(len(seqs)):
                non_bacterial_sequences.append((seq_names[j], seqs[j]))
                n_non_bacterial += 1

    with open(classification_results, "w", encoding="utf-8") as results_file:
        results_file.write(
            "header\tviral_score\tplas_score\tprokar_score\teukar_score\tpredicted_class\n"
        )

        with open_file(fasta_file, "r") as fp:
            try:
                try:
                    from .xgbclass import xgb_utils as utils
                except ImportError:
                    from xgbclass import xgb_utils as utils

                for header, seq, _ in utils.readfq(fp):
                    if len(seq) < min_contig_length:
                        continue

                    seq_names.append(header)
                    seqs.append(seq.upper())
                    n_total += 1

                    if len(seqs) >= 50000:
                        process_batch(seq_names, seqs, results_file)
                        seq_names, seqs = [], []

                # Process remaining sequences
                process_batch(seq_names, seqs, results_file)
            except ImportError:
                for header, seq in fasta_iter(fasta_file):
                    if len(seq) < min_contig_length:
                        continue
                    non_bacterial_sequences.append((header, seq))
                    n_total += 1
                    n_non_bacterial += 1

    logger.info(
        f"Processed {n_total} sequences: kept {n_non_bacterial}, filtered {n_filtered}"
    )

    if n_non_bacterial == 0:
        logger.warning("No non-bacterial sequences found, keeping all sequences")
        return fasta_file

    with open(filtered_fasta, "w", encoding="utf-8") as f:
        for header, sequence in non_bacterial_sequences:
            f.write(f">{header}\n")
            for i in range(0, len(sequence), 60):
                f.write(f"{sequence[i: i+60]}\n")

    logger.info(f"Filtered FASTA saved to: {filtered_fasta}")
    return filtered_fasta


def generate_augmented_fragments(
    sequence: str,
    header: str,
    min_contig_length: int,
    num_augmentations: int = 8,
    max_overlap: float = 0.25,
) -> list[tuple[str, str, int, int]]:
    """Generate diverse fragments using random masking strategies.

    For contigs >50kb, splits the contig in half and generates augmentations
    for each half separately with identifiers to distinguish views from each half.

    Args:
        sequence: The DNA sequence to fragment
        header: The contig header/name
        min_contig_length: Minimum length for generated fragments
        num_augmentations: Number of masked fragments to generate (in addition to original)
        max_overlap: Maximum allowed overlap fraction between augmented fragments (default 0.25)

    Returns:
        List of tuples: (fragment_header, fragment_seq, start_pos, fragment_length)
        For contigs >50kb: includes original full contig plus augmentations from each half
        For smaller contigs: original behavior with full contig augmentations
    """
    fragments: list[tuple[str, str, int, int]] = []
    seq_length = len(sequence)

    # Guard clause: contig too short
    if seq_length < min_contig_length:
        return fragments

    # Always include the original full contig as the first "fragment"
    original_header = f"{header}.original"
    fragments.append((original_header, sequence, 0, seq_length))

    # If no augmentations requested, return just the original
    if num_augmentations <= 0:
        return fragments

    # Check if contig is large enough to split (>50kb)
    split_threshold = 50000
    if seq_length > split_threshold:
        # Split contig in half and generate augmentations for each half
        mid_point = seq_length // 2
        first_half = sequence[:mid_point]
        second_half = sequence[mid_point:]

        # Generate augmentations for each half
        first_half_fragments = _generate_half_augmentations(
            first_half,
            header,
            "h1",
            0,
            min_contig_length,
            num_augmentations,
            max_overlap,
        )
        second_half_fragments = _generate_half_augmentations(
            second_half,
            header,
            "h2",
            mid_point,
            min_contig_length,
            num_augmentations,
            max_overlap,
        )

        fragments.extend(first_half_fragments)
        fragments.extend(second_half_fragments)

        logger.debug(
            f"Generated {len(fragments)} fragments for large contig {header} (length {seq_length}): "
            f"{len(first_half_fragments)} from first half, {len(second_half_fragments)} from second half"
        )
    else:
        # Original behavior for smaller contigs
        half_fragments = _generate_half_augmentations(
            sequence, header, "", 0, min_contig_length, num_augmentations, max_overlap
        )
        fragments.extend(half_fragments)

        logger.debug(
            f"Generated {len(fragments)} fragments for {header} (length {seq_length})"
        )

    return fragments


def _generate_half_augmentations(
    sequence: str,
    base_header: str,
    half_id: str,
    global_offset: int,
    min_contig_length: int,
    num_augmentations: int,
    max_overlap: float,
) -> list[tuple[str, str, int, int]]:
    """Generate augmented fragments for a sequence half (or full sequence for smaller contigs).

    Args:
        sequence: The sequence to augment (half or full contig)
        base_header: Base contig header
        half_id: Identifier for the half ("h1", "h2", or "" for full contig)
        global_offset: Offset in the original full sequence
        min_contig_length: Minimum fragment length
        num_augmentations: Number of augmentations to generate
        max_overlap: Maximum overlap fraction between fragments

    Returns:
        List of fragment tuples with appropriate half identifiers
    """
    fragments: list[tuple[str, str, int, int]] = []
    seq_length = len(sequence)

    # Skip if sequence is too short
    if seq_length < min_contig_length:
        return fragments

    # Seed RNG for reproducibility - include half_id for different seeds per half
    random.seed(hash(f"{base_header}_{half_id}") % 2**32)

    # Generate edge-masked fragments
    selected_count = 0
    max_attempts = num_augmentations * 10  # Limit attempts to avoid infinite loops
    attempts = 0

    while selected_count < num_augmentations and attempts < max_attempts:
        attempts += 1

        # Choose masking strategy: favor both edges (more variable) vs other strategies
        if random.choice([True, False]):
            mask_strategy = "both"  # Both edges - more variable
        else:
            mask_strategy = random.choice(
                ["left", "right", "center"]
            )  # Other strategies

        if mask_strategy == "left":
            # Mask from left edge: sequence[mask_size:]
            max_mask = seq_length - min_contig_length
            if max_mask <= 0:
                continue
            mask_size = random.randint(1, max_mask)
            start_pos = mask_size
            end_pos = seq_length

        elif mask_strategy == "right":
            # Mask from right edge: sequence[:-mask_size]
            max_mask = seq_length - min_contig_length
            if max_mask <= 0:
                continue
            mask_size = random.randint(1, max_mask)
            start_pos = 0
            end_pos = seq_length - mask_size

        elif mask_strategy == "both":
            # Mask from both edges: sequence[left_mask:-right_mask]
            max_total_mask = seq_length - min_contig_length
            if max_total_mask <= 0:
                continue
            total_mask = random.randint(1, max_total_mask)
            left_mask = random.randint(0, total_mask)
            right_mask = total_mask - left_mask
            start_pos = left_mask
            end_pos = seq_length - right_mask

        else:  # center
            # Mask center region, creating two fragments: left and right
            # Need enough space for two fragments of min_contig_length each
            if seq_length < 2 * min_contig_length:
                continue

            # Calculate center region to mask
            max_center_mask = seq_length - 2 * min_contig_length
            if max_center_mask <= 0:
                continue

            center_mask_size = random.randint(1, max_center_mask)
            center_start = random.randint(
                min_contig_length, seq_length - min_contig_length - center_mask_size
            )
            center_end = center_start + center_mask_size

            # Randomly choose left or right fragment
            if random.choice([True, False]):
                # Left fragment: sequence[:center_start]
                start_pos = 0
                end_pos = center_start
            else:
                # Right fragment: sequence[center_end:]
                start_pos = center_end
                end_pos = seq_length

        # Extract fragment
        fragment_seq = sequence[start_pos:end_pos]
        frag_len = len(fragment_seq)

        # Validate minimum length (should always pass due to our calculations)
        if frag_len < min_contig_length:
            continue

        # Check overlap with existing fragments
        valid_fragment = True
        for existing_header, existing_seq, existing_start, existing_len in fragments:
            existing_end = existing_start + existing_len
            current_start = start_pos + global_offset
            current_end = end_pos + global_offset

            # Calculate overlap
            overlap_start = max(current_start, existing_start)
            overlap_end = min(current_end, existing_end)
            overlap_length = max(0, overlap_end - overlap_start)

            # Calculate overlap as fraction of larger fragment
            max_length = max(frag_len, existing_len)
            overlap_fraction = overlap_length / max_length if max_length > 0 else 0

            if overlap_fraction >= max_overlap:
                valid_fragment = False
                break

        # Add fragment if it passes overlap check
        if valid_fragment:
            # Create header with half identifier
            if half_id:
                fragment_header = f"{base_header}.{half_id}.{selected_count}"
            else:
                fragment_header = f"{base_header}.{selected_count}"

            # Global position in the original full sequence
            global_start_pos = start_pos + global_offset
            fragments.append(
                (fragment_header, fragment_seq, global_start_pos, frag_len)
            )
            selected_count += 1

    return fragments


def get_features(
    fasta_file: str,
    bam_files: Optional[List[str]],
    tsv_files: Optional[List[str]],
    output_dir: str,
    min_contig_length: int = 1000,
    cores: int = 16,
    num_augmentations: int = 8,
    args = None,
) -> Tuple[pd.DataFrame, FragmentDict]:
    """
    Generate k-mer and coverage features for fragments.

    Args:
        fasta_file: Path to input FASTA file
        bam_files: Optional list of alignment files (BAM/CRAM) for coverage features (each represents a sample)
        tsv_files: Optional TSV files for coverage features
        output_dir: Output directory
        min_contig_length: Minimum contig length
        cores: Number of cores for processing
        num_augmentations: Number of random fragments per contig

    Returns:
        Tuple of (features DataFrame, fragments dictionary)
    """
    features_csv_path = get_features_csv_path(output_dir)
    fragments_path = os.path.join(output_dir, "fragments.pkl")

    # Try to load existing features
    if os.path.exists(features_csv_path) and os.path.exists(fragments_path):
        logger.info(f"Loading existing features from {features_csv_path}")
        try:
            df = pd.read_csv(features_csv_path, index_col=0)
            fragments_dict = pd.read_pickle(fragments_path)

            # Verify and update coverage if needed
            if bam_files and not any("_coverage" in col for col in df.columns):
                logger.info("Recalculating BAM coverage...")
                coverage_df = calculate_coverage_from_multiple_bams(
                    bam_files, fragments_dict, cores
                )
                # Remove any existing coverage columns and add new ones
                df = df.drop(
                    columns=[c for c in df.columns if "coverage" in c.lower()],
                    errors="ignore",
                )
                df = pd.concat([df, coverage_df], axis=1)
                if getattr(args, "keep_intermediate", False):
                    df.to_csv(features_csv_path)

            elif tsv_files:
                expected_cols = [
                    os.path.splitext(os.path.basename(f))[0] for f in tsv_files
                ]
                missing_cols = [col for col in expected_cols if col not in df.columns]

                if missing_cols:
                    logger.info("Recalculating TSV coverage...")
                    coverage_df = calculate_coverage_from_tsv(tsv_files, fragments_dict)
                    df = df.drop(
                        columns=[c for c in df.columns if c in coverage_df.columns],
                        errors="ignore",
                    )
                    df = pd.concat([df, coverage_df], axis=1)
                    if getattr(args, "keep_intermediate", False):
                        df.to_csv(features_csv_path)

            return df, fragments_dict

        except Exception as e:
            logger.warning(f"Error loading existing features: {e}. Regenerating...")

    # Generate new features
    logger.info("Generating k-mer features from FASTA file...")
    kmer_len = 4
    length_threshold = min_contig_length

    # Generate k-mer mapping (for consistency, though we use _calculate_kmer_composition)
    _, _ = generate_feature_mapping(kmer_len)
    fragments_dict = OrderedDict()

    # Process sequences and generate fragments
    sequence_count = 0
    fragment_count = 0
    filtered_count = 0
    short_fragment_count = 0

    logger.info(
        f"Using original contig + {num_augmentations} random fragments per contig"
    )

    # Collect all fragment sequences for batch processing
    all_fragments = []
    fragment_to_contig = {}

    for header, seq in tqdm(fasta_iter(fasta_file), desc="Processing sequences"):
        if len(seq) < length_threshold:
            filtered_count += 1
            continue

        # Store original sequence and initialize fragments
        clean_header = str(header.split()[0])
        fragments_dict[clean_header] = {
            "sequence": seq,
            "fragments": [],
            "fragment_info": {},
        }

        # Generate random augmented fragments
        augmented_fragments = generate_augmented_fragments(
            seq,
            clean_header,
            length_threshold,
            num_augmentations,
        )

        for fragment_header, fragment_seq, start_pos, frag_len in augmented_fragments:
            fragments_dict[clean_header]["fragments"].append(fragment_header)
            # Store fragment position and length information for coverage calculation
            fragments_dict[clean_header]["fragment_info"][fragment_header] = {
                "start_pos": start_pos,
                "length": frag_len,
            }

            fragment_length = len(fragment_seq)
            if fragment_length < length_threshold:
                short_fragment_count += 1
                continue

            # Collect fragment for batch processing
            all_fragments.append((fragment_header, fragment_seq))
            fragment_to_contig[fragment_header] = clean_header
            fragment_count += 1

        sequence_count += 1

    logger.info(f"Processed {sequence_count:,} sequences, {fragment_count:,} fragments")

    if not all_fragments:
        logger.error("No valid fragments generated.")
        return pd.DataFrame(), {}

    # Calculate k-mer composition for all fragments using helper function
    logger.info("Calculating k-mer composition for all fragments...")
    df = _calculate_kmer_composition(all_fragments, kmer_len=kmer_len)

    # Calculate coverage
    if bam_files:
        logger.debug("Calculating coverage from alignment files...")
        coverage_df = calculate_coverage_from_multiple_bams(
            bam_files, fragments_dict, cores
        )
        df = pd.concat([df, coverage_df.reindex(df.index).fillna(0.0)], axis=1)
        
        # Filter out fragments with zero coverage across all samples
        coverage_columns = [col for col in coverage_df.columns if "coverage" in col.lower()]
        if coverage_columns:
            zero_coverage_mask = (df[coverage_columns] == 0).all(axis=1)
            df = df[~zero_coverage_mask]
            if zero_coverage_mask.sum() > 0:
                logger.info(f"Filtered out {zero_coverage_mask.sum()} fragments with zero coverage across all samples")
    elif tsv_files:
        logger.debug("Calculating coverage from TSV files...")
        coverage_df = calculate_coverage_from_tsv(tsv_files, fragments_dict)
        df = pd.concat([df, coverage_df.reindex(df.index).fillna(0.0)], axis=1)
        
        # Filter out fragments with zero coverage across all samples
        coverage_columns = [col for col in coverage_df.columns if "coverage" in col.lower()]
        if coverage_columns:
            zero_coverage_mask = (df[coverage_columns] == 0).all(axis=1)
            df = df[~zero_coverage_mask]
            if zero_coverage_mask.sum() > 0:
                logger.info(f"Filtered out {zero_coverage_mask.sum()} fragments with zero coverage across all samples")
    else:
        logger.info("No coverage data provided - using k-mer features only")

    coverage_columns = [
        col for col in df.columns if isinstance(col, str) and "coverage" in col.lower()
    ]
    if coverage_columns:
        # Apply log transformation to coverage features
        df[coverage_columns] = df[coverage_columns].map(lambda x: np.log1p(x))
        logger.info(
            f"Applied log transformation to {len(coverage_columns)} coverage features"
        )

        # Apply global scaling to preserve co-abundance relationships across samples
        # Scale all coverage features together (preserves relative differences between samples)
        logger.info("Applying global scaling to preserve co-abundance patterns across samples")
        
        # Use MinMaxScaler to scale features to [0, 1] range
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[coverage_columns] = scaler.fit_transform(df[coverage_columns])
        logger.info(f"Applied MinMaxScaler (0-1 range) to {len(coverage_columns)} coverage features")
        
        # Log sample information for debugging
        sample_names = set()
        for col in coverage_columns:
            if "_coverage" in col:
                sample_name = col.replace("_coverage", "").replace("_std", "")
                sample_names.add(sample_name)
        logger.info(f"Processing coverage from {len(sample_names)} samples: {sorted(sample_names)}")
    else:
        logger.info("Using k-mer features only")

    compact_dict = {
        header: {
            "fragments": data["fragments"],
            "sequence": data["sequence"],
            "fragment_info": data.get("fragment_info", {}),
        }
        for header, data in fragments_dict.items()
    }
    # Save features and fragments only if keeping intermediate files
    if getattr(args, "keep_intermediate", False):
        pd.to_pickle(compact_dict, fragments_path, protocol=4)
        df.to_csv(features_csv_path)
    else:
        # Still save fragments dict for bin generation, but not features
        pd.to_pickle(compact_dict, fragments_path, protocol=4)

    return df, fragments_dict


# Coverage calculation helper functions
def _validate_alignment_file(alignment_file: str) -> bool:
    """Validate BAM/CRAM file and create index if needed."""
    if not os.path.exists(alignment_file):
        logger.error(f"Alignment file not found at {alignment_file}")
        return False

    file_ext = alignment_file.lower().split('.')[-1]
    
    if file_ext == 'bam':
        # Check for .bai index file (both .bam.bai and .bai extensions)
        bai_filepath = alignment_file + ".bai"
        alt_bai_filepath = os.path.splitext(alignment_file)[0] + ".bai"

        if not os.path.exists(bai_filepath) and not os.path.exists(alt_bai_filepath):
            logger.info(f"BAM index (.bai) not found for {alignment_file}, creating index...")
            try:
                pysam.index(alignment_file)
                logger.debug(f"BAM index created at {bai_filepath}")
                return True
            except Exception as e:
                logger.error(f"Error creating BAM index: {e}")
                logger.error("Please ensure samtools is installed and in your PATH.")
                return False
    
    elif file_ext == 'cram':
        # Check for .crai index file (both .cram.crai and .crai extensions)
        crai_filepath = alignment_file + ".crai"
        alt_crai_filepath = os.path.splitext(alignment_file)[0] + ".crai"

        if not os.path.exists(crai_filepath) and not os.path.exists(alt_crai_filepath):
            logger.info(f"CRAM index (.crai) not found for {alignment_file}, creating index...")
            try:
                pysam.index(alignment_file)
                logger.debug(f"CRAM index created at {crai_filepath}")
                return True
            except Exception as e:
                logger.error(f"Error creating CRAM index: {e}")
                logger.error("Please ensure samtools is installed and in your PATH.")
                return False
    
    else:
        logger.error(f"Unsupported alignment file format: {alignment_file}. Supported formats: BAM, CRAM")
        return False

    return True


def _map_fasta_to_bam_refs(
    fragments_dict: FragmentDict, bam_references: Set[str]
) -> Tuple[Dict[str, str], List[str]]:
    """Map FASTA headers to BAM references."""
    bam_ref_map = {}
    unmapped_fasta_headers = []

    for original_header in tqdm(fragments_dict.keys(), desc="Mapping headers"):
        fasta_key = original_header

        # Try direct match
        if fasta_key in bam_references:
            bam_ref_map[original_header] = fasta_key
            continue

        # Try matching after splitting on space
        base_name_space = fasta_key.split(" ")[0]
        if base_name_space in bam_references:
            bam_ref_map[original_header] = base_name_space
            continue

        # Try matching after splitting on last dot
        base_name_dot = fasta_key.rsplit(".", 1)[0] if "." in fasta_key else None
        if base_name_dot and base_name_dot in bam_references:
            bam_ref_map[original_header] = base_name_dot
            continue

        bam_ref_map[original_header] = None
        unmapped_fasta_headers.append(original_header)

    return bam_ref_map, unmapped_fasta_headers


def _process_contig_coverage_worker(args):
    """
    Optimized worker function to calculate coverage for fragments within a single contig.
    Now receives pre-loaded coverage data to avoid expensive BAM file I/O per worker.

    Args:
        args: Tuple of (bam_contig_name, contig_data_list, total_coverage_per_base, bam_contig_length)

    Returns:
        Tuple of (fragment_coverage, fragment_coverage_std, [])
    """
    (bam_contig_name, contig_data_list, total_coverage_per_base, bam_contig_length) = args
    fragment_coverage = {}
    fragment_coverage_std = {}

    try:
        if total_coverage_per_base is None or bam_contig_length is None:
            # No coverage data available - using zero coverage
            for original_header, data in contig_data_list:
                for fragment_header in data["fragments"]:
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_std[fragment_header] = 0.0
            return fragment_coverage, fragment_coverage_std, []

        # Collect all fragment coordinates and headers for vectorized processing
        fragment_coords = []
        fragment_headers = []
        
        for original_header, data in contig_data_list:
            fragment_info = data.get("fragment_info", {})

            for fragment_header in data["fragments"]:
                # Use stored fragment position and length information
                if fragment_header not in fragment_info:
                    logger.warning(
                        f"Missing fragment info for {fragment_header}, setting zero coverage"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_std[fragment_header] = 0.0
                    continue

                start_pos = fragment_info[fragment_header]["start_pos"]
                frag_len = fragment_info[fragment_header]["length"]
                end_pos = start_pos + frag_len

                # Validate fragment positions
                if start_pos < 0:
                    logger.warning(
                        f"Fragment {fragment_header} has negative start position: {start_pos}"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_std[fragment_header] = 0.0
                    continue

                if start_pos >= bam_contig_length:
                    logger.warning(
                        f"Fragment {fragment_header} starts beyond contig end: {start_pos} >= {bam_contig_length}"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_std[fragment_header] = 0.0
                    continue

                # Ensure positions are within BAM contig bounds
                effective_start = max(0, start_pos)
                effective_end = min(end_pos, bam_contig_length)


                if effective_start >= effective_end:
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_std[fragment_header] = 0.0
                    continue

                # Store valid fragment coordinates
                fragment_coords.append((effective_start, effective_end))
                fragment_headers.append(fragment_header)

        # Vectorized processing for all valid fragments
        if fragment_coords:
            try:
                means, stds = _calculate_fragment_stats_vectorized(
                    total_coverage_per_base, fragment_coords
                )
                
                # Assign results
                for i, fragment_header in enumerate(fragment_headers):
                    fragment_coverage[fragment_header] = float(means[i])
                    fragment_coverage_std[fragment_header] = float(stds[i])
                    
            except Exception as e:
                logger.warning(f"Error in vectorized coverage calculation: {e}")
                # Fallback to individual processing
                for i, (start, end) in enumerate(fragment_coords):
                    fragment_header = fragment_headers[i]
                    try:
                        fragment_coverage_array = total_coverage_per_base[start:end]
                        if fragment_coverage_array.size > 0:
                            fragment_coverage[fragment_header] = float(np.mean(fragment_coverage_array))
                            fragment_coverage_std[fragment_header] = float(np.std(fragment_coverage_array))
                        else:
                            fragment_coverage[fragment_header] = 0.0
                            fragment_coverage_std[fragment_header] = 0.0
                    except Exception as e2:
                        logger.warning(f"Error calculating coverage for fragment {fragment_header}: {e2}")
                        fragment_coverage[fragment_header] = 0.0
                        fragment_coverage_std[fragment_header] = 0.0

    except Exception as e:
        logger.error(f"Error processing contig {bam_contig_name}: {e}")
        # Set all fragments to zero coverage on any error
        for original_header, data in contig_data_list:
            for fragment_header in data["fragments"]:
                fragment_coverage[fragment_header] = 0.0
                fragment_coverage_std[fragment_header] = 0.0

    return fragment_coverage, fragment_coverage_std, []


def _calculate_fragment_stats_vectorized(coverage_array, fragment_coords):
    """
    Calculate mean and std coverage for multiple fragments using vectorized operations.
    This is significantly faster than individual fragment processing.
    
    Args:
        coverage_array: numpy array with coverage per base
        fragment_coords: List of (start, end) tuples for fragment coordinates
    
    Returns:
        Tuple of (means_array, stds_array)
    """
    n_fragments = len(fragment_coords)
    means = np.zeros(n_fragments, dtype=np.float64)
    stds = np.zeros(n_fragments, dtype=np.float64)
    
    # For small numbers of fragments, individual processing might be faster
    if n_fragments < 10:
        for i, (start, end) in enumerate(fragment_coords):
            fragment_coverage = coverage_array[start:end]
            if len(fragment_coverage) > 0:
                means[i] = np.mean(fragment_coverage)
                stds[i] = np.std(fragment_coverage)
        return means, stds
    
    # For larger numbers of fragments, use advanced indexing for speedup
    try:
        all_indices = []
        fragment_starts = []
        fragment_lengths = []
        
        for start, end in fragment_coords:
            length = end - start
            if length > 0:
                all_indices.extend(range(start, end))
                fragment_starts.append(len(all_indices) - length)
                fragment_lengths.append(length)
            else:
                fragment_starts.append(0)
                fragment_lengths.append(0)
        
        if all_indices:
            # Extract all fragment data at once using advanced indexing
            all_fragment_data = coverage_array[all_indices]
            
            # Calculate means and stds using reduceat operations
            cumulative_indices = np.cumsum([0] + fragment_lengths[:-1])
            
            for i, (start_idx, length) in enumerate(zip(cumulative_indices, fragment_lengths)):
                if length > 0:
                    fragment_data = all_fragment_data[start_idx:start_idx + length]
                    means[i] = np.mean(fragment_data)
                    stds[i] = np.std(fragment_data)
        
        return means, stds
        
    except (IndexError, ValueError) as e:
        # Fallback to individual processing if vectorized approach fails
        logger.debug(f"Vectorized processing failed, using fallback: {e}")
        for i, (start, end) in enumerate(fragment_coords):
            fragment_coverage = coverage_array[start:end]
            if len(fragment_coverage) > 0:
                means[i] = np.mean(fragment_coverage)
                stds[i] = np.std(fragment_coverage)
        return means, stds


def calculate_fragment_coverage(
    bam_file: str, fragments_dict: FragmentDict, cores: int = 16
) -> Tuple[CoverageDict, CoverageDict]:
    """Calculate average coverage and standard deviation for each fragment."""
    fragment_coverage: CoverageDict = {}
    fragment_coverage_std: CoverageDict = {}

    if not _validate_alignment_file(bam_file):
        return {}, {}

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bamfile:
            bam_references = set(bamfile.references)
            bam_lengths = dict(zip(bamfile.references, bamfile.lengths))

            if not bam_references:
                logger.error("Alignment file contains no reference sequences.")
                return {}, {}

            # Map FASTA headers to BAM references
            bam_ref_map, unmapped_headers = _map_fasta_to_bam_refs(
                fragments_dict, bam_references
            )

            if unmapped_headers:
                logger.warning(
                    f"{len(unmapped_headers)} FASTA headers could not be matched to BAM references."
                )

            # Group fragments by contig
            contig_fragments = {}
            for original_header, data in fragments_dict.items():
                bam_contig_name = bam_ref_map.get(original_header)
                if bam_contig_name is not None:
                    if bam_contig_name not in contig_fragments:
                        contig_fragments[bam_contig_name] = []
                    contig_fragments[bam_contig_name].append((original_header, data))

            # Pre-load coverage data for all contigs to avoid BAM I/O in workers
            logger.info(f"Pre-loading coverage data for {len(contig_fragments)} contigs...")
            coverage_data = {}
            
            for bam_contig_name in tqdm(contig_fragments.keys(), desc="Loading coverage"):
                bam_contig_length = bam_lengths.get(bam_contig_name)
                if bam_contig_length is None:
                    coverage_data[bam_contig_name] = (None, None)
                    continue
                    
                try:
                    coverage_arrays = bamfile.count_coverage(
                        contig=bam_contig_name,
                        start=0,
                        stop=bam_contig_length,
                        quality_threshold=0,
                    )
                    total_coverage_per_base = np.sum(coverage_arrays, axis=0)
                    coverage_data[bam_contig_name] = (total_coverage_per_base, bam_contig_length)
                except Exception as e:
                    logger.warning(f"Error loading coverage for contig {bam_contig_name}: {e}")
                    coverage_data[bam_contig_name] = (None, None)

            # Process contigs in parallel with pre-loaded coverage data
            logger.info(f"Processing {len(contig_fragments)} contigs using {cores} cores...")
            worker_args = [
                (
                    bam_contig_name,
                    contig_data_list,
                    coverage_data[bam_contig_name][0],  # total_coverage_per_base
                    coverage_data[bam_contig_name][1],  # bam_contig_length
                )
                for bam_contig_name, contig_data_list in contig_fragments.items()
            ]

            with Pool(processes=cores) as pool:
                results = list(
                    tqdm(
                        pool.imap(_process_contig_coverage_worker, worker_args),
                        total=len(worker_args),
                        desc="Processing fragments",
                    )
                )

            # Combine results
            for res_cov, res_std, _ in results:
                fragment_coverage.update(res_cov)
                fragment_coverage_std.update(res_std)

            # Handle unmapped headers
            for original_header in unmapped_headers:
                if original_header in fragments_dict:
                    for fragment_header in fragments_dict[original_header]["fragments"]:
                        fragment_coverage[fragment_header] = 0.0
                        fragment_coverage_std[fragment_header] = 0.0

    except Exception as e:
        logger.error(f"Error processing alignment file {bam_file}: {e}")
        return {}, {}

    # Assign zero coverage to any missing fragments
    all_fragment_headers = {
        fh for data in fragments_dict.values() for fh in data["fragments"]
    }
    missing_fragments = all_fragment_headers - set(fragment_coverage.keys())

    for missing_fh in missing_fragments:
        fragment_coverage[missing_fh] = 0.0
        fragment_coverage_std[missing_fh] = 0.0

    logger.info(
        f"Coverage calculation complete. Total fragments: {len(all_fragment_headers)}"
    )
    return fragment_coverage, fragment_coverage_std


# Coverage calculation functions
def calculate_coverage_from_tsv(
    tsv_files: List[str], fragments_dict: FragmentDict
) -> pd.DataFrame:
    """Calculate coverage from TSV files."""
    # Get all fragment headers for consistent indexing
    all_fragment_headers = [
        fragment_header
        for data in fragments_dict.values()
        for fragment_header in data["fragments"]
    ]

    all_coverage_series = []

    for i, tsv_file in enumerate(tsv_files):
        try:
            coverage_df = pd.read_csv(tsv_file, sep="\t", header=None)
            if len(coverage_df.columns) < 2:
                logger.error(f"TSV file {tsv_file} has fewer than 2 columns")
                continue

            header_coverage = dict(zip(coverage_df[0], coverage_df.iloc[:, -1]))

            # Process each fragment
            fragment_coverage = {}
            for original_header, data in fragments_dict.items():
                # Try to find coverage value for this contig
                coverage_value = 0.0  # Default to 0 if not found

                # Try exact match first
                if original_header in header_coverage:
                    coverage_value = float(header_coverage[original_header])
                else:
                    # Try matching after splitting on space
                    base_header = original_header.split()[0]
                    if base_header in header_coverage:
                        coverage_value = float(header_coverage[base_header])
                    else:
                        logger.warning(
                            f"No coverage found for {original_header} in {tsv_file}. Setting to 0."
                        )

                # Assign same coverage to all fragments from this contig
                for fragment_header in data["fragments"]:
                    fragment_coverage[fragment_header] = coverage_value

            col_name = os.path.splitext(os.path.basename(tsv_file))[0]
            coverage_series = pd.Series(fragment_coverage, name=col_name, dtype=float)
            all_coverage_series.append(coverage_series)

        except pd.errors.EmptyDataError:
            logger.error(f"TSV file {tsv_file} is empty. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error processing TSV file {tsv_file}: {e}. Skipping.")
            continue

    if not all_coverage_series:
        logger.warning("No coverage data could be loaded from any TSV files.")
        return pd.DataFrame(index=all_fragment_headers)

    # Combine all coverage series into a single dataframe
    coverage_features = pd.concat(all_coverage_series, axis=1)

    # Ensure all fragments are present and fill missing values with 0
    coverage_features = coverage_features.reindex(all_fragment_headers).fillna(0.0)

    return coverage_features


def _get_total_mapped_reads(bam_file: str) -> int:
    """Calculate total number of mapped reads in an alignment file.
    
    Args:
        bam_file: Path to alignment file (BAM/CRAM)
        
    Returns:
        Total number of mapped reads
    """
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bamfile:
            total_mapped = bamfile.mapped
            logger.info(f"Alignment file {os.path.basename(bam_file)}: {total_mapped:,} mapped reads")
            return total_mapped
    except Exception as e:
        logger.error(f"Error calculating mapped reads for {bam_file}: {e}")
        return 1  # Avoid division by zero


def calculate_coverage_from_multiple_bams(
    bam_files: List[str], fragments_dict: FragmentDict, cores: int = 16
) -> pd.DataFrame:
    """Calculate coverage from multiple alignment files (BAM/CRAM), creating separate columns for each sample.
    
    Coverage is normalized by total mapped reads per sample to account for different
    sequencing depths, then scaled per-sample to preserve within-sample relationships.

    Args:
        bam_files: List of alignment file paths (BAM/CRAM)
        fragments_dict: Dictionary containing fragment sequences
        cores: Number of cores for processing

    Returns:
        DataFrame with coverage columns for each alignment file (mean and std)
    """
    if not bam_files:
        return pd.DataFrame()

    logger.info(f"Calculating coverage from {len(bam_files)} alignment files...")

    # Get all fragment headers for consistent indexing
    all_fragment_headers = [
        fragment_header
        for data in fragments_dict.values()
        for fragment_header in data["fragments"]
    ]

    all_coverage_series = []

    for i, bam_file in enumerate(bam_files):
        logger.info(
            f"Processing alignment file {i+1}/{len(bam_files)}: {os.path.basename(bam_file)}"
        )

        try:
            # Calculate total mapped reads for normalization
            total_mapped_reads = _get_total_mapped_reads(bam_file)
            
            # Calculate coverage for this alignment file
            coverage, coverage_std = calculate_fragment_coverage(
                bam_file, fragments_dict, cores
            )
            
            # Normalize by total mapped reads (convert to reads per million, RPM)
            normalization_factor = total_mapped_reads / 1_000_000
            normalized_coverage = {k: v / normalization_factor for k, v in coverage.items()}
            normalized_coverage_std = {k: v / normalization_factor for k, v in coverage_std.items()}
            
            logger.info(f"Normalized coverage by {total_mapped_reads:,} mapped reads (factor: {normalization_factor:.2f})")

            sample_name = os.path.splitext(os.path.basename(bam_file))[0]
            mean_col_name = f"{sample_name}_coverage"
            std_col_name = f"{sample_name}_coverage_std"

            mean_series = pd.Series(normalized_coverage, name=mean_col_name, dtype=float)
            std_series = pd.Series(normalized_coverage_std, name=std_col_name, dtype=float)

            all_coverage_series.extend([mean_series, std_series])

        except Exception as e:
            logger.error(f"Error processing alignment file {bam_file}: {e}")
            sample_name = os.path.splitext(os.path.basename(bam_file))[0]
            mean_col_name = f"{sample_name}_coverage"
            std_col_name = f"{sample_name}_coverage_std"

            zero_coverage = {fh: 0.0 for fh in all_fragment_headers}
            mean_series = pd.Series(zero_coverage, name=mean_col_name, dtype=float)
            std_series = pd.Series(zero_coverage, name=std_col_name, dtype=float)

            all_coverage_series.extend([mean_series, std_series])

    if not all_coverage_series:
        logger.warning("No coverage data could be loaded from any alignment files.")
        return pd.DataFrame(index=all_fragment_headers)

    # Combine all coverage series into a single dataframe
    coverage_features = pd.concat(all_coverage_series, axis=1)

    # Ensure all fragments are present and fill missing values with 0
    coverage_features = coverage_features.reindex(all_fragment_headers).fillna(0.0)

    logger.info(
        f"Coverage calculation complete. Created {len(coverage_features.columns)} coverage columns."
    )
    return coverage_features
