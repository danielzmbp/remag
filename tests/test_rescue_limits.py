from unittest.mock import Mock

import numpy as np
import pandas as pd

from remag.rescue import rescue_fragmented_bins


def test_rescue_limits():
    """
    Test rescue limits:
    1. Max duplication INCREASE (relaxed to 5%)
    2. Max TOTAL duplication (strict 5%)
    """
    # Setup Data
    clusters_df = pd.DataFrame(
        {
            "contig": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "cluster": ["bin1", "bin1", "bin2", "bin3", "bin4", "bin5"],
        }
    )

    # Embeddings (Perfect matches for merging)
    embeddings = np.zeros((6, 2))
    embeddings[0:2] = [1.0, 0.0]  # bin1
    embeddings[2] = [1.0, 0.0]  # bin2 (match bin1)
    embeddings[3] = [0.0, 1.0]  # bin3
    embeddings[4] = [0.0, 1.0]  # bin4 (match bin3)
    embeddings[5] = [0.5, 0.5]  # bin5 (distinct)

    embeddings_df = pd.DataFrame(embeddings, index=["c1", "c2", "c3", "c4", "c5", "c6"])

    fragments_dict = {c: {"sequence": "A" * 1000} for c in clusters_df["contig"]}
    # Make target bins larger to ensure merge direction (small -> large)
    fragments_dict["c1"] = {"sequence": "A" * 10000}  # bin1 (target)
    fragments_dict["c2"] = {"sequence": "A" * 10000}  # bin1 (target)
    fragments_dict["c4"] = {"sequence": "A" * 10000}  # bin3 (target)
    # c3 (bin2) and c5 (bin4) stay small (1000)

    # Setup Gene Mappings
    args = Mock()
    gene_mappings = {}

    # Case 1: 0% -> 4% merge.
    # Should be ACCEPTED with max_increase=5.0 (previously rejected by 3.0)
    # bin3 (c4): 100 unique genes. 0 dup.
    # bin4 (c5): 4 genes that are also in c4.
    # Merge result: 104 total entries, 100 unique. 4 dups. 4% rate.

    c4_genes = {f"g3_{i}": {} for i in range(100)}
    c5_genes = {f"g3_{i}": {} for i in range(4)}  # 4 dups
    gene_mappings["c4"] = c4_genes
    gene_mappings["c5"] = c5_genes

    # Case 2: 5% -> 7% merge.
    # Should be REJECTED by max_total=5.0.
    # bin1 (c1, c2): 100 total genes. 95 unique. 5 dups. (5%)
    # bin2 (c3): Adds 2 more dups. Result 7%.

    c1_genes = {f"g1_{i}": {} for i in range(50)}
    c2_genes = {f"g1_{i}": {} for i in range(45, 95)}  # 5 overlaps (45,46,47,48,49).
    # c1: 0..49. c2: 45..94.
    # Total unique: 0..94 (95 genes).
    # Total count: 50 + 50 = 100.
    # Dups = 5. Rate = 5/100? No, get_bin_scg_stats uses dups/present.
    # Present = 95. Dups = 5. Rate = 5/95 = 5.26%.
    # That's > 5%. Let's adjust.
    # We want exactly 5% total duplication? Or just "start at 5%". 5.26% is fine.
    # Increase to 7%.
    # c3 adds genes that are already in c1/c2.
    c3_genes = {f"g1_{i}": {} for i in range(10, 12)}  # 2 genes.
    # New dups: 5 + 2 = 7.
    # New present: still 95 (genes already exist).
    # New rate: 7/95 = 7.3%.

    gene_mappings["c1"] = c1_genes
    gene_mappings["c2"] = c2_genes
    gene_mappings["c3"] = c3_genes

    # bin5 (c6) just noise
    gene_mappings["c6"] = {}

    args._gene_mappings_cache = gene_mappings

    # Execute Rescue
    # We pass explicit parameters to simulate the CLI update
    result_df = rescue_fragmented_bins(
        clusters_df.copy(),
        embeddings_df,
        fragments_dict,
        args,
        similarity_threshold=0.9,
        max_duplication_increase=5.0,  # Relaxed from 3.0
        max_total_duplication=5.0,  # New constraint
    )

    # Check Case 1: c5 should merge into bin3
    final_c5 = result_df[result_df["contig"] == "c5"]["cluster"].values[0]
    assert (
        final_c5 == "bin3"
    ), f"Case 1 Failed: 0%->4% merge was blocked. Got {final_c5}"

    # Check Case 2: c3 should NOT merge into bin1
    final_c3 = result_df[result_df["contig"] == "c3"]["cluster"].values[0]
    assert (
        final_c3 == "bin2"
    ), f"Case 2 Failed: 5%->7% merge was accepted. Got {final_c3}"


def test_rescue_blocks_bin_merge_above_ten_percent_duplication():
    """Bin-to-bin rescue merge must be blocked when duplication exceeds 10%."""
    clusters_df = pd.DataFrame(
        {
            "contig": ["c1", "c2"],
            "cluster": ["bin1", "bin2"],
        }
    )

    embeddings_df = pd.DataFrame(
        [[1.0, 0.0], [1.0, 0.0]],
        index=["c1", "c2"],
    )

    fragments_dict = {
        "c1": {"sequence": "A" * 10000},
        "c2": {"sequence": "A" * 1000},
    }

    args = Mock()
    args._gene_mappings_cache = {
        "c1": {f"g{i}": {} for i in range(100)},
        "c2": {f"g{i}": {} for i in range(11)},
    }

    result_df = rescue_fragmented_bins(
        clusters_df.copy(),
        embeddings_df,
        fragments_dict,
        args,
        similarity_threshold=0.9,
        max_duplication_increase=20.0,
        max_total_duplication=20.0,
    )

    final_c2 = result_df[result_df["contig"] == "c2"]["cluster"].values[0]
    assert final_c2 == "bin2", "Merge with 11% duplication should be blocked."


if __name__ == "__main__":
    test_rescue_limits()
