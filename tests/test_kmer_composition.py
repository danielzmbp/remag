"""Focused integration checks for exact k-mer features and the generic fallback."""

import numpy as np
import pandas as pd
import pytest

from remag.features import _calculate_kmer_composition, generate_feature_mapping


@pytest.mark.parametrize("kmer_len", [3, 4, 5])
@pytest.mark.parametrize("pseudocount", [0.0, 1e-5, 1.0])
def test_composition_matches_string_counting(kmer_len, pseudocount):
    sequences = [
        ("001", "atgcatgcAAAAATTTTT"),
        ("NA", "atgcNNNCGTA-RYSWKMBDHVN"),
        ("null", "ATGCßATGC🙂ATGC"),
        ("short", "AT"),
        ("empty", ""),
        ("invalid", "NNNNNNNN"),
        ("repeat", "AAAAAAA"),
        ("last", "TGCAT"),
        ("repeat", "CCCCC"),
    ]
    mapping, size = generate_feature_mapping(kmer_len)
    # Independent string-based counting checks the integrated vectorized path.
    rows = {}
    for name, sequence in sequences:
        sequence = sequence.upper()
        counts = [0] * size
        for start in range(len(sequence) - kmer_len + 1):
            window = sequence[start : start + kmer_len]
            if window in mapping:
                counts[mapping[window]] += 1
        rows[name] = counts
    expected = pd.DataFrame.from_dict(rows, orient="index", dtype=float)
    expected.columns = [str(column) for column in expected.columns]
    expected += pseudocount
    row_sums = expected.sum(axis=1)
    non_zero = row_sums > 1e-9
    expected[non_zero] = expected[non_zero].div(row_sums[non_zero], axis=0)
    expected[~non_zero] = 0.0

    actual = _calculate_kmer_composition(sequences, kmer_len, pseudocount)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert actual.index.tolist() == list(rows)
    if kmer_len == 4:
        assert actual.shape[1] == 136


def test_reverse_complement_features_share_the_existing_column():
    actual = _calculate_kmer_composition(
        [("forward", "AAAA"), ("reverse", "TTTT")], pseudocount=0
    )
    assert actual.loc["forward", "0"] == actual.loc["reverse", "0"] == 1.0
    np.testing.assert_array_equal(actual.sum(axis=1).to_numpy(), [1.0, 1.0])


def test_empty_composition_remains_an_empty_dataframe():
    pd.testing.assert_frame_equal(_calculate_kmer_composition([]), pd.DataFrame())
