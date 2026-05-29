import pandas as pd

from remag.utils import group_contigs_by_cluster, is_gzipped


def test_is_gzipped_true():
    """Test files with .gz extension."""
    assert is_gzipped("file.gz") is True
    assert is_gzipped("file.tar.gz") is True
    assert is_gzipped("path/to/file.gz") is True
    assert is_gzipped("/absolute/path/to/archive.gz") is True
    assert is_gzipped(".gz") is True


def test_is_gzipped_false():
    """Test files without .gz extension."""
    assert is_gzipped("file.txt") is False
    assert is_gzipped("file.tar") is False
    assert is_gzipped("file.fasta") is False
    assert is_gzipped("path/to/file.gz.txt") is False
    # The function uses .endswith('.gz') which is case-sensitive
    assert is_gzipped("file.GZ") is False
    assert is_gzipped("file.tar.GZ") is False


def test_is_gzipped_edge_cases():
    """Test edge cases like empty string."""
    assert is_gzipped("") is False
    assert is_gzipped(" ") is False
    assert is_gzipped("gz") is False


def test_group_contigs_by_cluster_normal():
    """Test normal grouping with multiple clusters and contigs."""
    df = pd.DataFrame(
        {"contig": ["c1", "c2", "c3", "c4", "c5"], "cluster": [1, 1, 2, 2, 2]}
    )

    result = group_contigs_by_cluster(df)

    assert len(result) == 2
    assert result[1] == {"c1", "c2"}
    assert result[2] == {"c3", "c4", "c5"}


def test_group_contigs_by_cluster_empty():
    """Test grouping with an empty dataframe."""
    df = pd.DataFrame(columns=["contig", "cluster"])

    result = group_contigs_by_cluster(df)

    assert len(result) == 0
    assert result == {}


def test_group_contigs_by_cluster_single():
    """Test grouping where each cluster has only one contig."""
    df = pd.DataFrame({"contig": ["c1", "c2", "c3"], "cluster": [1, 2, 3]})

    result = group_contigs_by_cluster(df)

    assert len(result) == 3
    assert result[1] == {"c1"}
    assert result[2] == {"c2"}
    assert result[3] == {"c3"}


def test_group_contigs_by_cluster_string_ids():
    """Test grouping with string cluster IDs instead of integers."""
    df = pd.DataFrame({"contig": ["c1", "c2", "c3"], "cluster": ["A", "B", "A"]})

    result = group_contigs_by_cluster(df)

    assert len(result) == 2
    assert result["A"] == {"c1", "c3"}
    assert result["B"] == {"c2"}
