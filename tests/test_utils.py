from remag.utils import is_gzipped


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
