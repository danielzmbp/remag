"""Keep actual FASTA names distinct from generated fragment names."""

from argparse import Namespace

import pandas as pd
import pytest

from remag.output import save_clusters_as_fasta
from remag.utils import ContigHeaderMapper, fasta_iter


@pytest.mark.parametrize("name", ["legit.original", "legit.h1.0", "legit.h2.7"])
def test_original_identifier_with_fragment_suffix_is_preserved(name):
    assert ContigHeaderMapper({name: {"sequence": "ACGT"}}).get_header(name) == name


def test_exact_identifier_wins_over_fragment_alias():
    fragments = {"c.original": {"sequence": "CCCC"}, "c": {"sequence": "AAAA"}}
    mapper = ContigHeaderMapper(fragments)
    assert mapper.get_header("c") == "c"
    assert mapper.get_header("c.original") == "c.original"


def test_saved_bin_keeps_all_distinct_input_identifiers(tmp_path):
    fragments = {
        "normal": {"sequence": "AAAA"},
        "legit.original": {"sequence": "CCCC"},
        "legit.h1.0": {"sequence": "GGGG"},
    }
    frame = pd.DataFrame({"contig": list(fragments), "cluster": ["bin_0"] * 3})
    valid = save_clusters_as_fasta(
        frame, fragments, Namespace(output=str(tmp_path), min_bin_size=12)
    )
    assert valid == {"bin_0"}
    assert dict(fasta_iter(str(tmp_path / "bins/bin_0.fa"))) == {
        name: record["sequence"] for name, record in fragments.items()
    }
