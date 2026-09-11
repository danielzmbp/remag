"""Guard against overwriting retained contigs with the same FASTA identifier."""

import pytest

from remag.features import get_features


@pytest.mark.parametrize("second_sequence", ["AAAA", "CCCC"])
def test_duplicate_id_stops_feature_generation(tmp_path, second_sequence):
    fasta = tmp_path / "input.fa"
    fasta.write_text(f">same first\nAAAA\n>same\tsecond\n{second_sequence}\n")

    with pytest.raises(ValueError, match="Duplicate FASTA identifier: 'same'"):
        get_features(
            str(fasta),
            None,
            None,
            str(tmp_path),
            min_contig_length=1,
            num_augmentations=0,
        )

    assert not (tmp_path / "fragments.pkl").exists()


def test_distinct_ids_preserve_identical_sequences(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">contig.1 first\nAAAA\n>contig.2 second\nAAAA\n")

    features, fragments = get_features(
        str(fasta),
        None,
        None,
        str(tmp_path),
        min_contig_length=1,
        num_augmentations=0,
    )

    assert {name: data["sequence"] for name, data in fragments.items()} == {
        "contig.1": "AAAA",
        "contig.2": "AAAA",
    }
    assert set(features.index) == {"contig.1.original", "contig.2.original"}
