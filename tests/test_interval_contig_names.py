"""Regression tests for exact contig names in interval coverage."""

import pytest

from remag.features import calculate_coverage_from_tsv


@pytest.mark.parametrize(
    "headers, expected",
    [
        (("c.1", "c"), {"c.1.original": 0.0, "c.original": 9.0}),
        (("c", "c.1"), {"c.original": 9.0, "c.1.original": 0.0}),
        (("c.1",), {"c.1.original": 9.0}),
    ],
    ids=["alias-before-exact", "exact-before-alias", "alias-without-exact"],
)
def test_interval_coverage_prefers_exact_contig_names(tmp_path, headers, expected):
    fragments_dict = {
        header: {
            "sequence": "AAAA",
            "fragments": [f"{header}.original"],
            "fragment_info": {
                f"{header}.original": {"start_pos": 0, "length": 4},
            },
        }
        for header in headers
    }
    coverage_file = tmp_path / "sample.bedgraph"
    coverage_file.write_text("c\t0\t4\t9\n", encoding="utf-8")

    coverage = calculate_coverage_from_tsv([str(coverage_file)], fragments_dict)

    assert coverage["sample_coverage"].to_dict() == expected
    assert (coverage["sample_coverage_std"] == 0.0).all()
