"""Regression tests for mapped-read normalization across BAM and CRAM."""

import pysam
import pytest

from remag.features import _get_total_mapped_reads


@pytest.mark.parametrize("file_format", ["bam", "cram"])
@pytest.mark.parametrize("mapped_flags", [[], [0, 256, 2048, 1024]])
def test_mapped_read_count_matches_records(tmp_path, file_format, mapped_flags):
    """CRAM must count mapped records even though its CRAI reports zero."""
    reference = tmp_path / "reference.fa"
    reference.write_text(">contig\n" + "A" * 1000 + "\n", encoding="utf-8")
    pysam.faidx(str(reference))
    alignment = tmp_path / f"sample.{file_format}"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "contig", "LN": 1000}],
    }
    options = {}
    if file_format == "cram":
        options = {
            "reference_filename": str(reference),
            "format_options": [b"embed_ref=1"],
        }

    mode = "wb" if file_format == "bam" else "wc"
    with pysam.AlignmentFile(str(alignment), mode, header=header, **options) as handle:
        # Include primary, secondary, supplementary, and duplicate alignments:
        # the normalization count must agree with BAM index semantics.
        for index, flag in enumerate(mapped_flags):
            read = pysam.AlignedSegment()
            read.query_name = f"mapped_{index}"
            read.query_sequence = "A" * 100
            read.query_qualities = pysam.qualitystring_to_array("I" * 100)
            read.flag = flag
            read.reference_id = 0
            read.reference_start = index * 100
            read.mapping_quality = 60
            read.cigar = [(0, 100)]
            handle.write(read)

        unmapped = pysam.AlignedSegment()
        unmapped.query_name = "unmapped"
        unmapped.query_sequence = "C" * 100
        unmapped.query_qualities = pysam.qualitystring_to_array("I" * 100)
        unmapped.flag = 4
        handle.write(unmapped)

    pysam.index(str(alignment))
    with pysam.AlignmentFile(str(alignment), "rb") as handle:
        if file_format == "cram":
            assert handle.mapped == 0
        actual_count = sum(
            not read.is_unmapped for read in handle.fetch(until_eof=True)
        )

    assert actual_count == len(mapped_flags)
    assert _get_total_mapped_reads(str(alignment)) == actual_count
