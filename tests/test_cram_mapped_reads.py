"""Regression tests for mapped-read normalization across BAM and CRAM."""

import errno
from unittest.mock import MagicMock, Mock

import pysam
import pytest

from remag import features
from remag.features import _get_total_mapped_reads


@pytest.mark.parametrize("file_format", ["bam", "cram"])
@pytest.mark.parametrize(
    "mapped_flags", [[], [0, 16, 256, 512, 1024, 2048, 256 | 1024, 2048 | 512]]
)
@pytest.mark.parametrize("include_unmapped", [False, True])
def test_mapped_read_count_matches_records(
    tmp_path, file_format, mapped_flags, include_unmapped
):
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
        # Count alignments, including repeated names, QC failures and MAPQ zero.
        for index, flag in enumerate(mapped_flags):
            read = pysam.AlignedSegment()
            read.query_name = "repeated_name"
            read.query_sequence = "A" * 100
            read.query_qualities = pysam.qualitystring_to_array("I" * 100)
            read.flag = flag
            read.reference_id = 0
            read.reference_start = index * 100
            read.mapping_quality = 0 if index == 0 else 60
            read.cigar = [(0, 100)]
            handle.write(read)

        if include_unmapped:
            for placed in [True, False]:
                unmapped = pysam.AlignedSegment()
                unmapped.query_name = f"unmapped_{placed}"
                unmapped.query_sequence = "C" * 100
                unmapped.query_qualities = pysam.qualitystring_to_array("I" * 100)
                unmapped.flag = 4
                if placed:
                    unmapped.reference_id = 0
                    unmapped.reference_start = 900
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


@pytest.mark.parametrize("is_cram", [False, True])
def test_count_uses_index_for_bam_and_single_thread_view_for_cram(monkeypatch, is_cram):
    reader = MagicMock(is_cram=is_cram, mapped=7)
    alignment_file = MagicMock()
    alignment_file.return_value.__enter__.return_value = reader
    view = Mock(return_value="11\n")
    monkeypatch.setattr(pysam, "AlignmentFile", alignment_file)
    monkeypatch.setattr(pysam, "view", view)

    assert _get_total_mapped_reads("sample") == (11 if is_cram else 7)
    alignment_file.assert_called_once_with("sample", "rb")
    if is_cram:
        view.assert_called_once_with("-c", "-F", "4", "-@", "0", "sample")
    else:
        view.assert_not_called()
    reader.fetch.assert_not_called()


@pytest.mark.parametrize("failure_source", ["open", "count"])
@pytest.mark.parametrize(
    "error,expected_exception",
    [
        (MemoryError("out of memory"), MemoryError),
        (OSError(errno.ENOMEM, "out of memory"), OSError),
        (OSError(errno.EIO, "read error"), None),
        (pysam.SamtoolsError("count failed"), None),
    ],
)
def test_mapped_count_preserves_error_handling(
    monkeypatch, failure_source, error, expected_exception
):
    alignment_file = MagicMock()
    alignment_file.return_value.__enter__.return_value.is_cram = True
    view = Mock(return_value="0\n")
    if failure_source == "open":
        alignment_file.side_effect = error
    else:
        view.side_effect = error
    monkeypatch.setattr(pysam, "AlignmentFile", alignment_file)
    monkeypatch.setattr(pysam, "view", view)
    logger = Mock()
    monkeypatch.setattr(features, "logger", logger)

    if expected_exception:
        with pytest.raises(expected_exception) as caught:
            _get_total_mapped_reads("sample.cram")
        assert caught.value is error
    else:
        assert _get_total_mapped_reads("sample.cram") == 1
        logger.error.assert_called_once()
