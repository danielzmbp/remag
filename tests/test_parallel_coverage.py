"""Coverage arithmetic and failure handling for independent alignment readers."""

import errno
import os
from argparse import Namespace
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import monotonic
from unittest.mock import Mock

import numpy as np
import pysam
import pytest

from remag import features as F


@pytest.fixture(params=["bam", "cram"])
def alignment(tmp_path, request):
    reference = tmp_path / "reference.fa"
    reference.write_text(
        "".join(f">{n}\n" + "ACGT" * 3000 + "\n" for n in ["chr.1", "chr2", "empty"])
    )
    pysam.faidx(str(reference))
    path = tmp_path / f"reads.{request.param}"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": n, "LN": 12000} for n in ["chr.1", "chr2", "empty"]],
    }
    options = (
        {}
        if request.param == "bam"
        else {"reference_filename": str(reference), "format_options": [b"embed_ref=1"]}
    )
    with pysam.AlignmentFile(
        str(path), "wb" if request.param == "bam" else "wc", header=header, **options
    ) as out:
        for rid in [0, 1]:
            for i in range(100):
                read = pysam.AlignedSegment()
                read.query_name = f"r{rid}_{i}"
                read.query_sequence = (
                    ("NNNN" + "ACGT" * 7)[:30] if i % 7 == 0 else ("ACGT" * 8)[:30]
                )
                read.query_qualities = pysam.qualitystring_to_array(
                    ("!" if i % 5 == 0 else "I") * 30
                )
                read.flag = [0, 16, 256, 512, 1024, 2048][i % 6]
                read.reference_id = rid
                read.reference_start = i * 10
                read.mapping_quality = [0, 60][i % 2]
                read.cigarstring = ["30M", "10M3D20M", "10M100N20M", "25M5S"][i % 4]
                out.write(read)
    pysam.index(str(path))
    return str(path)


@pytest.fixture
def fragments():
    records = OrderedDict()
    for name in ["chr.1", "chr2 description", "chr.1.aug", "empty", "missing"]:
        coords = [
            (0, 12000),
            (0, 1),
            (5, 29),
            (20, 100),
            (900, 200),
            (11990, 30),
            (-1, 10),
            (12000, 10),
            (100, 0),
            (50, 30),
            (100, 200),
            (50, 100),
        ]
        names = [f"{name}.f{i}" for i in range(len(coords) + 1)]
        records[name] = {
            "sequence": "ACGT" * 3000,
            "fragments": names,
            "fragment_info": {
                names[i]: {"start_pos": start, "length": length}
                for i, (start, length) in enumerate(coords)
            },
        }
    return records


def serial_reference(alignment, fragments):
    """Original depth call and unchanged statistics, before any parallel transport."""
    means, stds = {}, {}
    with pysam.AlignmentFile(alignment, "rb") as reader:
        mapping, unmatched = F._map_fasta_to_bam_refs(
            fragments, set(reader.references), True
        )
        groups = {}
        for name, data in fragments.items():
            if mapping[name] is not None:
                groups.setdefault(mapping[name], []).append((name, data))
        for name, records in groups.items():
            length = reader.get_reference_length(name)
            depth = np.sum(
                reader.count_coverage(
                    contig=name, start=0, stop=length, quality_threshold=0
                ),
                axis=0,
            )
            mean, std, _ = F._process_contig_coverage_worker(
                (name, records, depth, length)
            )
            means.update(mean)
            stds.update(std)
    for name in unmatched:
        for fragment in fragments[name]["fragments"]:
            means[fragment] = stds[fragment] = 0.0
    return means, stds


@pytest.mark.parametrize("cores,batch_size", [(1, 100000), (4, 100000), (16, 2)])
def test_parallel_and_serial_coverage_are_bitwise_identical(
    alignment, fragments, cores, batch_size
):
    expected = serial_reference(alignment, fragments)
    actual = F.calculate_fragment_coverage(
        alignment, fragments, cores, batch_size, True
    )
    for old, new in zip(expected, actual):
        assert list(new) == list(old)
        assert (
            np.array(list(new.values())).tobytes()
            == np.array(list(old.values())).tobytes()
        )
    assert any(v > 0 for v in actual[0].values())


def test_all_unmatched_fragments_are_zero_filled(alignment, fragments):
    records = {"missing": fragments["missing"]}
    means, stds = F.calculate_fragment_coverage(
        alignment, records, disable_progress=True
    )
    assert list(means) == records["missing"]["fragments"]
    assert all(value == 0 for value in means.values())
    assert means == stds
    assert F.calculate_fragment_coverage(alignment, {}, disable_progress=True) == (
        {},
        {},
    )


def test_worker_read_error_retains_zero_fallback(fragments):
    reader = Mock()
    reader.count_coverage.side_effect = OSError("bad contig")
    means, stds, _ = F._read_contig_coverage(
        ("chr.1", [("chr.1", fragments["chr.1"])], 12000), reader
    )
    assert list(means) == fragments["chr.1"]["fragments"]
    assert all(value == 0 for value in means.values())
    assert means == stds


def test_invalid_alignment_retains_empty_result(tmp_path, fragments):
    assert F.calculate_fragment_coverage(str(tmp_path / "absent.bam"), fragments) == (
        {},
        {},
    )


def test_missing_index_is_created(alignment, fragments):
    suffix = ".bai" if alignment.endswith("bam") else ".crai"
    os.unlink(alignment + suffix)
    means, _ = F.calculate_fragment_coverage(
        alignment, fragments, cores=1, disable_progress=True
    )
    assert os.path.isfile(alignment + suffix)
    assert any(v > 0 for v in means.values())


def test_sequence_strings_are_not_sent_to_workers(alignment, fragments, monkeypatch):
    def results(tasks, path, readers, batch_size):
        assert readers == 3  # Only three mapped references are available.
        with pysam.AlignmentFile(path, "rb") as reader:
            for task in tasks:
                assert all(
                    set(data) == {"fragments", "fragment_info"} for _, data in task[1]
                )
                yield F._read_contig_coverage(task, reader)

    monkeypatch.setattr(F, "_parallel_coverage_results", results)
    assert F.calculate_fragment_coverage(
        alignment, fragments, cores=16, disable_progress=True
    ) == serial_reference(alignment, fragments)


def _exit_worker(tasks):
    os._exit(17)


def test_killed_worker_does_not_stall_or_return_partial_coverage(
    alignment, fragments, monkeypatch
):
    # Spawn exercises independent imports/handles and works on macOS and Linux.
    def executor(**kwargs):
        return ProcessPoolExecutor(mp_context=get_context("spawn"), **kwargs)

    monkeypatch.setattr(F, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(F, "_read_coverage_chunk", _exit_worker)
    start = monotonic()
    with pytest.raises(F.CoverageError, match="worker failed.*reduce --cores"):
        F.calculate_fragment_coverage(
            alignment, fragments, cores=2, disable_progress=True
        )
    assert monotonic() - start < 30


@pytest.mark.parametrize("batch_size", [1, 2, 17, 128, 100000])
@pytest.mark.parametrize("readers", [2, 8, 16])
def test_pending_metadata_and_results_are_bounded(monkeypatch, batch_size, readers):
    state = {"submitted": 0, "finished": 0}

    class Future:
        def __init__(self, chunk):
            self.chunk = chunk

        def result(self):
            state["finished"] += len(self.chunk)
            return self.chunk

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == readers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, chunk):
            state["submitted"] += len(chunk)
            assert state["submitted"] - state["finished"] <= min(
                32 * readers, batch_size
            )
            return Future(chunk)

    monkeypatch.setattr(F, "ProcessPoolExecutor", Executor)
    assert list(
        F._parallel_coverage_results(
            iter(range(1000)), "reads.bam", readers, batch_size
        )
    ) == list(range(1000))


@pytest.mark.parametrize("cores", [4, 8, 16])
def test_multiple_chunks_preserve_fragment_order(tmp_path, monkeypatch, cores):
    path = str(tmp_path / "many.bam")
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": f"c{i}", "LN": 128} for i in range(48)],
    }
    with pysam.AlignmentFile(path, "wb", header=header) as handle:
        for i in range(48):
            read = pysam.AlignedSegment()
            read.query_name = f"r{i}"
            read.query_sequence = "A" * 100
            read.query_qualities = pysam.qualitystring_to_array("I" * 100)
            read.reference_id = i
            read.reference_start = i % 28
            read.mapping_quality = 60
            read.cigarstring = "100M"
            handle.write(read)
    pysam.index(path)
    records = {
        f"c{i}": {
            "sequence": "A" * 128,
            "fragments": [f"c{i}.original"],
            "fragment_info": {f"c{i}.original": {"start_pos": 0, "length": 128}},
        }
        for i in reversed(range(48))
    }
    expected = serial_reference(path, records)
    executor = Mock(wraps=ProcessPoolExecutor)
    warning = Mock()
    monkeypatch.setattr(F, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(F.logger, "warning", warning)
    result = F.calculate_fragment_coverage(
        path, records, cores=cores, disable_progress=True
    )
    assert executor.call_args.kwargs["max_workers"] == cores
    if cores > 4:
        assert "fewer --cores" in warning.call_args.args[0]
    else:
        warning.assert_not_called()
    for original, actual in zip(expected, result):
        assert list(actual.items()) == list(original.items())


def _fail_initializer(path):
    raise OSError("cannot open worker alignment")


def test_worker_initialization_failure_returns_promptly(
    alignment, fragments, monkeypatch
):
    monkeypatch.setattr(F, "_init_coverage_reader", _fail_initializer)
    start = monotonic()
    with pytest.raises(F.CoverageError, match="worker failed.*reduce --cores"):
        F.calculate_fragment_coverage(
            alignment, fragments, cores=2, disable_progress=True
        )
    assert monotonic() - start < 30


@pytest.mark.parametrize(
    "error", [MemoryError(), OSError(errno.ENOMEM, "out of memory")]
)
def test_reader_memory_failure_is_not_zero_coverage(fragments, error):
    reader = Mock()
    reader.count_coverage.side_effect = error
    with pytest.raises(type(error)):
        F._read_contig_coverage(
            ("chr.1", [("chr.1", fragments["chr.1"])], 12000), reader
        )


@pytest.mark.parametrize("fallback", [False, True])
def test_statistics_memory_failure_is_not_zero_coverage(
    fragments, monkeypatch, fallback
):
    monkeypatch.setattr(
        F,
        "_calculate_fragment_stats_vectorized",
        Mock(side_effect=ValueError("use fallback") if fallback else MemoryError()),
    )
    if fallback:
        monkeypatch.setattr(F.np, "mean", Mock(side_effect=MemoryError()))
    with pytest.raises(MemoryError):
        F._process_contig_coverage_worker(
            ("chr.1", [("chr.1", fragments["chr.1"])], np.zeros(12000), 12000)
        )


def _out_of_memory(tasks):
    raise MemoryError("injected allocation failure")


@pytest.mark.parametrize("cores", [1, 2])
def test_memory_failure_stops_samples(alignment, fragments, monkeypatch, cores):
    if cores == 1:
        monkeypatch.setattr(F, "_read_contig_coverage", Mock(side_effect=MemoryError()))
    else:
        monkeypatch.setattr(F, "_read_coverage_chunk", _out_of_memory)
    counts = Mock(return_value=100)
    monkeypatch.setattr(F, "_get_total_mapped_reads", counts)
    with pytest.raises(F.CoverageError, match="out of memory.*reduce --cores"):
        F.calculate_coverage_from_multiple_bams(
            [alignment, "must-not-be-processed.bam"], fragments, cores=cores
        )
    counts.assert_called_once_with(alignment)


@pytest.mark.parametrize(
    "error", [MemoryError(), OSError(errno.ENOMEM, "out of memory")]
)
def test_read_count_memory_failure_is_fatal(alignment, fragments, monkeypatch, error):
    monkeypatch.setattr(F.pysam, "AlignmentFile", Mock(side_effect=error))
    with pytest.raises(F.CoverageError, match="reduce --cores"):
        F.calculate_coverage_from_multiple_bams([alignment], fragments, cores=1)


def test_coverage_memory_failure_stops_before_training(
    tmp_path, alignment, monkeypatch
):
    from remag import core

    fasta = tmp_path / "input.fa"
    fasta.write_text(">chr.1\n" + "ACGT" * 3000 + "\n")
    output = tmp_path / "out"
    args = Namespace(
        output=str(output),
        fasta=str(fasta),
        verbose=False,
        skip_bacterial_filter=True,
        min_contig_length=1000,
        num_augmentations=0,
        cores=1,
        bam=[alignment],
        tsv=None,
        keep_intermediate=True,
    )
    train = Mock()
    error_log = Mock()
    monkeypatch.setattr(core, "setup_logging", Mock())
    monkeypatch.setattr(core, "train_siamese_network", train)
    monkeypatch.setattr(core.logger, "error", error_log)
    monkeypatch.setattr(F, "_read_contig_coverage", Mock(side_effect=MemoryError()))
    with pytest.raises(SystemExit) as caught:
        core.main(args)
    assert caught.value.code == 1
    train.assert_not_called()
    assert any("reduce --cores" in call.args[0] for call in error_log.call_args_list)
    assert not (output / "features.csv").exists()
