"""Check batch transfers preserve embedding values, exports and optional paths."""

from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from remag.models import generate_embeddings


@pytest.mark.parametrize("keep_intermediate", [False, True])
@pytest.mark.parametrize("has_coverage", [False, True])
def test_batch_transfers_preserve_values_and_exports(
    tmp_path, monkeypatch, keep_intermediate, has_coverage
):
    features = pd.DataFrame(
        np.arange(18).reshape(6, 3) / 7 - 1,
        index=[
            "001.original",
            "NA.0",
            "NA.original",
            "c.1.original",
            "null.original",
            "001.original",
        ],
    )
    original = features[features.index.str.endswith(".original")].copy()
    values = torch.tensor(original.values, dtype=torch.float32)
    # Match the previous implementation's batch shapes and memory layout:
    # PyTorch normalization can round differently for different tensor strides.
    reference_batches = [
        torch.tensor(original.iloc[start : start + 2].values, dtype=torch.float32)
        for start in range(0, len(original), 2)
    ]
    normalized = torch.cat(
        [
            torch.nn.functional.normalize(batch[:, :2] + 0.25, p=2, dim=1)
            for batch in reference_batches
        ]
    )
    names = ["001", "NA", "c.1", "null", "001"]
    # Build the reference rows with the previous per-contig conversion.
    expected = pd.DataFrame.from_dict(
        {name: normalized[i].cpu().numpy() for i, name in enumerate(names)},
        orient="index",
    )
    kmer = pd.DataFrame.from_dict(
        {name: (values[i, :2] * 2).cpu().numpy() for i, name in enumerate(names)},
        orient="index",
    )
    coverage = pd.DataFrame.from_dict(
        {name: (values[i, 2:] * 3).cpu().numpy() for i, name in enumerate(names)},
        orient="index",
    )

    model = Mock()

    def get_embedding(batch):
        assert not torch.is_grad_enabled()
        assert batch.dtype == torch.float32
        assert batch.device.type == "cpu"
        return batch[:, :2] + 0.25

    model.get_embedding.side_effect = get_embedding
    model.get_encoder_embeddings.side_effect = lambda batch: (
        batch[:, :2] * 2,
        batch[:, 2:] * 3 if has_coverage else None,
    )
    monkeypatch.setattr("remag.models.get_torch_device", lambda: torch.device("cpu"))
    transfers = []
    original_cpu = torch.Tensor.cpu

    def record_transfer(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_transfer)
    args = Namespace(
        output=str(tmp_path), batch_size=2, keep_intermediate=keep_intermediate
    )
    actual = generate_embeddings(model, features, args)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert (tmp_path / "embeddings.csv").read_bytes() == expected.astype(
        np.float64
    ).to_csv(float_format="%.17g").encode()
    model.eval.assert_called_once()
    assert [len(call.args[0]) for call in model.get_embedding.call_args_list] == [
        2,
        2,
        1,
    ]
    assert [call.args[0].stride() for call in model.get_embedding.call_args_list] == [
        batch.stride() for batch in reference_batches
    ]
    expected_transfers = []
    for size in [2, 2, 1]:
        expected_transfers.append((size, 2))
        if keep_intermediate:
            expected_transfers.append((size, 2))
            if has_coverage:
                expected_transfers.append((size, 1))
    assert transfers == expected_transfers
    assert [call[0] for call in model.mock_calls] == ["eval"] + (
        ["get_embedding", "get_encoder_embeddings"]
        if keep_intermediate
        else ["get_embedding"]
    ) * 3
    for filename, frame, should_exist in [
        ("kmer_embeddings.csv", kmer, keep_intermediate),
        ("coverage_embeddings.csv", coverage, keep_intermediate and has_coverage),
    ]:
        path = tmp_path / filename
        assert path.exists() == should_exist
        if should_exist:
            assert path.read_bytes() == frame.to_csv().encode()

    # A cached result must bypass model evaluation and retain literal identifiers.
    cached = generate_embeddings(Mock(spec=[]), features, args)
    pd.testing.assert_frame_equal(
        cached, expected.astype(np.float64).rename(columns=str), check_exact=True
    )
    assert transfers == expected_transfers


@pytest.mark.parametrize("keep_intermediate", [False, True])
@pytest.mark.parametrize("headers", [[], ["contig.0"]])
def test_no_original_contigs_preserves_empty_output(
    tmp_path, monkeypatch, keep_intermediate, headers
):
    features = pd.DataFrame(1.0, index=pd.Index(headers, dtype=str), columns=["kmer_0"])
    model = Mock()
    monkeypatch.setattr("remag.models.get_torch_device", lambda: torch.device("cpu"))
    args = Namespace(
        output=str(tmp_path), batch_size=2, keep_intermediate=keep_intermediate
    )

    actual = generate_embeddings(model, features, args)

    pd.testing.assert_frame_equal(actual, pd.DataFrame())
    assert (
        tmp_path / "embeddings.csv"
    ).read_bytes() == pd.DataFrame().to_csv().encode()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["embeddings.csv"]
    model.get_embedding.assert_not_called()
    model.get_encoder_embeddings.assert_not_called()
