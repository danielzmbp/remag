"""Guard training batches against singleton BatchNorm failures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from remag import models


def make_features(pair_count):
    headers = [
        f"c{i}{suffix}" for i in range(pair_count) for suffix in [".original", ".h1.0"]
    ]
    return pd.DataFrame(
        np.random.default_rng(42).random((len(headers), 136)), index=headers
    )


def setup(pair_count, batch_size, monkeypatch):
    monkeypatch.setattr(models, "get_torch_device", lambda: torch.device("cpu"))
    args = SimpleNamespace(
        batch_size=batch_size,
        max_positive_pairs=1000,
        epochs=2,
        base_learning_rate=0.005,
        barlow_lambda=0.003,
    )
    model = models.SiameseNetwork(136, 0, embedding_dim=16)
    trainer = models.TrainingManager(args)
    loader, optimizer, scheduler, criterion = trainer.setup_training(
        model, make_features(pair_count)
    )
    return model, trainer, loader, optimizer, criterion


@pytest.mark.parametrize("batch_size", [1, 0, -1])
def test_invalid_batch_size_has_clear_error(monkeypatch, batch_size):
    with pytest.raises(ValueError, match="batch size must be at least 2"):
        setup(4, batch_size, monkeypatch)


def test_one_positive_pair_has_clear_error(monkeypatch):
    with pytest.raises(ValueError, match="At least two positive pairs"):
        setup(1, 2048, monkeypatch)


@pytest.mark.parametrize("batch_size", [3, 7, 2048])
def test_reduced_batches_never_reach_one(monkeypatch, batch_size):
    _, _, loader, _, _ = setup(2, batch_size, monkeypatch)
    assert [len(first) for first, _ in loader] == [2]


@pytest.mark.parametrize("pair_count,batch_size", [(23, 2), (45, 4), (353, 32)])
def test_final_singleton_is_dropped(monkeypatch, pair_count, batch_size):
    _, _, loader, _, _ = setup(pair_count, batch_size, monkeypatch)
    sizes = [len(first) for first, _ in loader]
    assert sizes == [batch_size] * (pair_count // batch_size)
    assert sum(sizes) == pair_count - 1


@pytest.mark.parametrize(
    "pair_count,batch_size,expected",
    [(24, 2, [2] * 12), (46, 4, [4] * 11 + [2]), (7, 4, [4]), (5, 16, [4])],
)
def test_existing_valid_batching_is_preserved(
    monkeypatch, pair_count, batch_size, expected
):
    _, _, loader, _, _ = setup(pair_count, batch_size, monkeypatch)
    assert [len(first) for first, _ in loader] == expected


@pytest.mark.parametrize("pair_count,batch_size", [(2, 3), (23, 2)])
def test_previously_crashing_cases_complete_a_training_epoch(
    monkeypatch, pair_count, batch_size
):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        model, trainer, loader, optimizer, criterion = setup(
            pair_count, batch_size, monkeypatch
        )
        loss, _ = trainer.train_epoch(model, loader, optimizer, criterion)
        assert np.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    finally:
        torch.set_num_threads(original_threads)
