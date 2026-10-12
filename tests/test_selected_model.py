"""Selected model, dedicated sampler and architecture-aware cache regressions."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from remag import models
from remag.model_policy import (
    LEGACY_NO_COVERAGE_ARCHITECTURE,
    PROVENANCE_FILE,
    SAMPLER_POLICY,
    load_cached_embeddings,
    model_configuration,
    save_training_provenance,
    validate_cached_training_settings,
)
from remag.output import prepare_output_directory


@pytest.fixture(autouse=True)
def cpu_training(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(models, "get_torch_device", lambda: torch.device("cpu"))
    yield
    torch.set_num_threads(threads)


def arguments(tmp_path, **kwargs):
    values = dict(
        output=str(tmp_path),
        embedding_dim=16,
        batch_size=4,
        epochs=2,
        base_learning_rate=0.0005,
        barlow_lambda=0.003,
        random_seed=42,
        max_positive_pairs=1000,
        keep_intermediate=True,
        cores=4,
        skip_bacterial_filter=True,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def features(coverage=2, pairs=46):
    names = [f"c{i}{suffix}" for i in range(pairs) for suffix in (".original", ".h1.0")]
    return pd.DataFrame(
        np.random.default_rng(91).normal(size=(len(names), 136 + coverage)), index=names
    )


@pytest.mark.parametrize("coverage", [1, 2, 4, 6, 10, 18, 20, 40])
@pytest.mark.parametrize("dimension", [16, 256])
def test_selected_components_and_preprojection_embedding(coverage, dimension):
    torch.manual_seed(42)
    model = models.SiameseNetwork(136, coverage, dimension)
    assert [m.out_features for m in model.kmer_encoder if isinstance(m, nn.Linear)] == [
        128,
        64,
    ]
    assert [
        m.out_features for m in model.coverage_encoder if isinstance(m, nn.Linear)
    ] == [64, 32]
    assert (
        model.coverage_final_dim == model.fusion_layer.coverage_proj.in_features == 32
    )
    assert model.fusion_layer.kmer_proj.in_features == 64
    assert [
        m.out_features for m in model.projection_head if isinstance(m, nn.Linear)
    ] == [128, 512]
    assert not any("_gate." in key for key in model.state_dict())
    fusion = model.fusion_layer
    assert fusion.residual_weight.requires_grad and fusion.residual_weight.item() == 0.5
    assert (
        fusion.kmer_to_coverage_attn.num_heads
        == fusion.coverage_to_kmer_attn.num_heads
        == 4
    )
    assert len(fusion.multi_scale_fusion) == 3
    model.eval()
    x = torch.randn(3, 136 + coverage)
    with torch.no_grad():
        assert model.get_embedding(x).shape == (3, dimension)
        assert model.forward_one(x).shape == (3, 512)
    if coverage == 18 and dimension == 256:
        assert sum(p.numel() for p in model.parameters()) == 4753377
        assert sum(p.numel() for p in model.projection_head.parameters()) == 99200


def test_coverage_free_path_keeps_native_dimensions():
    model = models.SiameseNetwork(136, 0, 16)
    assert not hasattr(model, "fusion_layer")
    assert [m.out_features for m in model.kmer_encoder if isinstance(m, nn.Linear)] == [
        256,
        128,
    ]
    assert [
        m.out_features for m in model.projection_head if isinstance(m, nn.Linear)
    ] == [512, 512]
    assert (
        model_configuration(136, 0, 16)["architecture"]
        == LEGACY_NO_COVERAGE_ARCHITECTURE
    )


@pytest.mark.parametrize("seed", [42, 7])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_sampler_matches_reference_batches_and_rng(
    tmp_path, monkeypatch, seed, device
):
    args = arguments(tmp_path, random_seed=seed)
    data = features()
    model = models.SiameseNetwork(136, 2, 16)
    trainer = models.TrainingManager(args)
    trainer.device = torch.device(device)
    captured = {}

    def loader_factory(dataset, **kwargs):
        captured.update(kwargs)
        # Check the native GPU worker policy without requiring CUDA or spawning
        # GPU workers in this CPU unit test. Ordering is exercised below.
        kwargs.pop("num_workers", None)
        kwargs.pop("pin_memory", None)
        return DataLoader(dataset, **kwargs)

    monkeypatch.setattr(models, "DataLoader", loader_factory)
    loader, *_ = trainer.setup_training(model, data)
    assert captured.get("num_workers", 0) == (4 if device == "cuda" else 0)
    assert captured.get("pin_memory", False) == (device == "cuda")
    assert captured["shuffle"] is True and loader.drop_last is False
    assert loader.generator is loader.sampler.generator
    reference = DataLoader(loader.dataset, batch_size=4, shuffle=True, drop_last=False)
    generator = torch.Generator().manual_seed(seed)
    reference.generator = reference.sampler.generator = generator
    initial = loader.generator.get_state().clone()
    for _ in range(3):
        # Unrelated model/global RNG draws must not affect batch ordering.
        torch.rand(500)
        actual = list(loader)
        expected = list(reference)
        assert len(actual) == len(expected)
        assert all(
            torch.equal(a, b)
            for batches in zip(actual, expected)
            for a, b in zip(*batches)
        )
        assert torch.equal(loader.generator.get_state(), generator.get_state())
    assert not torch.equal(initial, loader.generator.get_state())


@pytest.mark.parametrize("coverage", [0, 2, 6, 18, 40])
def test_training_checkpoint_and_embedding_round_trip(tmp_path, coverage):
    args = arguments(tmp_path)
    data = features(coverage, pairs=6)
    model = models.train_siamese_network(data, args)
    state = torch.load(tmp_path / "siamese_model.pt", weights_only=True)
    assert set(state) == set(model.state_dict())  # Still a model-only state_dict.
    assert all(torch.isfinite(v).all() for v in state.values())
    fresh = models.generate_embeddings(model, data, args)
    with patch.object(
        models.TrainingManager,
        "setup_training",
        side_effect=AssertionError("must not retrain"),
    ):
        loaded = models.train_siamese_network(data, args)
    assert all(
        torch.equal(value, loaded.state_dict()[name]) for name, value in state.items()
    )
    cached = models.generate_embeddings(loaded, data, args)
    np.testing.assert_array_equal(fresh.values.astype(float), cached.values)
    assert fresh.index.tolist() == cached.index.tolist()
    assert fresh.shape == (6, 16)
    metadata = json.loads((tmp_path / PROVENANCE_FILE).read_text())
    assert metadata["model"] == model_configuration(136, coverage, 16)
    assert metadata["settings"]["sampler_policy"] == SAMPLER_POLICY
    assert metadata["loader"] == dict(
        batch_size=4,
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
        positive_pairs=6,
    )
    (tmp_path / "siamese_model.pt").unlink()
    pd.testing.assert_frame_equal(load_cached_embeddings(data, args), cached)


@pytest.fixture
def cached(tmp_path):
    args = arguments(tmp_path)
    data = features(pairs=4)
    model = models.SiameseNetwork(136, 2, 16)
    torch.save(model.state_dict(), tmp_path / "siamese_model.pt")
    save_training_provenance(model, data, args)
    models.generate_embeddings(model, data, args)
    return args, data, model


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "missing",
        "settings",
        "filter",
        "weights",
        "shape",
        "values",
        "rows",
        "columns",
        "embeddings",
    ],
)
def test_incompatible_caches_are_rejected_without_overwriting(tmp_path, cached, change):
    args, data, model = cached
    path = tmp_path / PROVENANCE_FILE
    metadata = json.loads(path.read_text())
    if change == "legacy":
        metadata["model"]["architecture"] = "learned_gates_adaptive_encoders"
        path.write_text(json.dumps(metadata))
    elif change == "missing":
        path.unlink()
    elif change == "settings":
        args.base_learning_rate = 0.005
    elif change == "filter":
        args.skip_bacterial_filter = False
    elif change in ("weights", "shape"):
        state = model.state_dict()
        if change == "weights":
            state["projection_head.0.weight"].add_(0.01)
        else:
            state["projection_head.0.weight"] = torch.zeros(512, 16)
        torch.save(state, tmp_path / "siamese_model.pt")
    elif change == "values":
        data.iloc[0, 0] += 0.1
    elif change == "rows":
        data = data.iloc[::-1]
    elif change == "columns":
        data.columns = [str(i) + "changed" for i in data.columns]
    else:
        with (tmp_path / "embeddings.csv").open("a") as handle:
            handle.write("unexpected\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="--force"):
        if change == "embeddings":
            load_cached_embeddings(data, args)
        else:
            models.train_siamese_network(data, args)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_legacy_binning_is_rejected_before_touching_outputs(tmp_path):
    args = arguments(tmp_path)
    (tmp_path / "bins.csv").write_text("historical result")
    with pytest.raises(ValueError, match="training provenance"):
        validate_cached_training_settings(args)
    assert (tmp_path / "bins.csv").read_text() == "historical result"


@pytest.mark.parametrize("kind", ["fasta", "coverage"])
def test_changed_inputs_are_rejected_before_cache_reuse(tmp_path, kind):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">c\nACGT\n")
    coverage = tmp_path / "sample.tsv"
    coverage.write_text("sample coverage\n")
    args = arguments(tmp_path, fasta=str(fasta), tsv=[str(coverage)])
    data = features(pairs=4)
    model = models.SiameseNetwork(136, 2, 16)
    models.generate_embeddings(model, data, args)
    changed = fasta if kind == "fasta" else coverage
    changed.write_text(changed.read_text() + "changed")
    with pytest.raises(ValueError, match="input FASTA or coverage"):
        validate_cached_training_settings(args)


def test_force_removes_training_provenance(tmp_path, cached):
    args, _, _ = cached
    args.fasta = str(tmp_path.parent / "input.fa")
    args.force = True
    assert prepare_output_directory(args)
    assert not list(tmp_path.iterdir())


def test_core_reuses_verified_embeddings_without_optional_weights(tmp_path):
    from remag import core

    fasta = tmp_path / "input.fa"
    fasta.write_text(">c0\n" + "A" * 1000 + "\n")
    args = arguments(
        tmp_path,
        fasta=str(fasta),
        min_contig_length=1000,
        num_augmentations=0,
        bam=None,
        tsv=None,
        verbose=False,
    )
    data = features(pairs=4)
    model = models.SiameseNetwork(136, 2, 16)
    embeddings = models.generate_embeddings(model, data, args)
    (tmp_path / "params.json").write_text(json.dumps({"min_contig_length": 1000}))
    assert not (tmp_path / "siamese_model.pt").exists()
    with patch("remag.core.get_features", return_value=(data, {})), patch(
        "remag.core.train_siamese_network"
    ) as train, patch("remag.core.generate_embeddings") as generate, patch(
        "remag.core.load_or_generate_gene_mappings", return_value={}
    ), patch(
        "remag.core.cluster_contigs",
        side_effect=RuntimeError("stop after embedding reuse"),
    ) as cluster:
        with pytest.raises(SystemExit):
            core.main(args)
    train.assert_not_called()
    generate.assert_not_called()
    np.testing.assert_array_equal(cluster.call_args.args[0].values, embeddings.values)


def test_legacy_architecture_rejected_by_strict_load_even_with_updated_file_hash(
    tmp_path, cached
):
    from remag.output import file_sha256

    args, data, model = cached
    state = model.state_dict()
    state["projection_head.0.weight"] = torch.zeros(512, 16)
    state["fusion_layer.kmer_gate.0.weight"] = torch.zeros(16, 32)
    torch.save(state, tmp_path / "siamese_model.pt")
    path = tmp_path / PROVENANCE_FILE
    metadata = json.loads(path.read_text())
    metadata["checkpoint_sha256"] = file_sha256(tmp_path / "siamese_model.pt")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint architecture does not match"):
        models.train_siamese_network(data, args)
