"""Architecture identity and provenance for model and embedding cache reuse."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .filter_policy import HYENADNA_PREDICTION_POLICY
from .output import file_sha256

SELECTED_ARCHITECTURE = "fixed_small_projection128_v1"
LEGACY_NO_COVERAGE_ARCHITECTURE = "legacy_kmer_only_v1"
SAMPLER_POLICY = "shared_dataloader_randomsampler_generator_v1"
PROVENANCE_FILE = "training_provenance.json"


def model_configuration(n_kmer_features, n_coverage_features, embedding_dim):
    covered = n_coverage_features > 0
    return {
        "architecture": (
            SELECTED_ARCHITECTURE if covered else LEGACY_NO_COVERAGE_ARCHITECTURE
        ),
        "n_kmer_features": n_kmer_features,
        "n_coverage_features": n_coverage_features,
        "kmer_hidden": [128, 64] if covered else [256, 128],
        "coverage_hidden": [64, 32] if covered else [],
        "fusion_gates": 0.5 if covered else None,
        "embedding_dim": embedding_dim,
        "projection_hidden": 128 if covered else 512,
        "projection_output": 512,
        "initialization": "native_then_selected_replacements_v1",
    }


def training_settings(args):
    return {
        **{
            key: getattr(args, key, None)
            for key in (
                "embedding_dim",
                "epochs",
                "batch_size",
                "base_learning_rate",
                "barlow_lambda",
                "max_positive_pairs",
                "num_augmentations",
                "min_contig_length",
                "cores",
            )
        },
        "random_seed": getattr(args, "random_seed", 42),
        "pair_selection_seed": 42,
        "sampler_policy": SAMPLER_POLICY,
        "filter_policy": (
            None
            if getattr(args, "skip_bacterial_filter", False)
            else HYENADNA_PREDICTION_POLICY
        ),
    }


def input_identity(args):
    """Hash FASTA content; identify large coverage inputs by path/size/mtime."""
    fasta = getattr(args, "fasta", None)
    coverage = []
    for path in getattr(args, "bam", None) or getattr(args, "tsv", None) or []:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        coverage.append([str(resolved), stat.st_size, stat.st_mtime_ns])
    return {
        "fasta_sha256": file_sha256(fasta) if fasta else None,
        "coverage_files": coverage,
    }


def feature_identity(features):
    # Training consumes float32; hash rows in bounded chunks without changing
    # their order or the layout of the arrays supplied to training/inference.
    digest = hashlib.sha256(
        json.dumps([list(map(str, features.columns)), list(features.index)]).encode()
    )
    for start in range(0, len(features), 4096):
        digest.update(
            np.ascontiguousarray(features.iloc[start : start + 4096], dtype="<f4")
        )
    return digest.hexdigest()


def state_identity(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(
            json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        )
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def incompatible_training_cache(detail):
    return ValueError(
        f"Existing model or embedding outputs are incompatible: {detail}. "
        "Choose a new output directory to preserve them, or use --force to recompute."
    )


def read_training_provenance(args):
    try:
        value = json.loads((Path(args.output) / PROVENANCE_FILE).read_text())
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise ValueError("unknown schema")
        return value
    except (OSError, ValueError) as error:
        raise incompatible_training_cache(
            "missing or invalid training provenance"
        ) from error


def validate_training_provenance(args, features=None, model=None):
    previous = read_training_provenance(args)
    config = previous.get("model", {})
    try:
        expected = model_configuration(
            model.n_kmer_features if model is not None else config["n_kmer_features"],
            (
                model.n_coverage_features
                if model is not None
                else config["n_coverage_features"]
            ),
            model.representation_dim if model is not None else config["embedding_dim"],
        )
    except (KeyError, TypeError) as error:
        raise incompatible_training_cache("missing model dimensions") from error
    if config != expected or previous.get("settings") != training_settings(args):
        raise incompatible_training_cache(
            "architecture or training/filter settings changed"
        )
    if previous.get("inputs") != input_identity(args):
        raise incompatible_training_cache("input FASTA or coverage files changed")
    checkpoint = Path(args.output) / "siamese_model.pt"
    if checkpoint.exists() and previous.get("checkpoint_sha256") != file_sha256(
        checkpoint
    ):
        raise incompatible_training_cache("checkpoint file changed or lacks provenance")
    if features is not None and previous.get("features_sha256") != feature_identity(
        features
    ):
        raise incompatible_training_cache(
            "feature values, columns or row order changed"
        )
    if model is not None and previous.get("state_sha256") != state_identity(
        model.state_dict()
    ):
        raise incompatible_training_cache("model weights changed")
    return previous


def validate_cached_training_settings(args):
    """Check before opening logs or generating features in a resumed full run."""
    root = Path(args.output)
    if any(
        (root / name).exists()
        for name in (
            "siamese_model.pt",
            "embeddings.csv",
            "bins.csv",
            "pre_rescue.csv",
            "rescue_assignments.csv",
            "clustering_provenance.json",
            "knn_graph_edges.csv",
        )
    ) or any(root.glob("bins/bin_*.fa")):
        validate_training_provenance(args)


def save_training_provenance(model, features, args, loader=None):
    config = model_configuration(
        model.n_kmer_features, model.n_coverage_features, model.representation_dim
    )
    # Retain the provenance of a deliberately reused filtered FASTA as well as
    # the requested policy. Never relabel a historical filtered result.
    try:
        params = json.loads((Path(args.output) / "params.json").read_text())
        filter_provenance = params.get("hyenadna_filter")
    except (OSError, ValueError):
        filter_provenance = None
    value = {
        "schema": 1,
        "model": config,
        "settings": training_settings(args),
        "inputs": input_identity(args),
        "features_sha256": feature_identity(features),
        "state_sha256": state_identity(model.state_dict()),
        "checkpoint_sha256": (
            file_sha256(Path(args.output) / "siamese_model.pt")
            if (Path(args.output) / "siamese_model.pt").exists()
            else None
        ),
        "filter_provenance": filter_provenance,
        "loader": (
            {
                "batch_size": loader.batch_size,
                "drop_last": loader.drop_last,
                "num_workers": loader.num_workers,
                "persistent_workers": loader.persistent_workers,
                "positive_pairs": len(loader.dataset),
            }
            if loader is not None
            else None
        ),
        "execution": {
            "torch": torch.__version__,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "cpu_threads": torch.get_num_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }
    write_training_provenance(args, value)
    return value


def write_training_provenance(args, value):
    path = Path(args.output) / PROVENANCE_FILE
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_cached_embeddings(features, args, model=None):
    """Return validated embeddings, or None when there is no embedding cache."""
    import pandas as pd

    path = Path(args.output) / "embeddings.csv"
    if not path.exists():
        return None
    previous = validate_training_provenance(args, features, model)
    if previous.get("embeddings_sha256") != file_sha256(path):
        raise incompatible_training_cache("embedding file changed or lacks provenance")
    cached = pd.read_csv(
        path, dtype={0: str}, keep_default_na=False, float_precision="round_trip"
    )
    return cached.set_index(cached.columns[0]).rename_axis(None)
