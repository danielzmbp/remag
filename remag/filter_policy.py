"""Identity and fixed execution settings of the validated filtering policy."""

MODEL_FILENAME = "pytorch_model.bin.20260724.4k"
MODEL_SHA256 = "39d05fcecfdb0849fcf1822ffb0c816ee119e0436a19e0284e33e42f280f3689"
WINDOW_SIZE = 4096
WINDOW_BUDGET = 16
RECORD_BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 256
EUKARYOTE_FILTER_THRESHOLD = 0.30
HYENADNA_PREDICTION_POLICY = (
    "single_4k_uniform16_v2:"
    f"sha256={MODEL_SHA256}:context=4096:span=4096:"
    "spacing=2048:round=python:mean=python:cutoff=0.30"
)
HYENADNA_FILTER_CONFIG = {
    "policy": HYENADNA_PREDICTION_POLICY,
    "model": MODEL_FILENAME,
    "model_sha256": MODEL_SHA256,
    "context_tokens": WINDOW_SIZE,
    "biological_tokens": WINDOW_SIZE - 2,
    "window_span": WINDOW_SIZE,
    "window_budget": WINDOW_BUDGET,
    "positions": "uniform_endpoints_half_context_python_round",
    "aggregation": "python_sum_of_native_probabilities",
    "threshold": EUKARYOTE_FILTER_THRESHOLD,
    "record_batch_size": RECORD_BATCH_SIZE,
    "window_batch_size": INFERENCE_BATCH_SIZE,
    "probability_backend": "selected_device",
}


class ClassifierPolicyError(RuntimeError):
    """The verified model or numerical policy cannot be safely applied."""
