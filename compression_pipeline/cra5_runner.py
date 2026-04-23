from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.era5_constants import ERA5_CHANNELS
from compression_pipeline.metrics import base_metrics
from compression_pipeline.torch_codecs import nested_bytes, synchronize_if_needed


@torch.no_grad()
def run_cra5_sample(sample: CanonicalSample, model: Any, device: str) -> dict:
    sample.require_layout("channel_height_width")
    if sample.kind != "scientific_field" or sample.array.shape[0] != ERA5_CHANNELS:
        raise ValueError(f"CRA5 requires an ERA5 scientific_field sample with {ERA5_CHANNELS} channels, got {sample.kind} {sample.array.shape}")

    x = torch.from_numpy(sample.array.astype(np.float32, copy=False)[None]).to(device)
    t0 = time.time()
    compressed = model.compress(x)
    synchronize_if_needed(x)
    t1 = time.time()
    decompressed = model.decompress(compressed["strings"], compressed["z_shape"])
    synchronize_if_needed(x)
    t2 = time.time()

    reconstruction = decompressed["x_hat"].detach().cpu().numpy()[0].astype(np.float32, copy=False)
    bitstream_bytes = nested_bytes(compressed["strings"])
    metrics = base_metrics(sample.array, reconstruction, bitstream_bytes, (t1 - t0, t2 - t1))
    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "model_view": "cra5_268",
        "shape": list(sample.array.shape),
    })
    return metrics

