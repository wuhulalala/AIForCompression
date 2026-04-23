from __future__ import annotations

import math

import numpy as np


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-12:
        return float("inf"), mse
    data_range = float(orig64.max() - orig64.min())
    if data_range < 1e-8:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def base_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bitstream_bytes: int,
    elapsed: tuple[float, float],
    group_count: int = 1,
) -> dict[str, float | int | None]:
    psnr, mse = calculate_psnr(original, reconstructed)
    original_bytes = int(original.size * original.dtype.itemsize)
    encode_time, decode_time = elapsed
    groups = max(int(group_count), 1)
    encode_throughput = original_bytes / encode_time if encode_time > 0 else None
    decode_throughput = original_bytes / decode_time if decode_time > 0 else None
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": psnr,
        "bpp": bitstream_bytes * 8.0 / (original.shape[-2] * original.shape[-1]),
        "bitstream_bytes": int(bitstream_bytes),
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / bitstream_bytes if bitstream_bytes > 0 else float("inf"),
        "group_count": groups,
        "encode_time_total": encode_time,
        "decode_time_total": decode_time,
        "encode_time_per_group_avg": encode_time / groups,
        "decode_time_per_group_avg": decode_time / groups,
        "encode_throughput_MBps": encode_throughput / 1e6 if encode_throughput is not None else None,
        "decode_throughput_MBps": decode_throughput / 1e6 if decode_throughput is not None else None,
        "sample_wall_time_total": None,
        "sample_wall_throughput_MBps": None,
        # Legacy names kept for existing plot/aggregation scripts. These are totals for grouped samples.
        "encode_time_avg": encode_time,
        "decode_time_avg": decode_time,
        "encode_throughput": encode_throughput,
        "decode_throughput": decode_throughput,
    }
