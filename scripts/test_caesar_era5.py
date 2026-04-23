import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader


VNAMES = dict(
    pressure=["z", "q", "u", "v", "t", "r", "w"],
    single=["v10", "u10", "v100", "u100", "t2m", "tcc", "sp", "tp", "msl"],
)
PRESSURE_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]
IN_CHANNELS = len(VNAMES["pressure"]) * len(PRESSURE_LEVELS) + len(VNAMES["single"])


def limit_channels(raw_cthw, max_channels):
    if max_channels is None or max_channels < 0:
        return raw_cthw
    if max_channels <= 0:
        raise ValueError(f"--max_channels must be positive, got {max_channels}")
    if max_channels > raw_cthw.shape[0]:
        raise ValueError(f"--max_channels {max_channels} exceeds available channels {raw_cthw.shape[0]}")
    return raw_cthw[:max_channels]


def center_crop_cthw(raw_cthw, resolution):
    if resolution is None:
        return raw_cthw
    target_h, target_w = resolution
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"--resolution values must be positive, got {resolution}")
    _, _, height, width = raw_cthw.shape
    if target_h > height or target_w > width:
        raise ValueError(f"--resolution {resolution} exceeds data size {(height, width)}")
    start_h = (height - target_h) // 2
    start_w = (width - target_w) // 2
    return raw_cthw[:, :, start_h:start_h + target_h, start_w:start_w + target_w]


def find_nc_pairs(data_root):
    pairs = []
    for root, _, files in os.walk(data_root):
        for name in sorted(files):
            if not name.endswith("_pressure.nc"):
                continue
            ts = name.replace("_pressure.nc", "")
            pressure = os.path.join(root, name)
            single = os.path.join(root, f"{ts}_single.nc")
            if os.path.exists(single):
                pairs.append((pressure, single, ts))
    return sorted(pairs, key=lambda x: x[2])


def read_nc(pressure_file, single_file):
    fields = []
    pressure_data = xr.open_dataset(pressure_file, engine="netcdf4")
    single_data = xr.open_dataset(single_file, engine="netcdf4")
    try:
        available_levels = list(pressure_data.pressure_level.data)
        level_mapping = [available_levels.index(level) for level in PRESSURE_LEVELS]
        for vname in VNAMES["pressure"]:
            data = pressure_data[vname].data
            for level_idx in level_mapping:
                fields.append(data[0][level_idx][None])
        for vname in VNAMES["single"]:
            data = single_data[vname].data
            if vname == "tp":
                data = data * 1000
            fields.append(data)
    finally:
        pressure_data.close()
        single_data.close()
    return np.concatenate(fields, axis=0).astype(np.float32)


def calculate_psnr(original, reconstructed):
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-12:
        return float("inf"), mse
    data_range = float(orig64.max() - orig64.min())
    if data_range < 1e-8:
        data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


def parse_args():
    parser = argparse.ArgumentParser(description="Test CAESAR-V/D on full 268-channel ERA5 data.")
    parser.add_argument("--data_root", default="/data/run01/scxj523/zsh/project/Data/ERA5/2024")
    parser.add_argument("--output_dir", default="/data/run01/scxj523/zsh/project/AIForCompression/models/CAESAR/results_era5")
    parser.add_argument("--model", choices=["caesar_v", "caesar_d", "both"], default="both")
    parser.add_argument("--caesar_root", default="/data/run01/scxj523/zsh/project/AIForCompression/models/CAESAR")
    parser.add_argument("--ckpt_dir", default="/data/run01/scxj523/zsh/project/AIForCompression/checkpoints/caesar")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eb", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    return parser.parse_args()


def run_one(model_name, raw_cthw, timestamps, args, device):
    n_frame = 8 if model_name == "caesar_v" else 16
    print(f"[{model_name}] preparing {n_frame} frames from {raw_cthw.shape[1]} timestamps", flush=True)
    if raw_cthw.shape[1] < n_frame:
        raise ValueError(
            f"{model_name} requires at least {n_frame} ERA5 time frames, "
            f"but only found {raw_cthw.shape[1]}: {timestamps}"
        )

    sys.path.insert(0, args.caesar_root)
    from CAESAR.compressor import CAESAR
    from dataset import ScientificDataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=f"_{model_name}_era5.npz", dir=output_dir, delete=False) as tmp:
        npz_path = tmp.name
    print(f"[{model_name}] writing temporary npz: {npz_path}", flush=True)
    np.savez(npz_path, data=raw_cthw[:, None, :n_frame])

    data_arg = {
        "data_path": npz_path,
        "name": f"ERA5-{raw_cthw.shape[0]}-{model_name}",
        "variable_idx": list(range(raw_cthw.shape[0])),
        "section_range": [0, 1],
        "frame_range": [0, n_frame],
        "n_frame": n_frame,
        "train": False,
        "test_size": (256, 256),
        "inst_norm": True,
        "norm_type": "mean_range",
    }

    dataset = ScientificDataset(data_arg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    checkpoint = Path(args.ckpt_dir) / f"{model_name}.pth"
    compressor = CAESAR(
        model_path=str(checkpoint),
        use_diffusion=(model_name == "caesar_d"),
        device=device,
        n_frame=n_frame,
    )
    total_params = 0
    if model_name == "caesar_v":
        total_params = sum(p.numel() for p in compressor.compressor_v.parameters())
    else:
        total_params = (
            sum(p.numel() for p in compressor.keyframe_model.parameters())
            + sum(p.numel() for p in compressor.diffusion_model.parameters())
        )

    t0 = time.time()
    print(f"[{model_name}] compress start", flush=True)
    compressed, compressed_size = compressor.compress(loader, eb=args.eb)
    t1 = time.time()
    print(f"[{model_name}] decompress start", flush=True)
    reconstructed = compressor.decompress(compressed)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.time()

    original = dataset.input_data().numpy()
    recon = dataset.recons_data(reconstructed).detach().cpu().numpy()
    psnr, mse = calculate_psnr(original, recon)
    rmse = math.sqrt(mse)
    original_bytes = original.size * 4
    compressed_bytes = float(compressed_size.item() if hasattr(compressed_size, "item") else compressed_size)

    Path(npz_path).unlink(missing_ok=True)
    return {
        "model_id": model_name,
        "arch": "CAESAR",
        "metric": "mse",
        "params": total_params,
        "timestamps": timestamps[:n_frame],
        "data_shape": list(original.shape),
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "bpp": compressed_bytes * 8.0 / (raw_cthw.shape[0] * original.shape[-2] * original.shape[-1] * original.shape[2]),
        "bitstream_bytes": compressed_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf"),
        "encode_time_avg": t1 - t0,
        "decode_time_avg": t2 - t1,
        "encode_throughput": original_bytes / (t1 - t0) if t1 > t0 else None,
        "decode_throughput": original_bytes / (t2 - t1) if t2 > t1 else None,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"

    pairs = find_nc_pairs(args.data_root)
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    if not pairs:
        raise SystemExit(f"No ERA5 pressure/single pairs found in {args.data_root}")
    print(f"Found {len(pairs)} ERA5 pressure/single pairs in {args.data_root}", flush=True)

    frames = []
    timestamps = []
    for idx, (pressure, single, ts) in enumerate(pairs, start=1):
        print(f"Reading ERA5 pair {idx}/{len(pairs)}: {ts}", flush=True)
        frames.append(read_nc(pressure, single))
        timestamps.append(ts)
    print("Stacking ERA5 frames", flush=True)
    raw_tchw = np.stack(frames, axis=0)
    raw_cthw = np.transpose(raw_tchw, (1, 0, 2, 3))
    raw_cthw = limit_channels(raw_cthw, args.max_channels)
    raw_cthw = center_crop_cthw(raw_cthw, args.resolution)
    print(f"Prepared raw_cthw shape: {raw_cthw.shape}", flush=True)

    models = ["caesar_v", "caesar_d"] if args.model == "both" else [args.model]
    summary = []
    for model_name in models:
        print(f"Running {model_name}", flush=True)
        try:
            result = run_one(model_name, raw_cthw, timestamps, args, device)
        except Exception as exc:
            result = {"model_id": model_name, "arch": "CAESAR", "metric": "mse", "error": str(exc)}
        summary.append(result)
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
