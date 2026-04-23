import argparse
import faulthandler
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr


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
GROUP_SIZE = 3


def find_nc_pairs(data_root):
    pairs = []
    for root, _, files in os.walk(data_root):
        for name in sorted(files):
            if name.endswith("_pressure.nc"):
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


def split_channel_groups(x):
    groups = []
    _, c, h, w = x.shape
    for idx in range(0, c, GROUP_SIZE):
        chunk = x[:, idx:idx + GROUP_SIZE]
        actual_c = chunk.shape[1]
        if actual_c < GROUP_SIZE:
            pad = chunk[:, -1:].expand(1, GROUP_SIZE - actual_c, h, w)
            chunk = torch.cat([chunk, pad], dim=1)
        groups.append((chunk, actual_c))
    return groups


def minmax_normalize(chunk):
    _, c, _, _ = chunk.shape
    cmin = chunk.reshape(1, c, -1).min(dim=-1, keepdim=True).values.unsqueeze(-1)
    cmax = chunk.reshape(1, c, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    scale = (cmax - cmin).clamp(min=1e-8)
    return (chunk - cmin) / scale, cmin, scale


def pad_replicate(x, divisor):
    _, _, h, w = x.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


def bitstream_size(bit_stream):
    if isinstance(bit_stream, (bytes, bytearray)):
        return len(bit_stream)
    if isinstance(bit_stream, (list, tuple)):
        return sum(bitstream_size(x) for x in bit_stream)
    return len(bit_stream)


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


def load_dcmvc(model_path, device):
    root = "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC"
    clear_src_modules()
    sys.path.insert(0, root)
    from src.models.image_model import IntraNoAR
    from src.utils.stream_helper import get_state_dict

    model = IntraNoAR(inplace=True)
    model.load_state_dict(get_state_dict(model_path))
    model = model.to(device).eval()
    model.update(force=True)
    return model


def load_dcvc(model_path, device):
    root = "/data/run01/scxj523/zsh/project/AIForCompression/models/DCVC"
    cpp_root = os.path.join(root, "src", "cpp")
    clear_src_modules()
    sys.path.insert(0, root)
    sys.path.insert(0, cpp_root)
    from src.models.image_model import DMCI
    from src.utils.common import get_state_dict

    model = DMCI()
    model.load_state_dict(get_state_dict(model_path))
    model = model.to(device).eval()
    model.update(None)
    model.half()
    return model


def clear_src_modules():
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [
        path for path in sys.path
        if not path.endswith("/models/DCMVC")
        and not path.endswith("/models/DCVC")
        and not path.endswith("/models/DCVC/src/cpp")
    ]


@torch.no_grad()
def test_dcmvc(model, x, q_index):
    x_pad, orig_h, orig_w = pad_replicate(x, 64)
    enc_start = time.time()
    compressed = model.compress(x_pad, True, q_index)
    if x.is_cuda:
        torch.cuda.synchronize()
    enc_end = time.time()
    decompressed = model.decompress(compressed["bit_stream"], x_pad.shape[-2], x_pad.shape[-1], True, q_index)
    if x.is_cuda:
        torch.cuda.synchronize()
    dec_end = time.time()
    return decompressed["x_hat"][:, :, :orig_h, :orig_w], bitstream_size(compressed["bit_stream"]), enc_end - enc_start, dec_end - enc_end


@torch.no_grad()
def test_dcvc(model, x, qp):
    x_pad, orig_h, orig_w = pad_replicate(x.half(), 64)
    enc_start = time.time()
    compressed = model.compress(x_pad, qp)
    if x.is_cuda:
        torch.cuda.synchronize()
    enc_end = time.time()
    sps = {"height": x_pad.shape[-2], "width": x_pad.shape[-1], "ec_part": 0}
    decompressed = model.decompress(compressed["bit_stream"], sps, qp)
    if x.is_cuda:
        torch.cuda.synchronize()
    dec_end = time.time()
    return decompressed["x_hat"].float()[:, :, :orig_h, :orig_w], bitstream_size(compressed["bit_stream"]), enc_end - enc_start, dec_end - enc_end


def parse_args():
    parser = argparse.ArgumentParser(description="Test video intra/image models on full 268-channel ERA5 data.")
    parser.add_argument("--data_root", default="/data/run01/scxj523/zsh/project/Data/ERA5/2024")
    parser.add_argument("--output_dir", default="/data/run01/scxj523/zsh/project/AIForCompression/unified_results/video_intra_era5")
    parser.add_argument("--model", choices=["DCMVC", "DCVC", "both"], default="both")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--dcmvc_checkpoint", default="/data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcmvc/cvpr2023_image_psnr.pth.tar")
    parser.add_argument("--dcvc_checkpoint", default="/data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcvc-rt/cvpr2025_image.pth.tar")
    return parser.parse_args()


def main():
    faulthandler.enable()
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_nc_pairs(args.data_root)
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    if not pairs:
        raise SystemExit(f"No ERA5 pairs found in {args.data_root}")

    model_groups = []
    if args.model in ("DCMVC", "both"):
        model_groups.append((
            "DCMVC",
            lambda: load_dcmvc(args.dcmvc_checkpoint, device),
            [(f"DCMVC_Intra_q{i}", lambda m, x, i=i: test_dcmvc(m, x, i)) for i in range(4)],
        ))
    if args.model in ("DCVC", "both"):
        model_groups.append((
            "DCVC",
            lambda: load_dcvc(args.dcvc_checkpoint, device),
            [(f"DCVC_RT_Intra_q{q}", lambda m, x, q=q: test_dcvc(m, x, q)) for q in (0, 21, 42, 63)],
        ))

    summary = []
    for arch, loader, variants in model_groups:
        model = None
        try:
            model = loader()
            params = sum(p.numel() for p in model.parameters())
            for model_id, runner in variants:
                print(f"[progress] start {model_id}", flush=True)
                try:
                    for pressure, single, ts in pairs:
                        raw = read_nc(pressure, single)
                        x = torch.from_numpy(raw[None]).to(device)
                        recon_parts = []
                        total_bytes = 0
                        total_enc = 0.0
                        total_dec = 0.0
                        for group_idx, (chunk, actual_c) in enumerate(split_channel_groups(x)):
                            print(f"[progress] {model_id} {ts} group {group_idx}", flush=True)
                            normed, cmin, scale = minmax_normalize(chunk)
                            x_hat_norm, stream_bytes, enc_time, dec_time = runner(model, normed)
                            x_hat = x_hat_norm * scale + cmin
                            recon_parts.append(x_hat[:, :actual_c])
                            total_bytes += stream_bytes
                            total_enc += enc_time
                            total_dec += dec_time
                        recon = torch.cat(recon_parts, dim=1).squeeze(0).cpu().numpy()
                        psnr, mse = calculate_psnr(raw, recon)
                        original_bytes = raw.size * 4
                        summary.append({
                            "model_id": model_id,
                            "arch": arch,
                            "metric": "mse",
                            "params": params,
                            "timestamp": ts,
                            "mse": mse,
                            "rmse": math.sqrt(mse),
                            "psnr": psnr,
                            "bpp": total_bytes * 8.0 / (raw.shape[-2] * raw.shape[-1]),
                            "bitstream_bytes": total_bytes,
                            "original_bytes": original_bytes,
                            "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float("inf"),
                            "encode_time_avg": total_enc,
                            "decode_time_avg": total_dec,
                            "encode_throughput": original_bytes / total_enc if total_enc > 0 else None,
                            "decode_throughput": original_bytes / total_dec if total_dec > 0 else None,
                        })
                except Exception as exc:
                    summary.append({"model_id": model_id, "arch": arch, "metric": "mse", "error": str(exc)})
                with (output_dir / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=2)
                print(json.dumps(summary[-1], indent=2))
        except Exception as exc:
            for model_id, _ in variants:
                summary.append({"model_id": model_id, "arch": arch, "metric": "mse", "error": str(exc)})
            with (output_dir / "summary.json").open("w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary[-1], indent=2))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
