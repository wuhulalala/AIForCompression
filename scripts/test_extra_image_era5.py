import argparse
import gc
import importlib
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


def pad_center(x, divisor):
    _, _, h, w = x.shape
    new_h = (h + divisor - 1) // divisor * divisor
    new_w = (w + divisor - 1) // divisor * divisor
    left = (new_w - w) // 2
    right = new_w - w - left
    top = (new_h - h) // 2
    bottom = new_h - h - top
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0), (left, right, top, bottom)


def unpad_center(x, padding):
    left, right, top, bottom = padding
    return F.pad(x, (-left, -right, -top, -bottom), mode="constant", value=0)


def strings_size(strings):
    total = 0
    for item in strings:
        if isinstance(item, (list, tuple)):
            total += strings_size(item)
        else:
            total += len(item)
    return total


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


def get_scale_table(min_val, max_val, levels):
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))


def load_hpcm(checkpoint, variant, device):
    root = "/data/run01/scxj523/zsh/project/AIForCompression/models/LIC-HPCM"
    sys.path.insert(0, root)
    model_name = "HPCM_Base" if variant == "base" else "HPCM_Large"
    net_cls = importlib.import_module(f"src.models.{model_name}").HPCM
    model = net_cls()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.update(get_scale_table(0.12, 64, 60))
    return model.to(device).eval()


def load_rwkv(checkpoint, device):
    root = "/data/run01/scxj523/zsh/project/AIForCompression/models/RwkvCompress"
    sys.path.insert(0, root)
    from eval import load_checkpoint
    from models import LALIC

    model = load_checkpoint(LALIC, checkpoint)
    model.update()
    return model.to(device).eval()


@torch.no_grad()
def run_compressai_like(model, x, divisor):
    x_pad, padding = pad_center(x, divisor)
    t0 = time.time()
    out_enc = model.compress(x_pad)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    if x.is_cuda:
        torch.cuda.synchronize()
    t2 = time.time()
    x_hat = unpad_center(out_dec["x_hat"], padding).clamp(0, 1)
    return x_hat, strings_size(out_enc["strings"]), t1 - t0, t2 - t1


def parse_args():
    parser = argparse.ArgumentParser(description="Test LIC-HPCM/RwkvCompress on full 268-channel ERA5 data.")
    parser.add_argument("--data_root", default="/data/run01/scxj523/zsh/project/Data/ERA5/2024")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", choices=["LIC-HPCM-base", "LIC-HPCM-large", "RwkvCompress"], required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_samples", type=int, default=1)
    return parser.parse_args()


def main():
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

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pth*"))
    summary = []
    for ckpt in ckpts:
        model = None
        model_id = f"{args.model}_{ckpt.stem.replace('.pth', '')}"
        try:
            if args.model.startswith("LIC-HPCM"):
                variant = "base" if args.model.endswith("base") else "large"
                model = load_hpcm(str(ckpt), variant, device)
                divisor = 256
            else:
                model = load_rwkv(str(ckpt), device)
                divisor = 128
            params = sum(p.numel() for p in model.parameters())
            for pressure, single, ts in pairs:
                raw = read_nc(pressure, single)
                x = torch.from_numpy(raw[None]).to(device)
                recon_parts = []
                total_bytes = 0
                total_enc = 0.0
                total_dec = 0.0
                for chunk, actual_c in split_channel_groups(x):
                    normed, cmin, scale = minmax_normalize(chunk)
                    x_hat_norm, stream_bytes, enc_time, dec_time = run_compressai_like(model, normed, divisor)
                    recon_parts.append((x_hat_norm * scale + cmin)[:, :actual_c])
                    total_bytes += stream_bytes
                    total_enc += enc_time
                    total_dec += dec_time
                recon = torch.cat(recon_parts, dim=1).squeeze(0).cpu().numpy()
                psnr, mse = calculate_psnr(raw, recon)
                original_bytes = raw.size * 4
                summary.append({
                    "model_id": model_id,
                    "arch": args.model,
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
                    "encode_throughput": original_bytes / total_enc / 1e6 if total_enc > 0 else None,
                    "decode_throughput": original_bytes / total_dec / 1e6 if total_dec > 0 else None,
                })
        except Exception as exc:
            summary.append({"model_id": model_id, "arch": args.model, "metric": "mse", "error": str(exc)})
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary[-1], indent=2))


if __name__ == "__main__":
    main()
