"""
Test CRA5 on Kodak/Tomo with channel stacking to minimize replication waste.

Kodak: stack all 24 images (3ch each → 72ch), replicate ~3.7x to fill 268
Tomo:  stack 268 consecutive projections, NO replication needed

ERA5 native path is unaffected — handled by run_cra5_sample in cra5_runner.py.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models" / "CRA5"))
from cra5.models.compressai.zoo import vaeformer_pretrained

CRA5_IMG_SIZE = (721, 1440)
CRA5_CHANS = 268
KODAK_DIR = Path("/data/run01/scxj523/zsh/project/Data/Kodac")
TOMO_PATH = Path("/data/run01/scxj523/zsh/project/Data/tomo/tomo_00083.h5")


def load_model(device):
    net = vaeformer_pretrained(quality=268, metric="mse", pretrained=True).eval().to(device)
    return net


def psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0: return float("inf")
    peak = orig.max() - orig.min()
    if peak == 0: peak = 1.0
    return 10 * np.log10(peak ** 2 / mse)


def stack_and_pad(arrays, target_ch, target_size):
    """Stack list of [C, H, W] arrays along channel dim, replicate to target_ch,
    then bilinear resize each to target_size."""
    # Each array may have different H,W — pad to max first
    max_h = max(a.shape[1] for a in arrays)
    max_w = max(a.shape[2] for a in arrays)

    padded = []
    vmins, vmaxs = [], []
    for a in arrays:
        c, h, w = a.shape
        vmin, vmax = a.min(), a.max()
        if vmax - vmin > 1e-8:
            a_norm = (a - vmin) / (vmax - vmin)
        else:
            a_norm = a
        vmins.append(vmin)
        vmaxs.append(vmax)
        # Pad to max_h, max_w
        ph = max_h - h
        pw = max_w - w
        if ph > 0 or pw > 0:
            a_norm = np.pad(a_norm, ((0, 0), (0, ph), (0, pw)), mode='edge')
        padded.append(a_norm)

    stacked = np.concatenate(padded, axis=0)  # [total_C, max_H, max_W]
    total_c = stacked.shape[0]

    # Minmax normalize entire stack to [0,1]
    s_min, s_max = stacked.min(), stacked.max()
    if s_max - s_min > 1e-8:
        stacked = (stacked - s_min) / (s_max - s_min)

    # Resize to target_size
    t = torch.from_numpy(stacked).unsqueeze(0)  # [1, total_C, max_H, max_W]
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)

    # Replicate channels to target_ch
    reps = target_ch // total_c
    rem = target_ch % total_c
    out = t.repeat(1, reps, 1, 1)
    if rem > 0:
        out = torch.cat([out, t[:, :rem, :, :]], dim=1)

    # Store per-channel info for reconstruction
    meta = {
        "total_c": total_c,
        "orig_shapes": [(a.shape[1], a.shape[2]) for a in arrays],
        "vmins": vmins,
        "vmaxs": vmaxs,
        "stack_min": s_min,
        "stack_max": s_max,
        "padded_h": max_h,
        "padded_w": max_w,
    }
    return out, meta


def unstack_and_reconstruct(x_hat, meta):
    """Reverse stack_and_pad: extract channels, un-normalize per original image."""
    x_hat = x_hat.squeeze(0)  # [268, 721, 1440]

    # Crop to total_c channels
    x_cropped = x_hat[:meta["total_c"]]

    # Resize back to padded size
    x_padded = F.interpolate(
        x_cropped.unsqueeze(0),
        size=(meta["padded_h"], meta["padded_w"]),
        mode="bilinear", align_corners=False
    ).squeeze(0)  # [total_C, padded_H, padded_W]

    # Denormalize from stack min/max
    smin, smax = meta["stack_min"], meta["stack_max"]
    if smax - smin > 1e-8:
        x_padded = x_padded * (smax - smin) + smin
    x_padded = torch.clamp(x_padded, 0.0, 1.0)

    # Split back into per-image channels and denormalize
    results = []
    offset = 0
    for c, (orig_h, orig_w), vmin, vmax in zip(
        [a.shape[0] for a in [np.empty((1, h, w)) for h, w in meta["orig_shapes"]]],
        meta["orig_shapes"], meta["vmins"], meta["vmaxs"]
    ):
        # Get slice, crop to original size
        slice_ch = x_padded[offset:offset + c, :orig_h, :orig_w]
        offset += c
        # Denormalize per-image
        if vmax - vmin > 1e-8:
            img = slice_ch * (vmax - vmin) + vmin
        else:
            img = slice_ch
        img = img.cpu().numpy().astype(np.float32)
        # Clip to original range
        img = np.clip(img, vmin, vmax)
        results.append(img)
    return results


def test_kodak(net, output_dir, device):
    from PIL import Image
    print("\n=== CRA5 Kodak (stacked 24) ===")

    # Load all Kodak images
    images = []
    sample_ids = []
    for p in sorted(KODAK_DIR.glob("*.png")):
        img = np.array(Image.open(p)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose(2, 0, 1)  # [C, H, W]
        images.append(img)
        sample_ids.append(p.stem)

    print(f"Loaded {len(images)} images, shapes: {[img.shape for img in images[:3]]}...")

    # Stack and pack
    cra5_input, meta = stack_and_pad(images, CRA5_CHANS, CRA5_IMG_SIZE)
    cra5_input = cra5_input.to(device)
    print(f"Stacked → CRA5 input: {cra5_input.shape}")

    # Compress / decompress
    with torch.no_grad():
        t0 = time.time()
        comp = net.compress(cra5_input)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.time()
        dec = net.decompress(comp["strings"], comp["z_shape"])
        torch.cuda.synchronize() if device == "cuda" else None
        t2 = time.time()

    bitstream_bytes = sum(len(s[0]) for s in comp["strings"])
    total_orig_bytes = sum(img.nbytes for img in images)
    total_pixels = sum(img.shape[1] * img.shape[2] * img.shape[0] for img in images)
    bpp = bitstream_bytes * 8 / total_pixels
    compression_ratio = total_orig_bytes / max(bitstream_bytes, 1)

    # Reconstruct per-image
    reconstructions = unstack_and_reconstruct(dec["x_hat"], meta)
    print(f"Decompressed, unstacked {len(reconstructions)} images")

    # Per-image metrics
    results = []
    for i, (orig, recon, sid) in enumerate(zip(images, reconstructions, sample_ids)):
        psnr_val = psnr(orig, recon)
        mse_val = np.mean((orig - recon) ** 2)
        orig_bytes = orig.nbytes
        # Allocate bitstream proportionally by pixel count
        n_pixels = orig.shape[1] * orig.shape[2] * orig.shape[0]
        prop_bpp = bpp  # use global bpp
        prop_cr = compression_ratio

        results.append({
            "dataset_id": "kodak",
            "sample_id": sid,
            "sample_kind": "natural_image",
            "shape": list(orig.shape),
            "model_name": "CRA5",
            "model_id": "cra5_268v_stacked",
            "metric": "mse",
            "mse": float(mse_val),
            "rmse": float(np.sqrt(mse_val)),
            "psnr": float(psnr_val),
            "bpp": float(prop_bpp),
            "bitstream_bytes": bitstream_bytes,
            "original_bytes": int(orig_bytes),
            "compression_ratio": float(prop_cr),
            "encode_time_total": float(t1 - t0),
            "decode_time_total": float(t2 - t1),
            "encode_time_avg": float((t1 - t0) / len(images)),
            "decode_time_avg": float((t2 - t1) / len(images)),
            "encode_throughput_MBps": float(total_orig_bytes / max(t1 - t0, 1e-8) / 1e6),
            "decode_throughput_MBps": float(total_orig_bytes / max(t2 - t1, 1e-8) / 1e6),
            "params": sum(p.numel() for p in net.parameters()),
        })
        print(f"  {sid}: PSNR={psnr_val:.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    avg_psnr = np.mean([r["psnr"] for r in results])
    print(f"Saved {out_path}")
    print(f"Kodak avg: PSNR={avg_psnr:.1f} dB, BPP={bpp:.4f}, CR={compression_ratio:.1f}x, "
          f"total_bitstream={bitstream_bytes/1024:.1f}KB")


def test_tomo(net, output_dir, device, group_size=268):
    import h5py
    from PIL import Image
    print(f"\n=== CRA5 Tomo (stacked {group_size}) ===")

    with h5py.File(TOMO_PATH, "r") as f:
        projections = f["exchange/data"][:]  # [N, H, W]
        dtype_orig = projections.dtype

    n_total = len(projections)
    max_groups = n_total // group_size
    print(f"Total projections: {n_total}, groups of {group_size}: {max_groups}")

    # Pre-resize ALL projections to 721x1440 on CPU (save 20x memory)
    print(f"  Pre-resizing {n_total} projections to {CRA5_IMG_SIZE} on CPU...")
    projections_small = np.zeros((n_total, CRA5_IMG_SIZE[0], CRA5_IMG_SIZE[1]),
                                  dtype=np.float32)
    for i in range(n_total):
        img = Image.fromarray(projections[i])
        img_small = img.resize((CRA5_IMG_SIZE[1], CRA5_IMG_SIZE[0]), Image.BILINEAR)
        projections_small[i] = np.array(img_small, dtype=np.float32)
    print(f"  Done. Memory: {projections_small.nbytes/1e9:.1f} GB")

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for g in range(max_groups):
        start = g * group_size
        end = start + group_size
        print(f"\n  Group {g}: frames {start}:{end}...", end=" ", flush=True)

        # Already resized, just stack
        frames = projections_small[start:end]  # [268, 721, 1440]
        orig_h, orig_w = CRA5_IMG_SIZE

        # Minmax normalize
        vmin, vmax = frames.min(), frames.max()
        if vmax - vmin > 1e-8:
            frames_norm = (frames - vmin) / (vmax - vmin)
        else:
            frames_norm = frames

        x = torch.from_numpy(frames_norm).unsqueeze(0).to(device)  # [1, 268, 721, 1440]

        # Compress / decompress
        with torch.no_grad():
            t0 = time.time()
            comp = net.compress(x)
            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.time()
            dec = net.decompress(comp["strings"], comp["z_shape"])
            torch.cuda.synchronize() if device == "cuda" else None
            t2 = time.time()

        bitstream_bytes = sum(len(s[0]) for s in comp["strings"])

        # Reconstruct: denormalize
        x_hat = dec["x_hat"].squeeze(0)  # [268, 721, 1440]
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
        recon = x_hat.cpu().numpy().astype(np.float32)
        if vmax - vmin > 1e-8:
            recon = recon * (vmax - vmin) + vmin
        recon = np.clip(recon, vmin, vmax)

        # Scale back to original uint16 range for PSNR
        # (metrics are computed against the pre-resized data for fairness)
        for i in range(group_size):
            # Compute PSNR on resized data (the actual content that was compressed)
            orig_frame = frames[i]
            recon_frame = recon[i]
            psnr_val = psnr(orig_frame, recon_frame)
            mse_val = np.mean((orig_frame - recon_frame) ** 2)
            # Use uint16 pixel count for BPP calculation
            # original uint16: 2 bytes per pixel, actual pixels at original resolution
            orig_pixel_count = 2048 * 2448
            orig_bytes_per_frame = orig_pixel_count * 2  # uint16
            prop_bytes = bitstream_bytes / group_size
            bpp = prop_bytes * 8 / orig_pixel_count
            cr = orig_bytes_per_frame / max(prop_bytes, 1)

            results.append({
                "dataset_id": "tomo",
                "sample_id": f"proj_{start + i:04d}",
                "sample_kind": "scientific_field",
                "shape": [1, 2048, 2448],
                "model_name": "CRA5",
                "model_id": f"cra5_268v_stacked{group_size}",
                "metric": "mse",
                "mse": float(mse_val),
                "rmse": float(np.sqrt(mse_val)),
                "psnr": float(psnr_val),
                "bpp": float(bpp),
                "bitstream_bytes": int(prop_bytes),
                "original_bytes": int(orig_bytes_per_frame),
                "compression_ratio": float(cr),
                "encode_time_total": float((t1 - t0) / group_size),
                "decode_time_total": float((t2 - t1) / group_size),
                "encode_time_avg": float((t1 - t0) / group_size),
                "decode_time_avg": float((t2 - t1) / group_size),
                "encode_throughput_MBps": float(orig_bytes_per_frame / max((t1 - t0) / group_size, 1e-8) / 1e6),
                "decode_throughput_MBps": float(orig_bytes_per_frame / max((t2 - t1) / group_size, 1e-8) / 1e6),
                "params": sum(p.numel() for p in net.parameters()),
            })

        avg_psnr = np.mean([r["psnr"] for r in results[-group_size:]])
        print(f"avg_psnr={avg_psnr:.1f} BPP={results[-1]['bpp']:.4f} "
              f"bitstream={bitstream_bytes/1024:.1f}KB enc={t1-t0:.1f}s dec={t2-t1:.1f}s")

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    all_psnr = [r["psnr"] for r in results]
    all_bpp = [r["bpp"] for r in results]
    all_cr = [r["compression_ratio"] for r in results]
    print(f"\nSaved {out_path} ({len(results)} frames)")
    print(f"Tomo avg: PSNR={np.mean(all_psnr):.1f} dB, BPP={np.mean(all_bpp):.4f}, "
          f"CR={np.mean(all_cr):.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kodak", "tomo", "all"], default="all")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tomo_group_size", type=int, default=268)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    net = load_model(device)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"Model: {param_count:,} params ({param_count/1e6:.1f}M)")

    if args.dataset in ("kodak", "all"):
        out = args.output_dir or str(PROJECT_ROOT / "unified_results" / "cra5_kodak_stacked")
        test_kodak(net, Path(out), device)

    if args.dataset in ("tomo", "all"):
        out = args.output_dir or str(PROJECT_ROOT / "unified_results" / "cra5_tomo_stacked")
        test_tomo(net, Path(out), device, args.tomo_group_size)


if __name__ == "__main__":
    main()
