#!/usr/bin/env python3
"""Minimal DCMVC P-frame test using test_video.py core logic, with fixed imports."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path("/data/run01/scxj523/zsh/project/AIForCompression")
CHECKPOINTS = PROJECT / "checkpoints"
DCMVC = PROJECT / "models" / "DCMVC"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--i_frame_model_path", required=True)
    p.add_argument("--p_frame_model_path", required=True)
    p.add_argument("--test_config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_frames", type=int, default=30)
    return p.parse_args()

def load_test_config(path):
    with open(path) as f:
        cfg = json.load(f)
    root = cfg["root_path"]
    for ds_name, ds_cfg in cfg["test_classes"].items():
        if ds_cfg.get("test", 0) == 0:
            continue
        base = ds_cfg["base_path"]
        for seq_name, seq_cfg in ds_cfg["sequences"].items():
            return {
                "src_path": str(Path(root) / base / seq_name),
                "src_type": ds_cfg["src_type"],
                "width": seq_cfg["width"],
                "height": seq_cfg["height"],
                "frames": min(seq_cfg["frames"], 30),
                "gop": seq_cfg.get("gop", 32),
            }
    raise ValueError("No valid sequence in test config")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Setup DCMVC package paths
    sys.path.insert(0, str(DCMVC))
    sys.path.insert(0, str(DCMVC / "src"))

    # Load DCMVC modules with proper package hierarchy for relative imports
    import importlib.util
    ROOT = "dcvc_pkg"

    # Create root package
    root_mod = type(sys)(ROOT)
    root_mod.__path__ = [str(DCMVC)]
    root_mod.__package__ = ROOT
    sys.modules[ROOT] = root_mod

    def load_mod(name, filepath):
        full = f"{ROOT}.{name}"
        spec = importlib.util.spec_from_file_location(full, filepath)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = full.rsplit(".", 1)[0]
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    sys.path.insert(0, str(DCMVC))  # for .so loading

    load_mod("transforms.functional", str(DCMVC / "transforms/functional.py"))
    load_mod("utils.stream_helper", str(DCMVC / "utils/stream_helper.py"))
    load_mod("utils.common", str(DCMVC / "utils/common.py"))
    load_mod("utils.video_reader", str(DCMVC / "utils/video_reader.py"))
    load_mod("utils.video_writer", str(DCMVC / "utils/video_writer.py"))
    load_mod("utils.metrics", str(DCMVC / "utils/metrics.py"))
    load_mod("models.entropy_models", str(DCMVC / "models/entropy_models.py"))
    load_mod("models.layers", str(DCMVC / "models/layers.py"))
    load_mod("models.video_net", str(DCMVC / "models/video_net.py"))
    load_mod("models.common_model", str(DCMVC / "models/common_model.py"))
    load_mod("models.image_model", str(DCMVC / "models/image_model.py"))
    load_mod("models.DCMVC_model", str(DCMVC / "models/DCMVC_model.py"))

    IntraNoAR = sys.modules[f"{ROOT}.models.image_model"].IntraNoAR
    DMC = sys.modules[f"{ROOT}.models.DCMVC_model"].DMC
    get_state_dict = sys.modules[f"{ROOT}.utils.stream_helper"].get_state_dict
    get_padding_size = sys.modules[f"{ROOT}.utils.stream_helper"].get_padding_size

    # Load test config
    cfg = load_test_config(args.test_config)
    print(f"Config: {cfg['src_path']}, {cfg['width']}x{cfg['height']}, {cfg['frames']} frames", flush=True)

    # Read PNG frames
    png_dir = Path(cfg["src_path"])
    pngs = sorted(png_dir.glob("*.png"))[:cfg["frames"]]
    if not pngs:
        raise FileNotFoundError(f"No PNGs in {png_dir}")
    print(f"Frames: {len(pngs)}", flush=True)

    def load_png(path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    first = load_png(pngs[0])
    _, _, h, w = first.shape
    print(f"Resolution: {w}x{h}", flush=True)

    # Load I-frame model (CPU first for load_state_dict, then GPU)
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_net = IntraNoAR(ec_thread=False, stream_part=1, inplace=True)
    i_net.load_state_dict(i_state_dict)
    i_net = i_net.to(device).eval()
    i_net.update(force=True)

    # Load P-frame model
    p_state_dict = get_state_dict(args.p_frame_model_path)
    p_net = DMC(ec_thread=False, stream_part=1, inplace=True)
    p_net.load_state_dict(p_state_dict, strict=False)
    p_net = p_net.to(device).eval()
    p_net.update(force=True)
    print("Models loaded", flush=True)

    # Run P-frame compression for each rate
    results = []
    for q_idx in [0, 1, 2, 3]:
        print(f"  q_idx={q_idx}...", flush=True)
        bits = []; psnrs = []; enc_times = []; dec_times = []
        dpb = {"ref_frame": None, "ref_feature": None, "ref_mv_feature": None,
               "ref_y": None, "ref_mv_y": None}
        gop = cfg["gop"]

        with torch.no_grad():
            for fi, png in enumerate(pngs):
                x = load_png(png).to(device)
                pl, pr, pt, pb = get_padding_size(h, w, 16)
                x_padded = F.pad(x, (pl, pr, pt, pb), mode="replicate")

                torch.cuda.synchronize()
                t0 = time.time()

                if fi % gop == 0:
                    result = i_net.encode_decode(x_padded, q_in_ckpt=True, q_index=q_idx,
                                                  output_path=None, pic_height=h, pic_width=w)
                    dpb = {"ref_frame": result["x_hat"], "ref_feature": None,
                           "ref_mv_feature": None, "ref_y": None, "ref_mv_y": None}
                    recon = result["x_hat"]
                    bits.append(result["bit"])
                    frame_time = time.time() - t0
                    enc_times.append(frame_time)
                    dec_times.append(frame_time)  # I-frame: encode+decode in one call
                else:
                    result = p_net.encode_decode(x_padded, dpb, q_in_ckpt=True, q_index=q_idx,
                                                  output_path=None, pic_height=h, pic_width=w,
                                                  frame_idx=fi % 4)
                    dpb = result["dpb"]
                    recon = dpb["ref_frame"]
                    bits.append(result["bit"])
                    enc_times.append(result["encoding_time"])
                    dec_times.append(result["decoding_time"])

                x_hat = recon.clamp_(0, 1)
                x_hat = F.pad(x_hat, (-pl, -pr, -pt, -pb))[:, :, :h, :w]
                mse = float(torch.mean((x[:, :, :h, :w] - x_hat) ** 2))
                psnr = float(10 * np.log10(1.0 / mse)) if mse > 1e-30 else float("inf")
                psnrs.append(psnr)

        total_bits = sum(bits)
        bpp = total_bits / (h * w * len(pngs))
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_enc = sum(enc_times) / len(enc_times) if enc_times else 0
        avg_dec = sum(dec_times) / len(dec_times) if dec_times else 0
        orig_bytes = h * w * 3 * len(pngs)

        results.append({
            "model_name": "DCMVC", "model_id": f"DCMVC_Pframe_q{q_idx}", "metric": "mse",
            "bpp": bpp, "psnr": avg_psnr,
            "bitstream_bytes": int(total_bits // 8), "original_bytes": orig_bytes,
            "compression_ratio": orig_bytes / (total_bits // 8) if total_bits > 0 else float("inf"),
            "encode_time_avg": avg_enc, "decode_time_avg": avg_dec,
            "groups": 1, "samples": len(pngs), "shape": [3, h, w],
        })
        print(f"    bpp={bpp:.4f} psnr={avg_psnr:.1f}dB cr={results[-1]['compression_ratio']:.0f} enc={avg_enc*1000:.1f}ms dec={avg_dec*1000:.1f}ms", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf = out / "summary_dcmvc_pframe.json"
    json.dump(results, open(sf, "w"), indent=2)
    print(f"Results: {sf}", flush=True)
    for r in results:
        print(json.dumps(r, indent=2), flush=True)

if __name__ == "__main__":
    main()
