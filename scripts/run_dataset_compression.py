import argparse
import gc
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from compression_pipeline.adapters.era5 import ERA5Adapter
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.caesar_runner import CAESAR_N_FRAMES, run_caesar_sequence
from compression_pipeline.cra5_runner import run_cra5_sample
from compression_pipeline.model_registry import image_model_jobs
from compression_pipeline.runner import run_image_grouped_sample
from compression_pipeline.torch_codecs import CompressAILikeCodec


def parse_args():
    parser = argparse.ArgumentParser(description="Run supported codecs on ERA5 or Kodak through the shared data adapter.")
    parser.add_argument("--dataset", choices=["era5", "kodak"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=None, help="Subset: CRA5 DCAE WeConvene LIC_TCM LIC-HPCM RwkvCompress caesar_v caesar_d")
    parser.add_argument("--max_model_jobs", type=int, default=-1, help="Limit number of checkpoint/model jobs after filtering; useful for smoke tests.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm to use the allocated GPU.")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--caesar_root", default=str(PROJECT_ROOT / "models" / "CAESAR"))
    parser.add_argument("--caesar_ckpt_dir", default=str(PROJECT_ROOT / "checkpoints" / "caesar"))
    parser.add_argument("--caesar_start_index", type=int, default=0)
    parser.add_argument("--caesar_eb", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def iter_dataset_samples(args):
    if args.dataset == "kodak":
        yield from KodakAdapter(args.data_root).iter_samples(max_samples=args.max_samples)
        return
    yield from ERA5Adapter(args.data_root).iter_samples(
        max_samples=args.max_samples,
        max_channels=args.max_channels,
        resolution=tuple(args.resolution) if args.resolution else None,
    )


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()} device={device}", flush=True)

    summary = []
    requested_models = set(args.models) if args.models else None
    caesar_models = requested_caesar_models(requested_models)
    non_caesar_models = None if requested_models is None else requested_models - {"CAESAR", "caesar_v", "caesar_d", "CAESAR-V", "CAESAR-D"}

    if caesar_models:
        if args.dataset != "era5":
            raise SystemExit("CAESAR requires --dataset era5 because it needs a continuous ERA5 time sequence")
        max_n_frame = max(CAESAR_N_FRAMES[name] for name in caesar_models)
        if args.max_samples > 0 and args.max_samples < args.caesar_start_index + max_n_frame:
            raise SystemExit(
                f"CAESAR requires at least {args.caesar_start_index + max_n_frame} contiguous ERA5 samples, "
                f"got --max_samples {args.max_samples}"
            )
        sequence, timestamps = ERA5Adapter(args.data_root).load_sequence(
            max_samples=args.max_samples,
            max_channels=args.max_channels,
            resolution=tuple(args.resolution) if args.resolution else None,
        )
        for model_name in caesar_models:
            try:
                print(f"[model] running {model_name} on continuous ERA5 sequence", flush=True)
                result = run_caesar_sequence(
                    sequence,
                    timestamps,
                    model_name=model_name,
                    caesar_root=args.caesar_root,
                    ckpt_dir=args.caesar_ckpt_dir,
                    output_dir=output_dir,
                    device=device,
                    batch_size=args.batch_size,
                    eb=args.caesar_eb,
                    start_index=args.caesar_start_index,
                )
                summary.append(result)
                print(json.dumps(result, indent=2), flush=True)
            except Exception as exc:
                summary.append({"model_name": "CAESAR", "model_id": model_name, "metric": "mse", "error": str(exc)})
                print(f"[error] {model_name}: {exc}", flush=True)
            finally:
                summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    jobs = list(image_model_jobs(args.project_root, non_caesar_models))
    if args.max_model_jobs > 0:
        jobs = jobs[:args.max_model_jobs]
    if non_caesar_models != set() and jobs:
        samples = list(iter_dataset_samples(args))
        if not samples:
            raise SystemExit(f"No samples found in {args.data_root}")
    else:
        samples = []

    for job in jobs:
        model = None
        try:
            print(f"[model] loading {job.model_id}", flush=True)
            model = job.loader(device)
            params = sum(p.numel() for p in model.parameters())
            codec = None if job.model_name == "CRA5" else CompressAILikeCodec(model, device=device, divisor=job.divisor)
            for sample in samples:
                print(f"[sample] {job.model_id} {sample.sample_id}", flush=True)
                if job.model_name == "CRA5":
                    result = run_cra5_sample(sample, model, device=device)
                else:
                    result = run_image_grouped_sample(sample, codec)
                result.update({
                    "model_name": job.model_name,
                    "model_id": job.model_id,
                    "metric": job.metric,
                    "checkpoint": job.checkpoint,
                    "params": params,
                })
                summary.append(result)
                print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:
            summary.append({
                "model_name": job.model_name,
                "model_id": job.model_id,
                "metric": job.metric,
                "checkpoint": job.checkpoint,
                "error": str(exc),
            })
            print(f"[error] {job.model_id}: {exc}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {summary_file}", flush=True)


def requested_caesar_models(requested_models):
    if requested_models is None:
        return []
    if "CAESAR" in requested_models:
        return ["caesar_v", "caesar_d"]
    models = []
    aliases = {
        "caesar_v": "caesar_v",
        "CAESAR-V": "caesar_v",
        "caesar_d": "caesar_d",
        "CAESAR-D": "caesar_d",
    }
    for requested in requested_models:
        if requested in aliases:
            models.append(aliases[requested])
    return models


if __name__ == "__main__":
    main()
