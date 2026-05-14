import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression_pipeline.adapters.era5 import ERA5Adapter
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.adapters.tomo_h5 import TomoH5Adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare canonical samples/views for supported compression datasets.")
    parser.add_argument("--dataset", choices=["era5", "kodak", "tomo"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kodak":
        adapter = KodakAdapter(args.data_root)
        samples = list(adapter.iter_samples(max_samples=args.max_samples))
        manifest = adapter.manifest()
    elif args.dataset == "tomo":
        adapter = TomoH5Adapter(args.data_root)
        samples = list(
            adapter.iter_samples(
                max_samples=args.max_samples,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        )
        manifest = adapter.manifest()
    else:
        adapter = ERA5Adapter(args.data_root)
        samples = list(
            adapter.iter_samples(
                max_samples=args.max_samples,
                max_channels=args.max_channels,
                resolution=tuple(args.resolution) if args.resolution else None,
            )
        )
        manifest = adapter.manifest()

    sample_rows = []
    for sample in samples:
        array_path = output_dir / f"{sample.sample_id}.npy"
        with array_path.open("wb") as f:
            import numpy as np

            np.save(f, sample.array)
        sample_rows.append({
            "dataset_id": sample.dataset_id,
            "sample_id": sample.sample_id,
            "kind": sample.kind,
            "layout": sample.layout,
            "array_path": str(array_path),
            "shape": list(sample.array.shape),
            "metadata": sample.metadata,
        })

    manifest_payload = {
        "dataset_id": manifest.dataset_id,
        "dataset_name": manifest.dataset_name,
        "dataset_type": manifest.dataset_type,
        "source_format": manifest.source_format,
        "canonical_layout": manifest.canonical_layout,
        "sample_count": len(samples),
        "metadata": manifest.metadata,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(samples)} canonical samples to {output_dir}")


if __name__ == "__main__":
    main()
