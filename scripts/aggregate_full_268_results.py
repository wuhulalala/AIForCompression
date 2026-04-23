import argparse
import json
import math
from pathlib import Path


RESULT_SOURCES = [
    ("DCAE", "models/DCAE/results_era5/summary.json"),
    ("LIC_TCM", "models/LIC_TCM/results_era5/summary.json"),
    ("WeConvene", "models/WeConvene/results_era5/summary.json"),
    ("CompressAI", "models/CRA5/results/all_compressai/summary.json"),
]


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def throughput(original_bytes, seconds):
    if not original_bytes or not seconds or seconds <= 0:
        return None
    return original_bytes / seconds


def mean_group_rmse(group_rmse):
    values = []
    for metrics in (group_rmse or {}).values():
        if "mean_rmse" in metrics:
            values.append(metrics["mean_rmse"])
        elif "rmse" in metrics:
            values.append(metrics["rmse"])
    if not values:
        return None
    return sum(values) / len(values)


def normalize_record(source_name, record):
    if "error" in record:
        return {
            "source": source_name,
            "model_id": record.get("model_id"),
            "arch": record.get("arch", source_name),
            "error": record["error"],
        }

    compress = record.get("compress", {})
    mse = compress.get("mse")
    rmse = math.sqrt(mse) if mse is not None and mse >= 0 else mean_group_rmse(record.get("group_rmse"))
    original_bytes = compress.get("original_bytes")
    encode_time = compress.get("encode_time")
    decode_time = compress.get("decode_time")

    return {
        "source": source_name,
        "model_id": record.get("model_id"),
        "arch": record.get("arch", source_name),
        "metric": record.get("metric"),
        "quality": record.get("quality"),
        "lmbda": record.get("lmbda"),
        "timestamp": record.get("timestamp"),
        "params": record.get("params"),
        "mse": finite_or_none(mse),
        "rmse": finite_or_none(rmse),
        "psnr": finite_or_none(compress.get("psnr")),
        "bpp": finite_or_none(compress.get("bpp")),
        "compression_ratio": finite_or_none(compress.get("compression_ratio")),
        "encode_time_avg": finite_or_none(encode_time),
        "decode_time_avg": finite_or_none(decode_time),
        "encode_throughput": finite_or_none(throughput(original_bytes, encode_time)),
        "decode_throughput": finite_or_none(throughput(original_bytes, decode_time)),
        "bitstream_bytes": compress.get("bitstream_bytes"),
        "original_bytes": original_bytes,
        "group_rmse": record.get("group_rmse"),
    }


def aggregate(project_root):
    rows = []
    missing = []
    errors = []
    for source_name, rel_path in RESULT_SOURCES:
        path = project_root / rel_path
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r") as f:
            records = json.load(f)
        for record in records:
            normalized = normalize_record(source_name, record)
            rows.append(normalized)
            if "error" in normalized:
                errors.append(normalized)
    return rows, missing, errors


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate full 268-channel ERA5 benchmark results.")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.project_root / "unified_results" / "full_268_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows, missing, errors = aggregate(args.project_root)
    payload = {
        "num_results": len(rows),
        "num_errors": len(errors),
        "missing_sources": missing,
        "required_metrics": [
            "params",
            "mse",
            "rmse",
            "psnr",
            "bpp",
            "compression_ratio",
            "encode_time_avg",
            "decode_time_avg",
            "encode_throughput",
            "decode_throughput",
        ],
        "results": rows,
    }
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(rows)} results to {output}")
    if missing:
        print("Missing sources:")
        for path in missing:
            print(f"  {path}")
    if errors:
        print(f"Records with errors: {len(errors)}")


if __name__ == "__main__":
    main()
