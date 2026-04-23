"""Compute mean/std from ERA5 data for normalization."""
import argparse
import json
import numpy as np
import xarray as xr
import os

DATA_DIR = '/data/run01/scxj523/zsh/project/AIForCompression/Data/2011'
OUT_DIR = '/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/api'

DAYS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
PL_VARS = ['z', 'q', 'u', 'v', 't', 'r', 'w']

def compute_pressure_file(pl_file, key):
    ds = xr.open_dataset(pl_file)
    entry = {"mean": {}, "std": {}}
    for var in PL_VARS:
        if var not in ds:
            continue
        data = ds[var].values
        entry["mean"][f"{var}_overall"] = float(np.nanmean(data))
        entry["std"][f"{var}_overall"] = float(np.nanstd(data))

        level_axis = data.ndim - 3
        n_levels = data.shape[level_axis]
        entry["mean"][var] = [
            float(np.nanmean(np.take(data, i, axis=level_axis)))
            for i in range(n_levels)
        ]
        entry["std"][var] = [
            float(np.nanstd(np.take(data, i, axis=level_axis)))
            for i in range(n_levels)
        ]
    ds.close()
    print(f"  {key} pressure done")
    return entry


def compute_single_file(sl_file, key):
    ds = xr.open_dataset(sl_file)
    entry = {"mean": {}, "std": {}}
    for var in ds.data_vars:
        data = ds[var].values
        entry["mean"][var] = float(np.nanmean(data))
        entry["std"][var] = float(np.nanstd(data))
    ds.close()
    print(f"  {key} single done")
    return entry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--pressure-file", default=None)
    parser.add_argument("--single-file", default=None)
    parser.add_argument("--key", default=None)
    return parser.parse_args()


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_pressure_outputs(out_dir, result):
    write_json(os.path.join(out_dir, "mean_std.json"), result)
    for key, entry in result.items():
        write_json(os.path.join(out_dir, f"mean_std_{key}.json"), {key: entry})


def write_single_outputs(out_dir, result):
    write_json(os.path.join(out_dir, "mean_std_single.json"), result)
    for key, entry in result.items():
        write_json(os.path.join(out_dir, f"mean_std_single_{key}.json"), {key: entry})


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pressure_file or args.single_file:
        key = args.key or os.path.basename(args.data_dir.rstrip(os.sep))
        if args.pressure_file:
            print("Computing pressure level mean/std...")
            pl_result = {key: compute_pressure_file(args.pressure_file, key)}
            write_pressure_outputs(args.out_dir, pl_result)
            print(f"Saved mean_std.json and mean_std_{key}.json")

        if args.single_file:
            print("\nComputing single level mean/std...")
            sl_result = {key: compute_single_file(args.single_file, key)}
            write_single_outputs(args.out_dir, sl_result)
            print(f"Saved mean_std_single.json and mean_std_single_{key}.json")

        print("\nDone!")
        return

    print("Computing pressure level mean/std per day...")
    pl_result = {}
    for day in DAYS:
        key = f"2011-06-{day:02d}"
        pl_file = os.path.join(args.data_dir, f"era5pl_2011_06_{day:02d}_00.nc")
        pl_result[key] = compute_pressure_file(pl_file, key)

    write_pressure_outputs(args.out_dir, pl_result)
    print("Saved mean_std.json and per-day mean_std_*.json")

    print("\nComputing single level mean/std per day...")
    sl_result = {}
    for day in DAYS:
        key = f"2011-06-{day:02d}"
        sl_file = os.path.join(args.data_dir, f"era5sl_2011_06_{day:02d}_00.nc")
        sl_result[key] = compute_single_file(sl_file, key)

    write_single_outputs(args.out_dir, sl_result)
    print("Saved mean_std_single.json and per-day mean_std_single_*.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
