"""Compute per-day mean/std for ERA5 2024 data, split into pressure and single files."""

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr


DATA_ROOT = Path("/data/run01/scxj523/zsh/project/Data/ERA5/2024")
OUT_DIR = Path("/data/run01/scxj523/zsh/project/AIForCompression/normalization")

VNAMES_PRESSURE = ["z", "q", "u", "v", "t", "r", "w"]
VNAMES_SINGLE = ["t2m", "u10", "v10", "u100", "v100", "tcc", "sp", "msl", "tp"]
PRESSURE_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]


def find_dates():
    dates = set()
    for f in os.listdir(DATA_ROOT):
        if f.endswith("_pressure.nc"):
            ts = f.replace("_pressure.nc", "")
            dates.add(ts)
    return sorted(dates)


def compute_day(pressure_file, single_file):
    pressure_data = xr.open_dataset(pressure_file, engine="netcdf4")
    single_data = xr.open_dataset(single_file, engine="netcdf4")

    # Pressure variables
    pressure_result = {"mean": {}, "std": {}}
    available_levels = list(pressure_data.pressure_level.data)
    level_mapping = [available_levels.index(l) for l in PRESSURE_LEVELS if l in available_levels]

    for vname in VNAMES_PRESSURE:
        data = pressure_data[vname].values.astype(np.float64)  # (time, level, H, W)
        mean_per_level = []
        std_per_level = []
        for idx in level_mapping:
            level_data = data[:, idx, :, :]
            mean_per_level.append(float(np.nanmean(level_data)))
            std_per_level.append(float(np.nanstd(level_data)))
        pressure_result["mean"][f"{vname}_overall"] = float(np.nanmean(data))
        pressure_result["std"][f"{vname}_overall"] = float(np.nanstd(data))
        pressure_result["mean"][vname] = mean_per_level
        pressure_result["std"][vname] = std_per_level

    # Single level variables
    single_result = {"mean": {}, "std": {}}
    for vname in VNAMES_SINGLE:
        data = single_data[vname].values.astype(np.float64)
        if vname == "tp":
            data = data * 1000  # convert to mm
        single_result["mean"][vname] = float(np.nanmean(data))
        single_result["std"][vname] = float(np.nanstd(data))

    pressure_data.close()
    single_data.close()
    return pressure_result, single_result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = find_dates()
    print(f"Found {len(dates)} dates")

    for ts in dates:
        pressure_file = DATA_ROOT / f"{ts}_pressure.nc"
        single_file = DATA_ROOT / f"{ts}_single.nc"

        day_key = ts.replace("-", "_").replace("T", "_").replace(":", "")
        out_pressure = OUT_DIR / f"mean_std_{day_key}.json"
        out_single = OUT_DIR / f"mean_std_single_{day_key}.json"

        if out_pressure.exists() and out_single.exists():
            print(f"skip {ts}: already exists")
            continue

        print(f"compute {ts}")
        pressure_result, single_result = compute_day(pressure_file, single_file)

        with open(out_pressure, "w") as f:
            json.dump(pressure_result, f, indent=2)
        print(f"  -> {out_pressure.name}")

        with open(out_single, "w") as f:
            json.dump(single_result, f, indent=2)
        print(f"  -> {out_single.name}")

    print("done")


if __name__ == "__main__":
    main()
