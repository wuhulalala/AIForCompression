"""Download ERA5 pressure/single pairs for CAESAR and other ERA5 tests."""
import argparse
from datetime import date, datetime, timedelta
import os
import zipfile
from pathlib import Path

import cdsapi

# CDS API settings (from cra5/api/era5_downloader.py)
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "ea3a2607-158c-48a4-bd27-b255256b2759"

DEFAULT_DATA_DIR = Path("/data/run01/scxj523/zsh/project/Data/ERA5/2024")

PRESSURE_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "vertical_velocity",
]

PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250",
    "300", "350", "400", "450", "500", "550", "600", "650", "700",
    "750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000",
]

SINGLE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_cloud_cover",
    "surface_pressure",
    "total_precipitation",
    "mean_sea_level_pressure",
]


def build_days(start_date, count):
    start = date.fromisoformat(start_date)
    return [start + timedelta(days=idx) for idx in range(count)]


def output_paths(data_dir, day, time):
    timestamp = f"{day.isoformat()}T{time}:00"
    data_dir = Path(data_dir)
    return data_dir / f"{timestamp}_pressure.nc", data_dir / f"{timestamp}_single.nc"


def extract_or_move(tmp_file, outfile, data_dir):
    tmp_file = Path(tmp_file)
    outfile = Path(outfile)
    data_dir = Path(data_dir)

    if zipfile.is_zipfile(tmp_file):
        print("Downloaded file is a zip, extracting...")
        import xarray as xr

        with zipfile.ZipFile(tmp_file, "r") as zf:
            nc_files = [name for name in zf.namelist() if name.endswith(".nc")]
            print(f"  Found {len(nc_files)} nc files in zip: {nc_files}")
            zf.extractall(data_dir, members=nc_files)
            extracted_paths = [data_dir / name for name in nc_files]

        if len(extracted_paths) == 1:
            extracted_paths[0].replace(outfile)
        else:
            datasets = [xr.open_dataset(path) for path in extracted_paths]
            merged = xr.merge(datasets)
            for ds in datasets:
                ds.close()
            merged.to_netcdf(outfile)
            merged.close()
            print(f"  Merged {len(extracted_paths)} files into {outfile}")
            for path in extracted_paths:
                if path.exists() and path != outfile:
                    path.unlink()
        tmp_file.unlink()
    else:
        tmp_file.replace(outfile)


def download_pressure(client, data_dir, day, time):
    outfile, _ = output_paths(data_dir, day, time)
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"Pressure file already exists: {outfile} ({outfile.stat().st_size} bytes)")
        return

    print(f"Downloading pressure level data for {day.isoformat()} {time}...")
    print(f"  Variables: {PRESSURE_VARIABLES}")
    print(f"  Levels: {len(PRESSURE_LEVELS)} levels")
    print(f"  Output: {outfile}")

    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": day.strftime("%d"),
        "time": time,
        "data_format": "netcdf",
    }

    tmp_file = outfile.with_suffix(outfile.suffix + ".tmp")
    client.retrieve("reanalysis-era5-pressure-levels", request, str(tmp_file))
    extract_or_move(tmp_file, outfile, data_dir)
    print(f"Done! File size: {outfile.stat().st_size} bytes")


def download_single(client, data_dir, day, time):
    _, outfile = output_paths(data_dir, day, time)
    if outfile.exists() and outfile.stat().st_size > 100_000:
        print(f"Single file already exists: {outfile} ({outfile.stat().st_size} bytes)")
        return

    print(f"Downloading single level data for {day.isoformat()} {time}...")
    print(f"  Variables: {SINGLE_VARIABLES}")
    print(f"  Output: {outfile}")

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_VARIABLES,
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": day.strftime("%d"),
        "time": time,
        "data_format": "netcdf",
    }

    tmp_file = outfile.with_suffix(outfile.suffix + ".tmp")
    client.retrieve("reanalysis-era5-single-levels", request, str(tmp_file))
    extract_or_move(tmp_file, outfile, data_dir)
    print(f"Done! File size: {outfile.stat().st_size} bytes")


def verify_data(data_dir, day, time):
    """Verify the downloaded data has correct structure."""
    try:
        import xarray as xr
        pressure_f, single_f = output_paths(data_dir, day, time)

        if not pressure_f.exists():
            print(f"MISSING: {pressure_f}")
            return False
        if not single_f.exists():
            print(f"MISSING: {single_f}")
            return False

        ds_p = xr.open_dataset(pressure_f)
        ds_s = xr.open_dataset(single_f)

        print("\n=== Pressure file ===")
        print(f"  Variables: {list(ds_p.data_vars)}")
        print(f"  Pressure levels: {len(ds_p.pressure_level) if 'pressure_level' in ds_p.sizes else 'N/A'}")
        print(f"  Shape: lat={len(ds_p.latitude)}, lon={len(ds_p.longitude)}")

        print("\n=== Single file ===")
        print(f"  Variables: {list(ds_s.data_vars)}")
        print(f"  Shape: lat={len(ds_s.latitude)}, lon={len(ds_s.longitude)}")

        n_pressure = len(list(ds_p.data_vars)) * len(ds_p.pressure_level)
        n_single = len(list(ds_s.data_vars))
        total = n_pressure + n_single
        print(f"\nTotal channels: {n_pressure} (pressure) + {n_single} (single) = {total}")

        ds_p.close()
        ds_s.close()
        return True
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Download contiguous ERA5 pressure/single nc pairs.")
    parser.add_argument("--start-date", default="2024-06-01", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--days", type=int, default=16, help="Number of contiguous days to download.")
    parser.add_argument("--time", default="00:00", help="ERA5 time of day, HH:MM.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datetime.strptime(args.time, "%H:%M")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for day in build_days(args.start_date, args.days):
        print(f"\n{'=' * 60}")
        print(f"Downloading data for {day.isoformat()} {args.time}")
        print(f"{'=' * 60}")
        download_pressure(client, args.data_dir, day, args.time)
        download_single(client, args.data_dir, day, args.time)
        if not args.skip_verify:
            verify_data(args.data_dir, day, args.time)


if __name__ == "__main__":
    main()
