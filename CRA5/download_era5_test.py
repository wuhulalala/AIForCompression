"""
Download ERA5 pressure level data for testing.
Uses CDS API with the key from cra5/api/era5_downloader.py.
Only downloads the pressure file for 2024-06-01T00:00:00 (single file already exists).
"""
import os
import sys
import cdsapi
import zipfile

# CDS API settings (from cra5/api/era5_downloader.py)
os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
os.environ['CDSAPI_KEY'] = 'ea3a2607-158c-48a4-bd27-b255256b2759'

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ERA5', '2024')
os.makedirs(DATA_DIR, exist_ok=True)

TIMESTAMP = '2024-06-01T00:00:00'

PRESSURE_VARIABLES = [
    'geopotential',
    'specific_humidity',
    'u_component_of_wind',
    'v_component_of_wind',
    'temperature',
    'relative_humidity',
    'vertical_velocity',
]

PRESSURE_LEVELS = [
    '1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
    '100', '125', '150', '175', '200', '225', '250',
    '300', '350', '400', '450', '500', '550', '600', '650', '700',
    '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000',
]


def download_pressure():
    outfile = os.path.join(DATA_DIR, f'{TIMESTAMP}_pressure.nc')
    if os.path.exists(outfile) and os.path.getsize(outfile) > 1_000_000:
        print(f"Pressure file already exists: {outfile} ({os.path.getsize(outfile)} bytes)")
        return

    print(f"Downloading pressure level data for {TIMESTAMP}...")
    print(f"  Variables: {PRESSURE_VARIABLES}")
    print(f"  Levels: {len(PRESSURE_LEVELS)} levels")
    print(f"  Output: {outfile}")

    client = cdsapi.Client()

    request = {
        'product_type': ['reanalysis'],
        'variable': PRESSURE_VARIABLES,
        'pressure_level': PRESSURE_LEVELS,
        'year': '2024',
        'month': '06',
        'day': '01',
        'time': '00:00',
        'data_format': 'netcdf',
    }

    tmp_file = outfile + '.tmp'
    client.retrieve('reanalysis-era5-pressure-levels', request, tmp_file)

    # Handle zip files
    if zipfile.is_zipfile(tmp_file):
        print("Downloaded file is a zip, extracting...")
        with zipfile.ZipFile(tmp_file, 'r') as zf:
            nc_files = [f for f in zf.namelist() if f.endswith('.nc')]
            if nc_files:
                zf.extract(nc_files[0], DATA_DIR)
                extracted = os.path.join(DATA_DIR, nc_files[0])
                os.rename(extracted, outfile)
                print(f"Extracted {nc_files[0]} -> {outfile}")
        os.remove(tmp_file)
    else:
        os.rename(tmp_file, outfile)

    print(f"Done! File size: {os.path.getsize(outfile)} bytes")


def verify_data():
    """Verify the downloaded data has correct structure."""
    try:
        import xarray as xr
        pressure_f = os.path.join(DATA_DIR, f'{TIMESTAMP}_pressure.nc')
        single_f = os.path.join(DATA_DIR, f'{TIMESTAMP}_single.nc')

        if not os.path.exists(pressure_f):
            print(f"MISSING: {pressure_f}")
            return False
        if not os.path.exists(single_f):
            print(f"MISSING: {single_f}")
            return False

        ds_p = xr.open_dataset(pressure_f)
        ds_s = xr.open_dataset(single_f)

        print(f"\n=== Pressure file ===")
        print(f"  Variables: {list(ds_p.data_vars)}")
        print(f"  Pressure levels: {len(ds_p.pressure_level) if 'pressure_level' in ds_p.dims else 'N/A'}")
        print(f"  Shape: lat={len(ds_p.latitude)}, lon={len(ds_p.longitude)}")

        print(f"\n=== Single file ===")
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


if __name__ == '__main__':
    download_pressure()
    verify_data()
