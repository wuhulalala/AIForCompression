import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "download_era5.py"


def load_download_module():
    sys.modules.setdefault("cdsapi", types.SimpleNamespace(Client=object))
    spec = importlib.util.spec_from_file_location("download_era5", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadEra5Test(unittest.TestCase):
    def test_build_days_from_start_date_returns_contiguous_dates(self):
        module = load_download_module()

        days = module.build_days("2024-06-01", 8)

        self.assertEqual(
            ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
            [day.isoformat() for day in days[:4]],
        )
        self.assertEqual("2024-06-08", days[-1].isoformat())

    def test_output_paths_match_caesar_pair_naming(self):
        module = load_download_module()
        day = module.build_days("2024-06-01", 1)[0]

        pressure, single = module.output_paths(Path("/tmp/era5"), day, "00:00")

        self.assertEqual(Path("/tmp/era5/2024-06-01T00:00:00_pressure.nc"), pressure)
        self.assertEqual(Path("/tmp/era5/2024-06-01T00:00:00_single.nc"), single)


if __name__ == "__main__":
    unittest.main()
