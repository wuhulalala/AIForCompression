import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "plot_results_summary.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_results_summary", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotResultsSummaryTest(unittest.TestCase):
    def test_normalize_record_reads_nested_compress_and_converts_bytes_per_second(self):
        module = load_module()

        row = module.normalize_record(
            "DCAE",
            {
                "model_id": "DCAE_mse_lmbda0.05",
                "arch": "DCAE",
                "compress": {
                    "psnr": 81.5,
                    "compression_ratio": 512.0,
                    "original_bytes": 200_000_000,
                    "encode_time": 2.0,
                    "decode_time": 4.0,
                },
            },
        )

        self.assertEqual("DCAE", row["label"])
        self.assertEqual(512.0, row["compression_ratio"])
        self.assertEqual(81.5, row["psnr"])
        self.assertEqual(100.0, row["encode_throughput"])
        self.assertEqual(50.0, row["decode_throughput"])

    def test_normalize_record_keeps_existing_megabytes_per_second_values(self):
        module = load_module()

        row = module.normalize_record(
            "LIC-HPCM-base",
            {
                "model_id": "LIC-HPCM-base_0.0018",
                "arch": "LIC-HPCM-base",
                "psnr": 74.2,
                "compression_ratio": 3159.8,
                "encode_throughput": 98.0,
                "decode_throughput": 116.3,
            },
        )

        self.assertEqual(98.0, row["encode_throughput"])
        self.assertEqual(116.3, row["decode_throughput"])

    def test_normalize_record_computes_cra5_throughput_in_megabytes_per_second(self):
        module = load_module()

        row = module.normalize_record(
            "CRA5",
            {
                "model": "CRA5-VAEformer",
                "psnr": 85.08,
                "compression_ratio": 468.18,
                "data_shape": [268, 721, 1440],
                "avg_encode_time": 1.0,
                "avg_decode_time": 2.0,
            },
        )

        self.assertAlmostEqual(1112.99328, row["encode_throughput"], places=5)
        self.assertAlmostEqual(556.49664, row["decode_throughput"], places=5)

    def test_caesar_records_are_split_by_source_and_model_id(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "psnr": 88.0,
                    "compression_ratio": 166.0,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_1em3",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "psnr": 87.0,
                    "compression_ratio": 177.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_1",
                    "psnr": 81.0,
                    "compression_ratio": 512.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_2",
                    "psnr": 80.0,
                    "compression_ratio": 700.0,
                },
            ),
        ]

        groups = module.grouped_rows(rows, "psnr")

        self.assertIn("CAESAR/caesar_v", groups)
        self.assertIn("CAESAR_eb_1em3/caesar_v", groups)
        self.assertEqual(1, len(groups["CAESAR/caesar_v"]))
        self.assertEqual(1, len(groups["CAESAR_eb_1em3/caesar_v"]))
        self.assertEqual(2, len(groups["DCAE"]))

    def test_model_params_are_grouped_by_concrete_model_id(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "params": 1_501_356,
                    "psnr": 88.0,
                    "compression_ratio": 166.0,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_1em3",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "params": 1_501_356,
                    "psnr": 87.0,
                    "compression_ratio": 177.0,
                },
            ),
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "params": 36_763_025,
                    "psnr": 85.0,
                    "compression_ratio": 132.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_mse_lmbda0.05",
                    "params": 119_400_351,
                    "psnr": 81.0,
                    "compression_ratio": 512.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_mse_lmbda0.025",
                    "params": 118_000_000,
                    "psnr": 80.0,
                    "compression_ratio": 700.0,
                },
            ),
            module.normalize_record(
                "CRA5",
                {
                    "model": "CRA5-VAEformer",
                    "psnr": 85.0,
                    "compression_ratio": 468.0,
                },
            ),
        ]

        params, missing = module.model_param_rows(rows)

        self.assertEqual(
            [
                ("CAESAR caesar_v", 1_501_356.0),
                ("CAESAR caesar_d", 36_763_025.0),
                ("DCAE_mse_lmbda0.025", 118_000_000.0),
                ("DCAE_mse_lmbda0.05", 119_400_351.0),
                ("CRA5-VAEformer", 404_900_000.0),
            ],
            [(row["name"], row["params"]) for row in params],
        )
        self.assertEqual([], missing)

    def test_model_param_ranges_group_quality_variants(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "LIC_TCM",
                {
                    "arch": "LICTCM",
                    "model_id": "LICTCM_mse_lmbda0.05",
                    "params": 45_180_752,
                    "psnr": 81.0,
                    "compression_ratio": 465.0,
                },
            ),
            module.normalize_record(
                "LIC_TCM",
                {
                    "arch": "LICTCM",
                    "model_id": "LICTCM_mse_lmbda0.05_large",
                    "params": 76_568_148,
                    "psnr": 82.0,
                    "compression_ratio": 500.0,
                },
            ),
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "params": 1_501_356,
                    "psnr": 88.0,
                    "compression_ratio": 166.0,
                },
            ),
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "params": 36_763_025,
                    "psnr": 85.0,
                    "compression_ratio": 132.0,
                },
            ),
        ]

        ranges, missing = module.model_param_ranges(rows)

        self.assertEqual(
            [
                ("CAESAR caesar_v", 1_501_356.0, 1_501_356.0, [1_501_356.0]),
                ("CAESAR caesar_d", 36_763_025.0, 36_763_025.0, [36_763_025.0]),
                ("LICTCM", 45_180_752.0, 76_568_148.0, [45_180_752.0, 76_568_148.0]),
            ],
            [
                (row["name"], row["min_params"], row["max_params"], row["params_values"])
                for row in ranges
            ],
        )
        self.assertEqual([], missing)

    def test_model_param_summary_uses_one_bar_per_model_with_range_label(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "LIC_TCM",
                {
                    "arch": "LICTCM",
                    "model_id": "LICTCM_mse_lmbda0.05",
                    "params": 45_180_752,
                    "psnr": 81.0,
                    "compression_ratio": 465.0,
                },
            ),
            module.normalize_record(
                "LIC_TCM",
                {
                    "arch": "LICTCM",
                    "model_id": "LICTCM_mse_lmbda0.05_large",
                    "params": 76_568_148,
                    "psnr": 82.0,
                    "compression_ratio": 500.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_mse_lmbda0.05",
                    "params": 119_400_351,
                    "psnr": 81.0,
                    "compression_ratio": 512.0,
                },
            ),
        ]

        summary, missing = module.model_param_summary_rows(rows)

        self.assertEqual(
            [
                ("LICTCM", 76_568_148.0, "45.2-76.6M"),
                ("DCAE", 119_400_351.0, "119.4M"),
            ],
            [(row["name"], row["params"], row["label_text"]) for row in summary],
        )
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
