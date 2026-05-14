import importlib.util
import json
import tempfile
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
                    "bpp": 16.0,
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
        self.assertEqual(16.0, row["bpp"])
        self.assertEqual(0.0625, row["bpv"])
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

        self.assertAlmostEqual((32.0 / 468.18) * 268.0, row["bpp"], places=6)
        self.assertAlmostEqual(32.0 / 468.18, row["bpv"], places=8)
        self.assertAlmostEqual(1112.99328, row["encode_throughput"], places=5)
        self.assertAlmostEqual(556.49664, row["decode_throughput"], places=5)

    def test_grouped_rows_merge_caesar_error_bound_variants_by_model_id(self):
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

        self.assertIn("CAESAR-V", groups)
        self.assertEqual(2, len(groups["CAESAR-V"]))
        self.assertEqual(2, len(groups["DCAE"]))

    def test_collect_records_keeps_caesar_error_bound_variants(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            caesar_dir = results_dir / "CAESAR"
            caesar_eb_dir = results_dir / "CAESAR_eb_1em3"
            dcmvc_dir = results_dir / "DCMVC"
            caesar_dir.mkdir()
            caesar_eb_dir.mkdir()
            dcmvc_dir.mkdir()

            base_record = {
                "arch": "CAESAR",
                "model_id": "caesar_v",
                "psnr": 88.0,
                "compression_ratio": 166.0,
            }
            eb_record = {
                "arch": "CAESAR",
                "model_id": "caesar_v",
                "psnr": 87.0,
                "compression_ratio": 177.0,
            }
            dcmvc_record = {
                "arch": "DCMVC",
                "model_id": "DCMVC_Intra_q0",
                "psnr": 74.5,
                "compression_ratio": 1335.6,
            }
            (caesar_dir / "summary.json").write_text(json.dumps([base_record]))
            (caesar_eb_dir / "summary.json").write_text(json.dumps([eb_record]))
            (dcmvc_dir / "summary.json").write_text(json.dumps([dcmvc_record]))

            rows = module.collect_records(
                results_dir=results_dir,
                cra5_summary=results_dir / "missing_cra5_summary.json",
            )

        self.assertEqual(
            ["CAESAR", "CAESAR_eb_1em3", "DCMVC"],
            [row["source"] for row in rows],
        )

    def test_plot_label_distinguishes_caesar_source_and_model_id(self):
        module = load_module()

        caesar_row = module.normalize_record(
            "CAESAR_eb_1em3",
            {
                "arch": "CAESAR",
                "model_id": "caesar_d",
                "psnr": 87.0,
                "compression_ratio": 177.0,
            },
        )
        dcmvc_row = module.normalize_record(
            "DCMVC",
            {
                "arch": "DCMVC",
                "model_id": "DCMVC_Intra_q0",
                "psnr": 74.5,
                "compression_ratio": 1335.6,
            },
        )

        self.assertEqual("CAESAR-D", module.plot_label(caesar_row))
        self.assertEqual("DCMVC", module.plot_label(dcmvc_row))

    def test_plot_group_key_merges_caesar_error_bound_variants_by_model_id(self):
        module = load_module()

        base_row = module.normalize_record(
            "CAESAR",
            {
                "arch": "CAESAR",
                "model_id": "caesar_v",
                "psnr": 88.0,
                "compression_ratio": 166.0,
            },
        )
        eb_row = module.normalize_record(
            "CAESAR_eb_1em3",
            {
                "arch": "CAESAR",
                "model_id": "caesar_v",
                "psnr": 87.0,
                "compression_ratio": 177.0,
            },
        )
        caesar_d_row = module.normalize_record(
            "CAESAR_eb_3em4",
            {
                "arch": "CAESAR",
                "model_id": "caesar_d",
                "psnr": 78.0,
                "compression_ratio": 213.0,
            },
        )

        self.assertEqual("CAESAR-V", module.plot_group_key(base_row))
        self.assertEqual("CAESAR-V", module.plot_group_key(eb_row))
        self.assertEqual("CAESAR-D", module.plot_group_key(caesar_d_row))

    def test_metric_axis_scales_use_log_for_compression_ratio_and_throughput(self):
        module = load_module()

        self.assertEqual(("log", "linear"), module.metric_axis_scales("psnr"))
        self.assertEqual(("log", "log"), module.metric_axis_scales("encode_throughput"))
        self.assertEqual(("log", "log"), module.metric_axis_scales("decode_throughput"))

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
                ("CAESAR-V", 1_501_356.0),
                ("CAESAR-D", 36_763_025.0),
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
                ("CAESAR-V", 1_501_356.0, 1_501_356.0, [1_501_356.0]),
                ("CAESAR-D", 36_763_025.0, 36_763_025.0, [36_763_025.0]),
                ("LICTCM", 45_180_752.0, 76_568_148.0, [45_180_752.0, 76_568_148.0]),
            ],
            [
                (row["name"], row["min_params"], row["max_params"], row["params_values"])
                for row in ranges
            ],
        )
        self.assertEqual([], missing)

    def test_prepare_rows_for_plot_keeps_single_default_caesar_v_point(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "psnr": 88.8,
                    "compression_ratio": 166.0,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_1em3",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_v",
                    "psnr": 87.4,
                    "compression_ratio": 177.0,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_1em3",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "psnr": 69.7,
                    "compression_ratio": 338.0,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_3em4",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "psnr": 78.0,
                    "compression_ratio": 213.0,
                },
            ),
        ]

        prepared = module.prepare_rows_for_plot(rows)
        grouped = module.grouped_rows(prepared, "psnr")

        self.assertEqual(1, len(grouped["CAESAR-V"]))
        self.assertEqual("CAESAR", grouped["CAESAR-V"][0]["source"])
        self.assertEqual(2, len(grouped["CAESAR-D"]))

    def test_caesar_error_bound_label_parses_from_source_name(self):
        module = load_module()

        self.assertIsNone(module.caesar_error_bound_label("CAESAR"))
        self.assertEqual("1e-3", module.caesar_error_bound_label("CAESAR_eb_1em3"))
        self.assertEqual("1.5e-3", module.caesar_error_bound_label("CAESAR_eb_1p5em3"))

    def test_caesar_annotation_text_only_marks_caesar_d_error_bounds(self):
        module = load_module()

        caesar_v = module.normalize_record(
            "CAESAR_eb_1em3",
            {
                "arch": "CAESAR",
                "model_id": "caesar_v",
                "psnr": 87.0,
                "compression_ratio": 177.0,
            },
        )
        caesar_d = module.normalize_record(
            "CAESAR_eb_1em3",
            {
                "arch": "CAESAR",
                "model_id": "caesar_d",
                "psnr": 69.0,
                "compression_ratio": 338.0,
            },
        )

        self.assertIsNone(module.caesar_annotation_text(caesar_v))
        self.assertEqual("eb=1e-3", module.caesar_annotation_text(caesar_d))

    def test_model_family_classifies_scientific_image_and_video(self):
        module = load_module()

        scientific = module.normalize_record(
            "CAESAR",
            {
                "arch": "CAESAR",
                "model_id": "caesar_d",
                "psnr": 85.0,
                "compression_ratio": 132.0,
            },
        )
        image = module.normalize_record(
            "DCAE",
            {
                "arch": "DCAE",
                "model_id": "DCAE_mse_lmbda0.05",
                "psnr": 81.0,
                "compression_ratio": 512.0,
            },
        )
        video = module.normalize_record(
            "DCMVC",
            {
                "arch": "DCMVC",
                "model_id": "DCMVC_Intra_q0",
                "psnr": 74.5,
                "compression_ratio": 1335.6,
            },
        )

        self.assertEqual("caesar", module.model_family(scientific))
        self.assertEqual("image", module.model_family(image))
        self.assertEqual("video", module.model_family(video))

    def test_reference_xticks_include_500_when_in_range(self):
        module = load_module()

        ticks = module.reference_xticks(
            [
                {"compression_ratio": 166.0},
                {"compression_ratio": 468.0},
                {"compression_ratio": 1335.0},
            ]
        )

        self.assertIn(500, ticks)

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

    def test_select_ranked_slots_picks_low_mid_high_per_model_group(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "DCMVC",
                {
                    "arch": "DCMVC",
                    "model_id": "DCMVC_Intra_q0",
                    "psnr": 74.6,
                    "compression_ratio": 1335.6,
                    "encode_throughput": 182.1,
                    "decode_throughput": 252.5,
                },
            ),
            module.normalize_record(
                "DCMVC",
                {
                    "arch": "DCMVC",
                    "model_id": "DCMVC_Intra_q1",
                    "psnr": 75.0,
                    "compression_ratio": 940.1,
                    "encode_throughput": 195.0,
                    "decode_throughput": 250.3,
                },
            ),
            module.normalize_record(
                "DCMVC",
                {
                    "arch": "DCMVC",
                    "model_id": "DCMVC_Intra_q2",
                    "psnr": 75.2,
                    "compression_ratio": 645.6,
                    "encode_throughput": 195.4,
                    "decode_throughput": 248.4,
                },
            ),
            module.normalize_record(
                "DCMVC",
                {
                    "arch": "DCMVC",
                    "model_id": "DCMVC_Intra_q3",
                    "psnr": 75.2,
                    "compression_ratio": 461.7,
                    "encode_throughput": 195.3,
                    "decode_throughput": 245.0,
                },
            ),
        ]

        slot_map = module.select_ranked_slots(rows, slot_count=3)

        self.assertEqual(["slot_1", "slot_2", "slot_3"], list(slot_map))
        self.assertEqual(["DCMVC_Intra_q3"], [row["model_id"] for row in slot_map["slot_1"]])
        self.assertEqual(["DCMVC_Intra_q2"], [row["model_id"] for row in slot_map["slot_2"]])
        self.assertEqual(["DCMVC_Intra_q0"], [row["model_id"] for row in slot_map["slot_3"]])

    def test_select_representative_points_picks_cr_speed_and_tradeoff(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "CAESAR_eb_3em3",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "psnr": 61.9,
                    "compression_ratio": 414.9,
                    "encode_throughput": 5.1,
                    "decode_throughput": 5.6,
                },
            ),
            module.normalize_record(
                "CAESAR_eb_3em4",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "psnr": 78.0,
                    "compression_ratio": 212.9,
                    "encode_throughput": 7.2,
                    "decode_throughput": 7.6,
                },
            ),
            module.normalize_record(
                "CAESAR",
                {
                    "arch": "CAESAR",
                    "model_id": "caesar_d",
                    "psnr": 85.6,
                    "compression_ratio": 132.3,
                    "encode_throughput": 5.3,
                    "decode_throughput": 5.6,
                },
            ),
        ]

        representative_map = module.select_representative_points(rows)

        self.assertEqual("CAESAR_eb_3em3", representative_map["best_cr"][0]["source"])
        self.assertEqual("CAESAR_eb_3em4", representative_map["best_speed"][0]["source"])
        self.assertEqual("CAESAR_eb_3em4", representative_map["best_tradeoff"][0]["source"])

    def test_grouped_bar_metric_specs_include_psnr_panel(self):
        module = load_module()

        specs = module.grouped_bar_metric_specs()

        self.assertEqual(
            ["psnr", "compression_ratio", "encode_throughput", "decode_throughput"],
            [spec[0] for spec in specs],
        )
        self.assertEqual("PSNR (dB)", specs[0][1])

    def test_selection_series_styles_use_distinct_non_gray_colors(self):
        module = load_module()

        styles = module.selection_series_styles(["best_cr", "best_speed", "best_tradeoff"])

        self.assertEqual(3, len({style["color"] for style in styles.values()}))
        self.assertNotEqual("#bdbdbd", styles["best_cr"]["color"])

    def test_bpv_plot_specs_split_into_image_video_and_scientific_groups(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "DCAE",
                {"arch": "DCAE", "model_id": "DCAE_1", "psnr": 80.0, "compression_ratio": 500.0},
            ),
            module.normalize_record(
                "DCMVC",
                {"arch": "DCMVC", "model_id": "DCMVC_q0", "psnr": 75.0, "compression_ratio": 900.0},
            ),
            module.normalize_record(
                "CRA5",
                {"model": "CRA5-VAEformer", "psnr": 85.0, "compression_ratio": 468.0, "data_shape": [268, 10, 10]},
            ),
            module.normalize_record(
                "CAESAR",
                {"arch": "CAESAR", "model_id": "caesar_v", "psnr": 88.0, "compression_ratio": 166.0},
            ),
        ]

        specs = module.bpv_plot_specs(rows)

        self.assertEqual(
            ["scientific_family", "video_family", "image_family"],
            [spec[1] for spec in specs],
        )
        self.assertEqual(
            [["CAESAR-V", "CRA5-VAEformer"], ["DCMVC"], ["DCAE"]],
            [[module.plot_group_key(row) for row in spec[0]] for spec in specs],
        )

    def test_bpv_bar_width_helper_prefers_thinner_bars_for_dense_points(self):
        module = load_module()

        width = module._bar_width_from_sorted_x([0.012, 0.016, 0.020, 0.024])

        self.assertLess(width, 0.002)

    def test_throughput_bpv_overview_specs_return_encode_and_decode(self):
        module = load_module()

        specs = module.throughput_bpv_overview_specs()

        self.assertEqual(
            [
                ("encode_throughput", "Encode Throughput vs BPV", "encode_throughput_vs_bpv.png"),
                ("decode_throughput", "Decode Throughput vs BPV", "decode_throughput_vs_bpv.png"),
            ],
            specs,
        )

    def test_bpv_plot_specs_are_reused_for_throughput_relation_plots(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "DCAE",
                {"arch": "DCAE", "model_id": "DCAE_1", "psnr": 80.0, "compression_ratio": 500.0},
            ),
            module.normalize_record(
                "DCMVC",
                {"arch": "DCMVC", "model_id": "DCMVC_q0", "psnr": 75.0, "compression_ratio": 900.0},
            ),
            module.normalize_record(
                "CRA5",
                {"model": "CRA5-VAEformer", "psnr": 85.0, "compression_ratio": 468.0, "data_shape": [268, 10, 10]},
            ),
        ]

        specs = module.bpv_plot_specs(rows)

        self.assertEqual(
            ["scientific_family", "video_family", "image_family"],
            [spec[1] for spec in specs],
        )

    def test_average_throughput_rows_average_per_model(self):
        module = load_module()

        rows = [
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_a",
                    "psnr": 80.0,
                    "compression_ratio": 500.0,
                    "encode_throughput": 80.0,
                    "decode_throughput": 70.0,
                },
            ),
            module.normalize_record(
                "DCAE",
                {
                    "arch": "DCAE",
                    "model_id": "DCAE_b",
                    "psnr": 79.0,
                    "compression_ratio": 800.0,
                    "encode_throughput": 100.0,
                    "decode_throughput": 90.0,
                },
            ),
            module.normalize_record(
                "DCMVC",
                {
                    "arch": "DCMVC",
                    "model_id": "DCMVC_q0",
                    "psnr": 75.0,
                    "compression_ratio": 900.0,
                    "encode_throughput": 200.0,
                    "decode_throughput": 250.0,
                },
            ),
        ]

        averages = module.average_throughput_rows(rows)

        self.assertEqual(
            [
                ("DCAE", 90.0, 80.0),
                ("DCMVC", 200.0, 250.0),
            ],
            [(row["name"], row["encode_throughput"], row["decode_throughput"]) for row in averages],
        )

    def test_bpv_stale_plot_names_include_old_bar_outputs(self):
        module = load_module()

        stale_names = module.bpv_stale_plot_names("image_family")

        self.assertEqual(
            [
                "encode_throughput_vs_bpv.png",
                "decode_throughput_vs_bpv.png",
                "image_family_encode_throughput_vs_bpv.png",
                "image_family_decode_throughput_vs_bpv.png",
                "image_family_encode_throughput_by_bpv_bar.png",
                "image_family_decode_throughput_by_bpv_bar.png",
            ],
            stale_names,
        )


if __name__ == "__main__":
    unittest.main()
