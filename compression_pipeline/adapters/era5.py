from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

from compression_pipeline.canonical import CanonicalSample, DatasetManifest
from compression_pipeline.era5_constants import ERA5_CHANNELS, PRESSURE_LEVELS, VNAMES, era5_channel_names


class ERA5Adapter:
    """Reads paired ERA5 pressure/single NetCDF files into canonical CHW samples."""

    def __init__(self, data_root: str | Path, dataset_id: str = "era5_2024") -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id

    def find_pairs(self) -> list[tuple[Path, Path, str]]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"ERA5 data root does not exist: {self.data_root}")
        pairs: list[tuple[Path, Path, str]] = []
        for root, _, files in os.walk(self.data_root):
            for name in sorted(files):
                if not name.endswith("_pressure.nc"):
                    continue
                timestamp = name.replace("_pressure.nc", "")
                pressure = Path(root) / name
                single = Path(root) / f"{timestamp}_single.nc"
                if single.exists():
                    pairs.append((pressure, single, timestamp))
        return sorted(pairs, key=lambda item: item[2])

    def manifest(self) -> DatasetManifest:
        pairs = self.find_pairs()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_name="ERA5",
            dataset_type="scientific_field",
            source_format="netcdf",
            canonical_layout="channel_height_width",
            sample_count=len(pairs),
            metadata={
                "data_root": str(self.data_root),
                "channels": ERA5_CHANNELS,
                "channel_names": era5_channel_names(),
            },
        )

    def read_pair(self, pressure_file: str | Path, single_file: str | Path, max_channels: int | None = None) -> np.ndarray:
        fields: list[np.ndarray] = []
        pressure_data = xr.open_dataset(pressure_file, engine="netcdf4")
        single_data = xr.open_dataset(single_file, engine="netcdf4")
        try:
            available_levels = list(pressure_data.pressure_level.data)
            level_mapping = [available_levels.index(level) for level in PRESSURE_LEVELS]
            for vname in VNAMES["pressure"]:
                data = pressure_data[vname].data
                for level_idx in level_mapping:
                    fields.append(data[0][level_idx][None])
                    if _has_enough_channels(fields, max_channels):
                        return np.concatenate(fields, axis=0).astype(np.float32)
            for vname in VNAMES["single"]:
                data = single_data[vname].data
                if vname == "tp":
                    data = data * 1000
                fields.append(data)
                if _has_enough_channels(fields, max_channels):
                    return np.concatenate(fields, axis=0).astype(np.float32)
        finally:
            pressure_data.close()
            single_data.close()
        return np.concatenate(fields, axis=0).astype(np.float32)

    def iter_samples(
        self,
        max_samples: int | None = None,
        max_channels: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> Iterator[CanonicalSample]:
        pairs = self.find_pairs()
        if max_samples is not None and max_samples > 0:
            pairs = pairs[:max_samples]
        for pressure, single, timestamp in pairs:
            array = limit_channels(self.read_pair(pressure, single, max_channels=max_channels), max_channels)
            array = center_crop_chw(array, resolution)
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=timestamp,
                kind="scientific_field",
                array=array,
                layout="channel_height_width",
                metadata={
                    "pressure_path": str(pressure),
                    "single_path": str(single),
                    "timestamp": timestamp,
                    "dtype": "float32",
                    "channel_names": era5_channel_names()[: array.shape[0]],
                    "height": int(array.shape[1]),
                    "width": int(array.shape[2]),
                    "channels": int(array.shape[0]),
                },
            )

    def load_sequence(
        self,
        max_samples: int | None = None,
        max_channels: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        samples = list(self.iter_samples(max_samples=max_samples, max_channels=max_channels, resolution=resolution))
        if not samples:
            raise ValueError(f"No ERA5 samples found in {self.data_root}")
        timestamps = [sample.sample_id for sample in samples]
        tchw = np.stack([sample.array for sample in samples], axis=0)
        return np.transpose(tchw, (1, 0, 2, 3)), timestamps


def limit_channels(array: np.ndarray, max_channels: int | None) -> np.ndarray:
    if max_channels is None or max_channels < 0:
        return array
    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")
    if max_channels > array.shape[0]:
        raise ValueError(f"max_channels {max_channels} exceeds available channels {array.shape[0]}")
    return array[:max_channels]


def _has_enough_channels(fields: list[np.ndarray], max_channels: int | None) -> bool:
    return max_channels is not None and max_channels > 0 and len(fields) >= max_channels


def center_crop_chw(array: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None:
        return array
    target_h, target_w = resolution
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"resolution values must be positive, got {resolution}")
    _, height, width = array.shape
    if target_h > height or target_w > width:
        raise ValueError(f"resolution {resolution} exceeds data size {(height, width)}")
    start_h = (height - target_h) // 2
    start_w = (width - target_w) // 2
    return array[:, start_h:start_h + target_h, start_w:start_w + target_w]
