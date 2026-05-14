from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from compression_pipeline.adapters.era5 import center_crop_chw
from compression_pipeline.canonical import CanonicalSample, DatasetManifest


class TomoH5Adapter:
    """Reads a tomography HDF5 file and yields each projection angle as a 1-channel CHW sample.

    When ``group_frames > 1`` consecutive frames are stacked as channels (e.g. group_frames=3
    produces [3, H, W] pseudo-RGB from three neighbouring projection angles).  This improves
    generalisation of image codecs that were trained on natural RGB imagery.
    """

    def __init__(self, data_root: str | Path, dataset_id: str = "tomo", group_frames: int = 1) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = dataset_id
        self.group_frames = group_frames
        if not self.data_root.exists():
            raise FileNotFoundError(f"Tomo H5 file does not exist: {self.data_root}")

    def manifest(self) -> DatasetManifest:
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["exchange/data"]
            n_frames, h, w = data.shape
            theta = f["exchange/theta"][:]
        channels = self.group_frames
        sample_count = n_frames if channels == 1 else n_frames // channels
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_name="Tomography",
            dataset_type="scientific_projection",
            source_format="hdf5",
            canonical_layout="channel_height_width",
            sample_count=sample_count,
            metadata={
                "data_root": str(self.data_root),
                "height": int(h),
                "width": int(w),
                "channels": channels,
                "dtype": "uint16",
                "theta": theta.tolist(),
            },
        )

    def iter_samples(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> Iterator[CanonicalSample]:
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["exchange/data"]
            theta = f["exchange/theta"][:]
            n_frames = data.shape[0]
            if max_samples is not None and max_samples > 0:
                n_frames = min(n_frames, max_samples)

            gf = self.group_frames
            if gf > 1:
                # Yield consecutive frames stacked as channels, e.g. [3, H, W]
                for start in range(0, n_frames - gf + 1, gf):
                    frames = []
                    angles = []
                    for j in range(gf):
                        frame = data[start + j].astype(np.float32)
                        if resolution is not None:
                            frame = center_crop_chw(frame[None, ...], resolution)[0]
                        frames.append(frame)
                        angles.append(float(theta[start + j]))
                    chw = np.stack(frames, axis=0)  # [gf, H, W]
                    yield CanonicalSample(
                        dataset_id=self.dataset_id,
                        sample_id=f"proj_{start:04d}-{start+gf-1:04d}",
                        kind="scientific_field",
                        array=chw,
                        layout="channel_height_width",
                        metadata={
                            "source_path": str(self.data_root),
                            "dtype": "float32",
                            "source_dtype": "uint16",
                            "height": int(chw.shape[1]),
                            "width": int(chw.shape[2]),
                            "channels": gf,
                            "angle_deg": angles,
                            "frame_index": start,
                        },
                    )
                return

            for i in range(n_frames):
                frame = data[i].astype(np.float32)
                # Add channel dimension: [H, W] -> [1, H, W]
                chw = frame[None, ...]
                if resolution is not None:
                    chw = center_crop_chw(chw, resolution)

                yield CanonicalSample(
                    dataset_id=self.dataset_id,
                    sample_id=f"proj_{i:04d}",
                    kind="scientific_field",
                    array=chw,
                    layout="channel_height_width",
                    metadata={
                        "source_path": str(self.data_root),
                        "dtype": "float32",
                        "source_dtype": "uint16",
                        "height": int(chw.shape[1]),
                        "width": int(chw.shape[2]),
                        "channels": 1,
                        "angle_deg": float(theta[i]),
                        "frame_index": i,
                    },
                )

    def load_sequence(
        self,
        max_samples: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load projection frames as a sequence [C, T, H, W] for CAESAR/video models."""
        import h5py
        with h5py.File(self.data_root, "r") as f:
            data = f["exchange/data"]
            theta = f["exchange/theta"][:]
            n_frames = data.shape[0]
            if max_samples is not None and max_samples > 0:
                n_frames = min(n_frames, max_samples)

            arrays = []
            timestamps = []
            for i in range(n_frames):
                frame = data[i].astype(np.float32)
                chw = frame[None, ...]
                if resolution is not None:
                    chw = center_crop_chw(chw, resolution)
                arrays.append(chw)
                # Use angle as pseudo-timestamp
                timestamps.append(f"angle_{theta[i]:.8f}")

            # Stack: [T, C, H, W] -> transpose to [C, T, H, W]
            tchw = np.stack(arrays, axis=0)
            return np.transpose(tchw, (1, 0, 2, 3)), timestamps
