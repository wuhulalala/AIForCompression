from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from compression_pipeline.canonical import CanonicalSample


class ShanghaiXrayAdapter:
    """Reads Shanghai Synchrotron X-ray TIF images as single-channel → 3-channel samples.

    Each TIF is a [H, W] image (int32 or similar). The single channel is duplicated
    to 3 channels for image compression models.
    """

    def __init__(
        self,
        data_root: str | Path,
        subdataset: str = "saxs",
        dataset_id: str = "shanghai_xray",
    ) -> None:
        self.data_root = Path(data_root)
        self.subdataset = subdataset
        self.dataset_id = dataset_id

    def _image_paths(self) -> list[Path]:
        # Data is expected under <subdataset>/SAXS/Data/ or similar
        base = self.data_root
        # Try common layouts
        candidates = [
            base / self.subdataset.upper() / "SAXS" / "Data",
            base / self.subdataset / "Data",
            base / "SAXS" / "Data",
        ]
        for data_dir in candidates:
            if data_dir.is_dir():
                return sorted(
                    p for p in data_dir.glob("*.tif")
                    if p.is_file() and "Label" not in str(p)
                )
        # Fallback: search recursively
        return sorted(
            p for p in base.rglob("*.tif")
            if p.is_file() and "Label" not in str(p)
        )

    def iter_samples(self, max_samples: int = 30) -> Iterator[CanonicalSample]:
        paths = self._image_paths()
        if max_samples > 0:
            paths = paths[:max_samples]

        for path in paths:
            with Image.open(path) as img:
                hw = np.asarray(img, dtype=np.float32)

            if hw.ndim != 2:
                # Some TIFs might be 3-channel already
                if hw.ndim == 3 and hw.shape[2] == 3:
                    chw = np.transpose(hw, (2, 0, 1))
                elif hw.ndim == 3 and hw.shape[0] == 3:
                    chw = hw.astype(np.float32)
                else:
                    raise ValueError(f"Unexpected shape {hw.shape} for {path}")
            else:
                chw = np.stack([hw, hw, hw], axis=0)

            yield CanonicalSample(
                dataset_id=self.dataset_id,
                sample_id=path.stem,
                kind="xray",
                array=chw,
                layout="channel_height_width",
                metadata={
                    "source_path": str(path),
                    "source_format": "tif",
                    "dtype": "float32",
                    "height": int(chw.shape[1]),
                    "width": int(chw.shape[2]),
                    "channels": 3,
                    "subdataset": self.subdataset,
                },
            )
