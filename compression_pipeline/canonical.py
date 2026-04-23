from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CanonicalSample:
    dataset_id: str
    sample_id: str
    kind: str
    array: np.ndarray
    layout: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_layout(self, expected: str) -> None:
        if self.layout != expected:
            raise ValueError(f"{self.sample_id} uses layout {self.layout!r}, expected {expected!r}")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    dataset_name: str
    dataset_type: str
    source_format: str
    canonical_layout: str
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

