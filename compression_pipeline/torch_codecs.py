from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CodecResult:
    reconstruction: np.ndarray
    bitstream_bytes: int
    encode_time: float
    decode_time: float


class CompressAILikeCodec:
    """Adapter for models exposing compress(x) and decompress(strings, shape)."""

    def __init__(self, model: Any, device: str, divisor: int = 64) -> None:
        self.model = model
        self.device = device
        self.divisor = divisor

    @torch.no_grad()
    def roundtrip(self, tensor_bchw: np.ndarray) -> CodecResult:
        x = torch.from_numpy(tensor_bchw).to(self.device)
        x_pad, padding = pad_center(x, self.divisor)
        t0 = time.time()
        compressed = self.model.compress(x_pad)
        synchronize_if_needed(x)
        t1 = time.time()
        decompressed = self.model.decompress(compressed["strings"], compressed["shape"])
        synchronize_if_needed(x)
        t2 = time.time()
        x_hat = unpad_center(decompressed["x_hat"], padding).clamp(0, 1)
        return CodecResult(
            reconstruction=x_hat.detach().cpu().numpy(),
            bitstream_bytes=nested_bytes(compressed["strings"]),
            encode_time=t1 - t0,
            decode_time=t2 - t1,
        )


def pad_center(x: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = x.shape
    new_h = (height + divisor - 1) // divisor * divisor
    new_w = (width + divisor - 1) // divisor * divisor
    left = (new_w - width) // 2
    right = new_w - width - left
    top = (new_h - height) // 2
    bottom = new_h - height - top
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0), (left, right, top, bottom)


def unpad_center(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    return F.pad(x, (-left, -right, -top, -bottom), mode="constant", value=0)


def nested_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(nested_bytes(item) for item in value)
    return len(value)


def synchronize_if_needed(x: torch.Tensor) -> None:
    if x.is_cuda:
        torch.cuda.synchronize()

