from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F


class ImageGroupCodec(Protocol):
    def roundtrip(self, tensor_bchw: np.ndarray) -> CodecResult: ...


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


class ForwardLikelihoodCodec:
    """Adapter for models evaluated through forward(x) and likelihood-based BPP estimation."""

    def __init__(self, model: Any, device: str, divisor: int = 64) -> None:
        self.model = model
        self.device = device
        self.divisor = divisor

    @torch.no_grad()
    def roundtrip(self, tensor_bchw: np.ndarray) -> CodecResult:
        x = torch.from_numpy(tensor_bchw).to(self.device)
        x_pad, padding = pad_center(x, self.divisor)
        t0 = time.time()
        out = self.model(x_pad)
        synchronize_if_needed(x)
        t1 = time.time()
        x_hat = unpad_center(out["x_hat"], padding).clamp(0, 1)
        bits = estimated_bits(out["likelihoods"], x_hat.shape[-2], x_hat.shape[-1])
        return CodecResult(
            reconstruction=x_hat.detach().cpu().numpy(),
            bitstream_bytes=int(round(bits / 8.0)),
            encode_time=t1 - t0,
            decode_time=0.0,
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


def estimated_bits(likelihoods: Any, height: int, width: int) -> float:
    total = 0.0
    for likelihood in likelihoods.values():
        lk = likelihood
        if lk.shape[-2] >= height and lk.shape[-1] >= width:
            lk = lk[..., :height, :width]
        total += float((-torch.log2(lk.clamp(min=1e-9))).sum().item())
    return total


def synchronize_if_needed(x: torch.Tensor) -> None:
    if x.is_cuda:
        torch.cuda.synchronize()


def pad_replicate(x: torch.Tensor, divisor: int) -> tuple[torch.Tensor, int, int]:
    """Pad with replicate mode to make height/width divisible by divisor."""
    _, _, h, w = x.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


def bitstream_size(bit_stream: Any) -> int:
    if isinstance(bit_stream, (bytes, bytearray)):
        return len(bit_stream)
    if isinstance(bit_stream, (list, tuple)):
        return sum(bitstream_size(item) for item in bit_stream)
    return len(bit_stream)


class DCVCRTCodec:
    """Codec wrapper for DCVC-RT image/intra model (DMCI).
    DMCI expects YCbCr input (BT.709); this codec converts RGB<->YCbCr.
    """

    def __init__(self, model: Any, device: str, divisor: int = 64, qp: int = 0) -> None:
        self.model = model
        self.device = device
        self.divisor = divisor
        self.qp = qp

    @torch.no_grad()
    def roundtrip(self, tensor_bchw: np.ndarray) -> CodecResult:
        x = torch.from_numpy(tensor_bchw).to(self.device)
        # RGB -> YCbCr (BT.709)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  = 0.2126*r + 0.7152*g + 0.0722*b
        cb = -0.1146*r - 0.3854*g + 0.5000*b + 0.5
        cr = 0.5000*r - 0.4542*g - 0.0458*b + 0.5
        x_ycbcr = torch.cat([y, cb, cr], dim=1).clamp(0, 1)

        x_pad, orig_h, orig_w = pad_replicate(x_ycbcr, self.divisor)
        t0 = time.time()
        compressed = self.model.compress(x_pad, self.qp)
        synchronize_if_needed(x)
        t1 = time.time()
        sps = {"height": x_pad.shape[-2], "width": x_pad.shape[-1], "qp": self.qp, "ec_part": 0}
        decompressed = self.model.decompress(compressed["bit_stream"], sps, self.qp)
        synchronize_if_needed(x)
        t2 = time.time()
        x_hat_ycbcr = decompressed["x_hat"].float()[:, :, :orig_h, :orig_w].clamp(0, 1)
        # YCbCr -> RGB
        yh, cbh, crh = x_hat_ycbcr[:, 0:1], x_hat_ycbcr[:, 1:2], x_hat_ycbcr[:, 2:3]
        r_hat = yh + 1.5748*(crh - 0.5)
        g_hat = yh - 0.1873*(cbh - 0.5) - 0.4681*(crh - 0.5)
        b_hat = yh + 1.8556*(cbh - 0.5)
        x_hat_rgb = torch.cat([r_hat, g_hat, b_hat], dim=1).clamp(0, 1)
        return CodecResult(
            reconstruction=x_hat_rgb.detach().cpu().numpy(),
            bitstream_bytes=bitstream_size(compressed["bit_stream"]),
            encode_time=t1 - t0,
            decode_time=t2 - t1,
        )


class DCMVCCodec:
    """Codec wrapper for DCMVC image/intra model (IntraNoAR)."""

    def __init__(self, model: Any, device: str, divisor: int = 64, q_index: int = 0) -> None:
        self.model = model
        self.device = device
        self.divisor = divisor
        self.q_index = q_index

    @torch.no_grad()
    def roundtrip(self, tensor_bchw: np.ndarray) -> CodecResult:
        x = torch.from_numpy(tensor_bchw).to(self.device)
        x_pad, orig_h, orig_w = pad_replicate(x, self.divisor)
        t0 = time.time()
        compressed = self.model.compress(x_pad, True, self.q_index)
        synchronize_if_needed(x)
        t1 = time.time()
        decompressed = self.model.decompress(
            compressed["bit_stream"], x_pad.shape[-2], x_pad.shape[-1], True, self.q_index
        )
        synchronize_if_needed(x)
        t2 = time.time()
        x_hat = decompressed["x_hat"][:, :, :orig_h, :orig_w].clamp(0, 1)
        return CodecResult(
            reconstruction=x_hat.detach().cpu().numpy(),
            bitstream_bytes=bitstream_size(compressed["bit_stream"]),
            encode_time=t1 - t0,
            decode_time=t2 - t1,
        )
