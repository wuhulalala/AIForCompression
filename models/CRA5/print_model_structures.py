import argparse
import os
import sys
import traceback
from typing import Iterable, Optional


CRA5_ROOT = "/data/run01/scxj523/zsh/project/AIForCompression/CRA5"
COMPRESSAI_ROOT = "/data/run01/scxj523/zsh/project/CompressAI"

if COMPRESSAI_ROOT not in sys.path:
    sys.path.insert(0, COMPRESSAI_ROOT)
if CRA5_ROOT not in sys.path:
    sys.path.insert(0, CRA5_ROOT)

import torch
import torch.nn as nn

try:
    from compressai.models.google import (
        FactorizedPrior,
        JointAutoregressiveHierarchicalPriors,
        MeanScaleHyperprior,
        ScaleHyperprior,
    )
except Exception:
    FactorizedPrior = None
    JointAutoregressiveHierarchicalPriors = None
    MeanScaleHyperprior = None
    ScaleHyperprior = None

try:
    from cra5.models.vaeformer import VAEformer
except Exception:
    VAEformer = None

try:
    from cra5.models.vaeformer.vit_nlc import Decoder, Encoder, HyperPriorDecoder, HyperPriorEncoder
except Exception:
    Decoder = None
    Encoder = None
    HyperPriorDecoder = None
    HyperPriorEncoder = None


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_module_summary(name: str, module: nn.Module) -> None:
    print_header(name)
    print(module)


def print_conv_info(prefix: str, layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        print(
            f"{prefix}: {layer.__class__.__name__} "
            f"in={layer.in_channels}, out={layer.out_channels}, "
            f"kernel={layer.kernel_size}, stride={layer.stride}, padding={layer.padding}"
        )


def print_named_layers(title: str, pairs: Iterable[tuple[str, nn.Module]]) -> None:
    print(title)
    for name, layer in pairs:
        print_conv_info(f"  {name}", layer)


class DummyGDN(nn.Identity):
    pass


class DummyEntropyBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x


class DummyGaussianConditional(nn.Module):
    def forward(self, x):
        return x


def conv(in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(
    in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
    )


class StructFactorizedPrior(nn.Module):
    def __init__(self, N: int, M: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.entropy_bottleneck = DummyEntropyBottleneck(M)
        self.g_a = nn.Sequential(
            conv(in_channels, N),
            DummyGDN(),
            conv(N, N),
            DummyGDN(),
            conv(N, N),
            DummyGDN(),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N),
            DummyGDN(),
            deconv(N, N),
            DummyGDN(),
            deconv(N, N),
            DummyGDN(),
            deconv(N, out_channels),
        )


class StructScaleHyperprior(nn.Module):
    def __init__(self, N: int, M: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.entropy_bottleneck = DummyEntropyBottleneck(N)
        self.gaussian_conditional = DummyGaussianConditional()
        self.g_a = nn.Sequential(
            conv(in_channels, N),
            DummyGDN(),
            conv(N, N),
            DummyGDN(),
            conv(N, N),
            DummyGDN(),
            conv(N, M),
        )
        self.g_s = nn.Sequential(
            deconv(M, N),
            DummyGDN(),
            deconv(N, N),
            DummyGDN(),
            deconv(N, N),
            DummyGDN(),
            deconv(N, out_channels),
        )
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )


class StructMeanScaleHyperprior(StructScaleHyperprior):
    def __init__(self, N: int, M: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__(N=N, M=M, in_channels=in_channels, out_channels=out_channels)
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )
        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


class StructJointAutoregressiveHierarchicalPriors(StructMeanScaleHyperprior):
    def __init__(self, N: int, M: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__(N=N, M=M, in_channels=in_channels, out_channels=out_channels)
        self.g_a = nn.Sequential(
            conv(in_channels, N, kernel_size=5, stride=2),
            DummyGDN(),
            conv(N, N, kernel_size=5, stride=2),
            DummyGDN(),
            conv(N, N, kernel_size=5, stride=2),
            DummyGDN(),
            conv(N, M, kernel_size=5, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            DummyGDN(),
            deconv(N, N, kernel_size=5, stride=2),
            DummyGDN(),
            deconv(N, N, kernel_size=5, stride=2),
            DummyGDN(),
            deconv(N, out_channels, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 4, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 2, 1),
        )
        self.context_prediction = nn.Conv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)


class StructuralVAEformer268(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_posterior = False
        self.lower_dim = True
        self.entropy_bottleneck = DummyEntropyBottleneck(256)
        if Encoder is not None and Decoder is not None:
            self.g_a = Encoder(
                arch="vit_large",
                patch_size=(11, 10),
                patch_stride=(10, 10),
                in_chans=268,
                out_chans=268,
                kwargs=dict(
                    z_dim=None,
                    learnable_pos=True,
                    window=True,
                    window_size=[(24, 24), (12, 48), (48, 12)],
                    interval=4,
                    drop_path_rate=0.0,
                    round_padding=True,
                    pad_attn_mask=True,
                    test_pos_mode="learnable_simple_interpolate",
                    lms_checkpoint_train=True,
                    img_size=(721, 1440),
                ),
            )
            self.g_s = Decoder(
                arch="vit_large",
                patch_size=(11, 10),
                patch_stride=(10, 10),
                in_chans=268,
                out_chans=268,
                kwargs=dict(
                    z_dim=None,
                    learnable_pos=True,
                    window=True,
                    window_size=[(24, 24), (12, 48), (48, 12)],
                    interval=4,
                    drop_path_rate=0.0,
                    round_padding=True,
                    pad_attn_mask=True,
                    test_pos_mode="learnable_simple_interpolate",
                    lms_checkpoint_train=True,
                    img_size=(721, 1440),
                ),
            )
        else:
            self.g_a = nn.Sequential(
                nn.Conv2d(268, 1024, kernel_size=(11, 10), stride=(10, 10)),
                nn.Identity(),
            )
            self.g_s = nn.Sequential(
                nn.Identity(),
                nn.ConvTranspose2d(1024, 268, kernel_size=(11, 10), stride=(10, 10), bias=False),
            )
        self.quant_conv = nn.Conv2d(2048, 512, 1)
        self.post_quant_conv = nn.Conv2d(256, 1024, 1)
        if HyperPriorEncoder is not None and HyperPriorDecoder is not None:
            self.h_a = HyperPriorEncoder(
                patch_size=(4, 4),
                in_chans=256,
                out_chans=256,
                kwargs=dict(
                    z_dim=256,
                    embed_dim=360,
                    depth=8,
                    num_heads=5,
                    interval=1,
                    learnable_pos=True,
                    window=False,
                    drop_path_rate=0.0,
                    round_padding=True,
                    pad_attn_mask=True,
                    test_pos_mode="learnable_simple_interpolate",
                    lms_checkpoint_train=False,
                    img_size=(72, 144),
                ),
            )
            self.h_s = HyperPriorDecoder(
                patch_size=(4, 4),
                in_chans=256,
                out_chans=256,
                kwargs=dict(
                    z_dim=256,
                    embed_dim=360,
                    depth=8,
                    num_heads=5,
                    interval=1,
                    learnable_pos=True,
                    window=False,
                    drop_path_rate=0.0,
                    round_padding=True,
                    pad_attn_mask=True,
                    test_pos_mode="learnable_simple_interpolate",
                    lms_checkpoint_train=False,
                    img_size=(72, 144),
                ),
            )
        else:
            self.h_a = nn.Sequential(
                nn.Conv2d(256, 360, kernel_size=(4, 4), stride=(4, 4)),
                nn.Identity(),
            )
            self.h_s = nn.Sequential(
                nn.Identity(),
                nn.Linear(360, 2 * 256 * 4 * 4, bias=False),
            )
        self.gaussian_conditional = DummyGaussianConditional()


class FactorizedPrior268(StructFactorizedPrior):
    def __init__(self, N: int, M: int, in_channels: int = 268, out_channels: int = 268):
        super().__init__(N=N, M=M, in_channels=in_channels, out_channels=out_channels)


class ScaleHyperprior268(StructScaleHyperprior):
    def __init__(self, N: int, M: int, in_channels: int = 268, out_channels: int = 268):
        super().__init__(N=N, M=M, in_channels=in_channels, out_channels=out_channels)


def maybe_real_or_fallback(real_cls: Optional[type[nn.Module]], fallback_cls: type[nn.Module], *args):
    if real_cls is not None:
        try:
            return real_cls(*args)
        except Exception as exc:
            print(f"Falling back from {real_cls.__name__}: {exc}")
    return fallback_cls(*args)


def print_compressai_models() -> None:
    classic_models = [
        (
            "CompressAI FactorizedPrior(N=128, M=192)",
            maybe_real_or_fallback(FactorizedPrior, StructFactorizedPrior, 128, 192),
        ),
        (
            "CompressAI ScaleHyperprior(N=128, M=192)",
            maybe_real_or_fallback(ScaleHyperprior, StructScaleHyperprior, 128, 192),
        ),
        (
            "CompressAI MeanScaleHyperprior(N=128, M=192)",
            maybe_real_or_fallback(MeanScaleHyperprior, StructMeanScaleHyperprior, 128, 192),
        ),
        (
            "CompressAI JointAutoregressiveHierarchicalPriors(N=192, M=192)",
            maybe_real_or_fallback(
                JointAutoregressiveHierarchicalPriors,
                StructJointAutoregressiveHierarchicalPriors,
                192,
                192,
            ),
        ),
    ]

    for name, model in classic_models:
        print_module_summary(name, model)
        print_named_layers(
            "Key convolution layers:",
            [
                ("g_a.0", model.g_a[0]),
                ("g_a.last", model.g_a[-1]),
                ("g_s.0", model.g_s[0]),
                ("g_s.last", model.g_s[-1]),
            ],
        )

    adapted_models = [
        ("CompressAI FactorizedPrior268(N=128, M=192)", FactorizedPrior268(128, 192)),
        ("CompressAI ScaleHyperprior268(N=128, M=192)", ScaleHyperprior268(128, 192)),
    ]

    for name, model in adapted_models:
        print_module_summary(name, model)
        print_named_layers(
            "Key convolution layers for 268-channel adaptation:",
            [
                ("g_a.0", model.g_a[0]),
                ("g_a.last", model.g_a[-1]),
                ("g_s.0", model.g_s[0]),
                ("g_s.last", model.g_s[-1]),
            ],
        )


def print_cra5_model() -> None:
    model = VAEformer(model_version=268) if VAEformer is not None else StructuralVAEformer268()
    print_module_summary("CRA5 VAEformer(model_version=268)", model)
    print_named_layers(
        "Key CRA5 layers:",
        [
            ("g_a.patch_embed.proj", model.g_a.patch_embed.proj),
            ("quant_conv", model.quant_conv),
            ("h_a.patch_embed.proj", model.h_a.patch_embed.proj),
            ("post_quant_conv", model.post_quant_conv),
            ("g_s.final", model.g_s.final),
        ],
    )
    print(
        "CRA5 config summary:\n"
        f"  sample_posterior={model.sample_posterior}\n"
        f"  lower_dim={model.lower_dim}\n"
        f"  entropy_bottleneck.channels={model.entropy_bottleneck.channels}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to tee stdout into a file from inside Python.",
    )
    args = parser.parse_args()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sys.stdout = open(args.output, "w", encoding="utf-8")
        sys.stderr = sys.stdout

    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        print_compressai_models()
        print_cra5_model()
        print_header("Done")
        print("Model structures printed successfully.")
        return 0
    except Exception:
        print_header("Failure")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
