from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch


@dataclass(frozen=True)
class ModelJob:
    model_name: str
    model_id: str
    metric: str
    checkpoint: str | None
    divisor: int
    loader: Callable[[str], torch.nn.Module]
    codec_cls: Any = None
    codec_kwargs: dict[str, Any] = field(default_factory=dict)


def image_model_jobs(project_root: str | Path, names: set[str] | None = None) -> Iterator[ModelJob]:
    root = Path(project_root)
    ckpt_base = root / "checkpoints"
    allowed = {"DCAE", "WeConvene", "LIC_TCM", "LIC-HPCM", "RwkvCompress", "CRA5", "DCVC-RT", "DCMVC"} if names is None else names

    if "CRA5" in allowed:
        yield ModelJob("CRA5", "CRA5_vaeformer_q268", "mse", None, 1, lambda device: load_cra5(root, device))

    if "DCAE" in allowed:
        for ckpt in sorted((ckpt_base / "dcae").glob("*.pth.tar")):
            yield ModelJob("DCAE", f"DCAE_{ckpt.stem}", "mse", str(ckpt), 128, lambda device, c=str(ckpt): load_dcae(root, c, device))

    if "WeConvene" in allowed:
        for ckpt in sorted((ckpt_base / "weconvene").glob("*.pth.tar")):
            yield ModelJob("WeConvene", f"WeConvene_{ckpt.stem}", "mse", str(ckpt), 128, lambda device, c=str(ckpt): load_weconvene(root, c, device))

    if "LIC_TCM" in allowed:
        for ckpt in sorted((ckpt_base / "lictcm").glob("*.pth.tar")):
            n = lictcm_n_for_checkpoint(ckpt)
            model_id = f"LICTCM_{checkpoint_id(ckpt)}"
            if n == 128:
                model_id = f"{model_id}_large"
            yield ModelJob("LIC_TCM", model_id, "mse", str(ckpt), 128, lambda device, c=str(ckpt), n=n: load_lictcm(root, c, device, n=n))

    if "LIC-HPCM" in allowed:
        for variant in ("base", "large"):
            for ckpt in sorted((ckpt_base / "lic-hpcm" / f"hpcm-{variant}" / "mse").glob("*.pth*")):
                yield ModelJob("LIC-HPCM", f"LIC-HPCM-{variant}_{ckpt.stem}", "mse", str(ckpt), 256, lambda device, c=str(ckpt), v=variant: load_hpcm(root, c, v, device))

    if "RwkvCompress" in allowed:
        for ckpt in sorted((ckpt_base / "rwkvcompress" / "mse").glob("*.pth*")):
            yield ModelJob("RwkvCompress", f"RwkvCompress_{ckpt.stem}", "mse", str(ckpt), 128, lambda device, c=str(ckpt): load_rwkv(root, c, device))

    if "DCVC-RT" in allowed:
        from compression_pipeline.torch_codecs import DCVCRTCodec

        ckpt_path = str(ckpt_base / "dcvc-rt" / "cvpr2025_image.pth.tar")
        for qp in (0, 21, 42, 63):
            yield ModelJob(
                "DCVC-RT", f"DCVC_RT_Intra_q{qp}", "mse", ckpt_path, 64,
                lambda device, c=ckpt_path: load_dcvc_rt(root, c, device),
                codec_cls=DCVCRTCodec, codec_kwargs={"qp": qp},
            )

    if "DCMVC" in allowed:
        from compression_pipeline.torch_codecs import DCMVCCodec

        ckpt_path = str(ckpt_base / "dcmvc" / "cvpr2023_image_psnr.pth.tar")
        for q_index in range(4):
            yield ModelJob(
                "DCMVC", f"DCMVC_Intra_q{q_index}", "mse", ckpt_path, 64,
                lambda device, c=ckpt_path: load_dcmvc(root, c, device),
                codec_cls=DCMVCCodec, codec_kwargs={"q_index": q_index},
            )


def load_dcae(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "DCAE"))
    # DCAE has an autoregressive slice architecture where compress() and
    # decompress() independently recompute hyper-latent parameters.  cuDNN
    # non-determinism introduces ~1e-4 differences across calls, which
    # are enough to shift CDF indexes and cause incorrect RANS symbol
    # decoding.  The error cascades exponentially across slices.
    # Disabling cuDNN matches the original DCAE eval.py and guarantees
    # bit-identical latent parameters between encode and decode.
    torch.backends.cudnn.enabled = False
    from models import DCAE

    net = DCAE().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def load_cra5(project_root: Path, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "CRA5"))
    from cra5.models.compressai.zoo import vaeformer_pretrained

    net = vaeformer_pretrained(quality=268, pretrained=True).eval().to(device)
    net.update()
    return net


def load_weconvene(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "WeConvene"))
    from model import TCM_residual_wave_two_entropy_modified_y_downsample_8

    net = TCM_residual_wave_two_entropy_modified_y_downsample_8(
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=128,
        M=320,
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def load_lictcm(project_root: Path, checkpoint_path: str, device: str, n: int = 64):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "LIC_TCM"))
    from models import TCM

    net = TCM(
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=n,
        M=320,
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    net.update()
    return net


def lictcm_n_for_checkpoint(checkpoint_path: str | Path) -> int:
    name = Path(checkpoint_path).name
    if name == "mse_lambda_0.05.pth.tar":
        return 128
    return 64


def checkpoint_id(checkpoint_path: str | Path) -> str:
    name = Path(checkpoint_path).name
    for suffix in (".pth.tar", ".pth"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_hpcm(project_root: Path, checkpoint_path: str, variant: str, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "LIC-HPCM"))
    model_name = "HPCM_Base" if variant == "base" else "HPCM_Large"
    net_cls = importlib.import_module(f"src.models.{model_name}").HPCM
    model = net_cls()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.update(torch.exp(torch.linspace(torch.log(torch.tensor(0.12)), torch.log(torch.tensor(64.0)), 60)))
    return model.to(device).eval()


def load_rwkv(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "RwkvCompress"))
    from eval import load_checkpoint
    from models import LALIC

    model = load_checkpoint(LALIC, checkpoint_path)
    model.update()
    return model.to(device).eval()


def load_dcvc_rt(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    dcvc_root = project_root / "models" / "DCVC"
    cpp_root = dcvc_root / "src" / "cpp"
    ext_root = dcvc_root / "src" / "layers" / "extensions" / "inference"
    sys.path.insert(0, str(dcvc_root))
    sys.path.insert(0, str(cpp_root))
    sys.path.insert(0, str(ext_root))
    from src.models.image_model import DMCI

    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "net" in ckpt:
        ckpt = ckpt["net"]
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")

    model = DMCI().to(device).eval()
    model.load_state_dict(ckpt)
    model.update()
    return model


def load_dcmvc(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    dcmvc_root = project_root / "models" / "DCMVC"
    sys.path.insert(0, str(dcmvc_root))
    from src.models.image_model import IntraNoAR
    from src.utils.stream_helper import get_state_dict

    model = IntraNoAR(inplace=True)
    model.load_state_dict(get_state_dict(checkpoint_path))
    model = model.to(device).eval()
    model.update(force=True)
    return model


def clear_model_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [
        path for path in sys.path
        if "/AIForCompression/models/" not in path
    ]
