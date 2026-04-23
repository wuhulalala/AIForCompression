from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch


@dataclass(frozen=True)
class ModelJob:
    model_name: str
    model_id: str
    metric: str
    checkpoint: str | None
    divisor: int
    loader: Callable[[str], torch.nn.Module]


def image_model_jobs(project_root: str | Path, names: set[str] | None = None) -> Iterator[ModelJob]:
    root = Path(project_root)
    ckpt_base = root / "checkpoints"
    allowed = {"DCAE", "WeConvene", "LIC_TCM", "LIC-HPCM", "RwkvCompress", "CRA5"} if names is None else names

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


def load_dcae(project_root: Path, checkpoint_path: str, device: str):
    clear_model_modules()
    sys.path.insert(0, str(project_root / "models" / "DCAE"))
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


def clear_model_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [
        path for path in sys.path
        if "/AIForCompression/models/" not in path
    ]
