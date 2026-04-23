"""Download all compressai pretrained weights to local checkpoints dir."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CRA5'))

from torch.hub import download_url_to_file
from cra5.models.compressai.zoo.image import model_urls

CKPT_BASE = '/data/run01/scxj523/zsh/project/AIForCompression/checkpoints'

for arch, metric_dict in model_urls.items():
    for metric, quality_dict in metric_dict.items():
        for quality, url in quality_dict.items():
            if isinstance(url, str) and url.startswith('http'):
                model_dir = os.path.join(CKPT_BASE, arch, metric)
                os.makedirs(model_dir, exist_ok=True)
                filename = os.path.basename(url)
                dest = os.path.join(model_dir, filename)
                if os.path.exists(dest):
                    print(f"Skip (exists): {arch}/{metric}/{filename}")
                    continue
                print(f"Downloading: {arch}/{metric}/q{quality} -> {dest}")
                try:
                    download_url_to_file(url, dest, progress=True)
                except Exception as e:
                    print(f"  FAILED: {e}")

print("Done.")
