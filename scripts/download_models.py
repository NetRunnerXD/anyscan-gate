"""Fetch third-party pretrained weights into models/third_party/ (only needed if they are missing;
the release zip already bundles them). After this, inference runs fully offline on CPU."""
import os, shutil, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "models" / "third_party"; (T / "xrv").mkdir(parents=True, exist_ok=True)
if not (T / "resnet18-f37072fd.pth").exists():
    torch.hub.download_url_to_file("https://download.pytorch.org/models/resnet18-f37072fd.pth", str(T / "resnet18-f37072fd.pth"))
import torchxrayvision as xrv
xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(T / "xrv") + os.sep)
for f in ("ich_resnet18.pt", "modality_head.npz", "plugin_config.json", "quality_config.json"):
    print(("ok      " if (ROOT / "models" / f).exists() else "MISSING ") + f)
print("third-party weights in", T)
