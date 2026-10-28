"""Model loading (lazy, cached, CPU-only, offline after first download) + shared preprocessing."""
import json, os, threading
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("ANYSCAN_MODELS", ROOT / "models"))
torch.set_num_threads(int(os.environ.get("ANYSCAN_THREADS", max(1, min(8, os.cpu_count() or 1)))))
torch.manual_seed(0)
_lock = threading.Lock()
_cache = {}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(gray, side=224, pad_val=None):
    h, w = gray.shape[:2]
    s = side / max(h, w)
    r = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    pv = int(np.median(np.concatenate([gray[0], gray[-1]]))) if pad_val is None else pad_val
    out = np.full((side, side) + gray.shape[2:], pv, gray.dtype)
    y, x = (side - r.shape[0]) // 2, (side - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def to_tensor_rgb(img, side=224):
    """uint8 gray or BGR -> normalised 1x3xSxS tensor (ImageNet stats)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = letterbox(img, side).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


THIRD = MODEL_DIR / "third_party"


def resnet18(num_out=None, pretrained=True):
    local = THIRD / "resnet18-f37072fd.pth"
    if pretrained and local.exists():      # bundled copy -> fully offline
        m = torchvision.models.resnet18(weights=None)
        m.load_state_dict(torch.load(local, map_location="cpu"))
    else:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    if num_out is not None:
        m.fc = torch.nn.Linear(512, num_out)
    return m


def get_backbone():
    """Frozen ImageNet ResNet-18 feature extractor (512-d) used by the modality classifier."""
    with _lock:
        if "backbone" not in _cache:
            m = resnet18(pretrained=True); m.fc = torch.nn.Identity(); m.eval()
            _cache["backbone"] = m
        return _cache["backbone"]


def embed(imgs, side=224):
    m = get_backbone()
    x = torch.cat([to_tensor_rgb(i, side) for i in imgs])
    with torch.no_grad():
        return m(x).numpy()


def load_json(name):
    with open(MODEL_DIR / name) as f:
        return json.load(f)


def get_ich():
    with _lock:
        if "ich" not in _cache:
            ck = torch.load(MODEL_DIR / "ich_resnet18.pt", map_location="cpu", weights_only=False)
            m = resnet18(1, pretrained=False); m.load_state_dict(ck["state_dict"]); m.eval()
            _cache["ich"] = (m, ck["meta"])
        return _cache["ich"]


def get_xrv():
    with _lock:
        if "xrv" not in _cache:
            import torchxrayvision as xrv
            cd = THIRD / "xrv"
            m = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(cd) + os.sep if cd.exists() else None); m.eval()
            _cache["xrv"] = m
        return _cache["xrv"]


def gradcam(model, layer, x, target_fn):
    """Generic Grad-CAM. target_fn(output)->scalar. Returns HxW float map in [0,1] at input size."""
    store = {}

    def fwd(m, i, o):
        c = o.clone()                      # clone: some backbones apply in-place ReLU right after this layer
        c.register_hook(lambda g: store.__setitem__("g", g))
        store["a"] = c
        return c
    h = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            out = model(x.clone().requires_grad_(True))
            model.zero_grad(); target_fn(out).backward()
        a, g = store["a"][0].detach(), store["g"][0]
        cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0)).numpy()
    finally:
        h.remove()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
    return cam / (cam.max() + 1e-8)
