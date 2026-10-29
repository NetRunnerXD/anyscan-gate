"""Head CT: intracranial haemorrhage (any subtype) present/absent.

ResNet-18 fine-tuned on REAL RSNA ICH slices (brain window, as printed on film / shown on screen),
Platt-calibrated on held-out RSNA patients, externally tested on CQ500 (India). Works on a single
slice or on a film montage (grid of slices is split automatically; image-level p = max over tiles).
Decision aid only - not clinically validated."""
import cv2
import numpy as np
import torch
from ..models import get_ich, to_tensor_rgb, gradcam, load_json, letterbox
from .base import abstain_gate, colorize


def _runs(mask, min_len):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def _grid_split(gray):
    """Film montages: slices are separated by dark gutters -> use row/column intensity profiles."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    bg = np.percentile(g, 5)
    content = g > bg + 25
    cols = _runs(content.mean(0) > 0.02, max(8, W // 40))
    rows = _runs(content.mean(1) > 0.02, max(8, H // 40))
    if len(cols) < 2 or len(rows) < 2:
        return None
    widths = np.array([c[1] - c[0] for c in cols]); heights = np.array([r[1] - r[0] for r in rows])
    # a montage has near-equal cell sizes; drop outlier runs (e.g. a bright table edge)
    cols = [c for c, w in zip(cols, widths) if 0.5 * np.median(widths) < w < 1.6 * np.median(widths)]
    rows = [r for r, h in zip(rows, heights) if 0.5 * np.median(heights) < h < 1.6 * np.median(heights)]
    if len(cols) < 2 or len(rows) < 2:
        return None
    boxes = []
    for y0, y1 in rows:
        for x0, x1 in cols:
            if content[y0:y1, x0:x1].mean() > 0.15:
                w, h = x1 - x0, y1 - y0; s = int(max(w, h) * 1.15)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                X0, Y0 = max(0, cx - s // 2), max(0, cy - s // 2)
                boxes.append((int(X0), int(Y0), int(min(W - X0, s)), int(min(H - Y0, s))))
    return boxes if len(boxes) >= 2 else None


def split_montage(gray):
    """Return list of (x, y, w, h) boxes, one per head slice on a film grid (or the whole image)."""
    grid = _grid_split(gray)
    if grid:
        return grid
    H, W = gray.shape
    m = (cv2.GaussianBlur(gray, (5, 5), 0) > np.percentile(gray, 5) + 25).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cand = [st[i, :4] for i in range(1, n)
            if st[i, 4] > 0.004 * H * W and 0.55 < st[i, 2] / max(1, st[i, 3]) < 1.8]
    if len(cand) >= 2:
        areas = np.array([c[2] * c[3] for c in cand]); med = np.median(areas)
        cand = [c for c, a in zip(cand, areas) if med / 3 < a < med * 3]
    if len(cand) < 2:
        return [(0, 0, W, H)]
    out = []
    for x, y, w, h in cand:
        p = int(0.12 * max(w, h)); s = max(w, h) + 2 * p
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
        out.append((int(x0), int(y0), int(min(W - x0, s)), int(min(H - y0, s))))
    return sorted(out, key=lambda b: (b[1] // max(1, H // 10), b[0]))


class HeadCTICH:
    name = "head_ct_ich_resnet18_rsna_v1"
    applies_to = ("head_ct",)
    description = "Head CT: intracranial haemorrhage present/absent (ResNet-18, trained on RSNA ICH, external test CQ500)."

    def __init__(self):
        self._cfg = None

    @property
    def cfg(self):
        if self._cfg is None:
            self._cfg = load_json("plugin_config.json")["head_ct"]
        return self._cfg

    def slice_probs(self, tiles):
        model, meta = get_ich()
        a, b = meta["platt"]
        x = torch.cat([to_tensor_rgb(t) for t in tiles])
        with torch.no_grad():
            lg = model(x).squeeze(1).numpy()
        return 1 / (1 + np.exp(-(a * lg + b)))

    def run(self, scan_gray, ctx):
        boxes = split_montage(scan_gray)
        tiles = [scan_gray[y:y + h, x:x + w] for x, y, w, h in boxes]
        ps = self.slice_probs(tiles)
        k = int(np.argmax(ps)); p = float(ps[k])
        lo, hi = self.cfg["lo"], self.cfg["hi"]
        abst, reasons, conf = abstain_gate(p, ctx, lo, hi)
        # Grad-CAM on the most suspicious tile, pasted back into the full scan
        model, _ = get_ich()
        cam = gradcam(model, model.layer4, to_tensor_rgb(tiles[k]), lambda o: o[0, 0])
        x, y, w, h = boxes[k]
        side = max(w, h); cam_t = cv2.resize(cam, (side, side))
        oy, ox = (side - h) // 2, (side - w) // 2
        full = np.zeros(scan_gray.shape, np.float32)
        full[y:y + h, x:x + w] = cam_t[oy:oy + h, ox:ox + w] * (p >= lo)
        overlay = colorize(cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR), full)
        for i, (bx, by, bw, bh) in enumerate(boxes if len(boxes) > 1 else []):
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255) if i == k else (0, 255, 0), 2)
        positive = p >= hi
        return {
            "plugin": self.name, "status": "abstain" if abst else "result",
            "finding": None if abst else ("possible intracranial haemorrhage" if positive else "no haemorrhage detected"),
            "p_abnormal": round(p, 4), "confidence": conf, "abstain_reasons": reasons,
            "positive": bool(positive and not abst), "severity": "high" if positive else "low",
            "operating_points": {"lo": lo, "hi": hi, **self.cfg.get("val_metrics", {})},
            "features": {"n_slices_on_image": len(boxes), "per_slice_p": [round(float(v), 3) for v in ps]},
            "overlay": overlay,
            "notes": "Trained on RSNA ICH (brain-window slices); externally evaluated on CQ500 - see docs/EVAL.md. "
                     "Misses are possible for tiny/thin bleeds and on slices not shown on the photo.",
        }
