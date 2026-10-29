"""Chest X-ray: abnormal vs normal.

Backbone: TorchXRayVision DenseNet-121 'densenet121-res224-all' (trained on NIH, PadChest, CheXpert,
MIMIC, Google, OpenI, RSNA pneumonia) -> 18 pathology outputs. Head: logistic regression on those 18
outputs fitted on REAL labelled CXRs (scripts/fit_cxr_head.py), evaluated on held-out Shenzhen/Montgomery
TB sets, NIH, and CheXphoto real smartphone photos. Decision aid only."""
import cv2
import numpy as np
import torch
from ..models import get_xrv, gradcam, load_json
from .base import abstain_gate, colorize


def xrv_input(gray):
    h, w = gray.shape
    s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
    g = cv2.resize(gray[y0:y0 + s, x0:x0 + s], (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return torch.from_numpy((g / 255.0) * 2048 - 1024)[None, None], (x0, y0, s)


class ChestXrayAbnormal:
    name = "cxr_abnormal_xrv_densenet_v1"
    applies_to = ("chest_xray",)
    description = "Chest X-ray: abnormal vs normal (TorchXRayVision DenseNet-121 + head fitted on real labelled CXRs)."

    def __init__(self):
        self._cfg = None

    @property
    def cfg(self):
        if self._cfg is None:
            self._cfg = load_json("plugin_config.json")["chest_xray"]
        return self._cfg

    def features(self, grays):
        m = get_xrv()
        x = torch.cat([xrv_input(g)[0] for g in grays])
        with torch.no_grad():
            return m(x).numpy()

    def prob(self, F):
        c = self.cfg
        z = ((F - np.array(c["mean"])) / np.array(c["scale"])) @ np.array(c["coef"]) + c["intercept"]
        return 1 / (1 + np.exp(-z))

    def run(self, scan_gray, ctx):
        m = get_xrv()
        F = self.features([scan_gray])
        p = float(self.prob(F)[0])
        lo, hi = self.cfg["lo"], self.cfg["hi"]
        abst, reasons, conf = abstain_gate(p, ctx, lo, hi)
        w = torch.tensor(np.array(self.cfg["coef"]) / np.array(self.cfg["scale"]), dtype=torch.float32)
        x, (x0, y0, s) = xrv_input(scan_gray)
        cam = gradcam(m, m.features, x, lambda o: (o[0] * w).sum())
        full = np.zeros(scan_gray.shape, np.float32)
        full[y0:y0 + s, x0:x0 + s] = cv2.resize(cam, (s, s)) * (p >= lo)
        overlay = colorize(cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR), full)
        top = sorted(zip(m.pathologies, F[0]), key=lambda t: -t[1])[:5]
        positive = p >= hi
        return {
            "plugin": self.name, "status": "abstain" if abst else "result",
            "finding": None if abst else ("abnormal chest X-ray" + (f" (top signals: {', '.join(t[0] for t in top[:3])})" if positive else "")
                                          if positive else "no abnormality detected"),
            "p_abnormal": round(p, 4), "confidence": conf, "abstain_reasons": reasons,
            "positive": bool(positive and not abst), "severity": "moderate" if positive else "low",
            "operating_points": {"lo": lo, "hi": hi, **self.cfg.get("val_metrics", {})},
            "features": {"top_pathology_scores": {k: round(float(v), 3) for k, v in top}},
            "overlay": overlay,
            "notes": "Pathology scores are TorchXRayVision outputs (0.5 ~ that label's operating point). "
                     "Abnormal-vs-normal head and thresholds fitted on real public CXRs; see docs/EVAL.md.",
        }
