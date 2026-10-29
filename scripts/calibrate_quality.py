"""Fit quality-gate thresholds on the PhotoShift-IN *calibration* split (simulated phone captures of
REAL training-pool images with known degradations). Test split is scored in scripts/evaluate.py.
  python scripts/calibrate_quality.py [n_images]
"""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyscan.photosim import render
from anyscan.pipeline import frontend
from anyscan.quality import measure
from anyscan.models import MODEL_DIR
from train_modality import pools  # noqa: E402  (scripts/ on path)

RULES = {  # metric, direction (True = retake when metric > thr), degradation kind it targets
    "blur_retake": ("blur_laplacian_var", False, "blur"),
    "glare_retake": ("glare_fraction", True, "glare"),
    "block_retake": ("jpeg_blockiness", True, "jpeg"),
    "min_side_retake": ("min_side", False, "lowres"),
    "dark_retake": ("mean_brightness", False, "dark"),
}


def sample_set(split, n, seed):
    rng = np.random.default_rng(seed)
    P = pools(split)
    srcs = []
    for c in ("chest_xray", "head_ct", "brain_mri", "other_xray"):
        f = list(P[c]); rng.shuffle(f); srcs += [(c, x) for x in f[: n // 4]]
    rows = []
    for i, (c, f) in enumerate(srcs):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g.shape[0] < 400:
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        kind = ["clean", "blur", "glare", "jpeg", "lowres", "dark", "mixed"][i % 7]
        sev = "clean" if kind == "clean" else (None if kind == "mixed" else {kind: rng.choice(["mild", "severe"])})
        ph, info = render(g, rng, severity=sev)
        scan, quad, finfo, geom = frontend(ph)
        m = measure(ph, cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY))
        rows.append({"cls": c, "src": f, "info": info, "metrics": m, "photo": ph})
    return rows


def fit(rows):
    th = {}
    for name, (metric, gt, kind) in RULES.items():
        pos = [r["metrics"][metric] for r in rows if r["info"]["degradations"].get(kind) == "severe"]
        neg = [r["metrics"][metric] for r in rows if r["info"]["expected_quality"] == "PASS"]
        cands = np.unique(np.percentile(pos + neg, np.linspace(1, 99, 197)))
        best = (-1, None)
        for t in cands:
            tp = np.mean([(v > t) if gt else (v < t) for v in pos]); fp = np.mean([(v > t) if gt else (v < t) for v in neg])
            score = tp - 2 * fp          # penalise false RETAKEs on good captures more than misses
            best = max(best, (score, float(t)))
        th[name] = best[1]
    return th


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 280
    rows = sample_set("train", n, 0)
    th = fit(rows)
    # native CT/MRI matrices are 512 px: never reject an image just for being a native 512-px export
    th["min_side_retake"] = min(th["min_side_retake"], 480.0)
    json.dump({"thresholds": th, "fitted_on": f"PhotoShift-IN calibration split, {len(rows)} simulated captures of real train-pool images"},
              open(MODEL_DIR / "quality_config.json", "w"), indent=1)
    print(json.dumps(th, indent=1))
