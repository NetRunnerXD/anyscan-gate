"""Capture-quality scoring -> PASS / RETAKE with human-readable reasons."""
import cv2
import numpy as np
from .utils import resize_max

TH = {
    "blur_retake": 60.0,       # Laplacian variance on 768px normalised scan
    "blur_warn": 150.0,
    "glare_retake": 0.012,     # fraction of scan covered by solid saturated blobs
    "glare_warn": 0.004,
    "tilt_retake": 20.0,       # degrees
    "tilt_warn": 8.0,
    "persp_retake": 0.25,
    "block_retake": 1.9,       # JPEG 8x8 blockiness ratio
    "block_warn": 1.45,
    "min_side_retake": 480,
    "min_side_warn": 800,
    "dark_retake": 25, "bright_retake": 225,
    "contrast_retake": 25,
}


def blur_score(gray):
    g, _ = resize_max(gray, 768)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def glare_score(gray):
    """Glare = large *solid* near-saturated blobs with soft falloff.
    Bone (e.g. CT skull ring) is saturated but thin -> low solidity, ignored."""
    g, _ = resize_max(gray, 768)
    m = (g >= 245).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    total = 0
    worst = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.001 * g.size:
            continue
        cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(cnt[0])) or 1
        solidity = a / hull
        x, y, w, h = stats[i, :4]
        thickness = a / max(cv2.arcLength(cnt[0], True), 1)
        if solidity > 0.6 and thickness > 4:
            total += a
            if worst is None or a > worst[2] * worst[3]:
                worst = (x, y, w, h)
    return float(total / g.size), worst


def blockiness(gray):
    """Ratio of luminance jumps on 8x8 JPEG grid lines vs elsewhere (1.0 = none)."""
    g = gray.astype(np.float32)
    h, w = g.shape
    h8, w8 = h - h % 8, w - w % 8
    g = g[:h8, :w8]
    dx = np.abs(np.diff(g, axis=1)).mean(0)
    dy = np.abs(np.diff(g, axis=0)).mean(1)
    def ratio(d):
        idx = np.arange(len(d))
        on = d[(idx % 8) == 7].mean()
        off = d[(idx % 8) != 7].mean() + 1e-6
        return on / off
    return float((ratio(dx) + ratio(dy)) / 2)


def moire_score(gray):
    """Screen photos: strong periodic high-frequency peaks in the spectrum."""
    g, _ = resize_max(gray, 512)
    g = cv2.resize(g, (512, 512)).astype(np.float32)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[-256:256, -256:256]
    r = np.hypot(yy, xx)
    hf = f[(r > 60) & (r < 240)]
    return float(np.percentile(hf, 99.99) / (np.median(hf) + 1e-6))


def _load_calibrated():
    """Thresholds re-fitted on PhotoShift-IN calibration split (scripts/calibrate_quality.py)."""
    import json
    from .models import MODEL_DIR
    f = MODEL_DIR / "quality_config.json"
    if f.exists():
        TH.update(json.load(open(f))["thresholds"])


_load_calibrated()


def measure(original_bgr, scan_gray):
    m = {}
    m["blur_laplacian_var"] = round(blur_score(scan_gray), 1)
    m["glare_fraction"], _ = glare_score(scan_gray)
    m["jpeg_blockiness"] = round(blockiness(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)), 3)
    m["min_side"] = int(min(original_bgr.shape[:2]))
    m["mean_brightness"] = float(scan_gray.mean())
    m["contrast_std"] = float(scan_gray.std())
    return m


def assess(original_bgr, scan_gray, geom, film_found, digital=False):
    """original_bgr: raw upload; scan_gray: rectified scan; geom: quad geometry or None."""
    reasons, warnings = [], []
    if digital:
        return {"verdict": "PASS", "score": 100.0, "reasons": [], "warnings": ["Digital DICOM input: capture checks not applicable."],
                "metrics": {"resolution": f"{original_bgr.shape[1]}x{original_bgr.shape[0]}"}, "glare_box": None}
    H, W = original_bgr.shape[:2]
    m = measure(original_bgr, scan_gray)
    m["resolution"] = f"{W}x{H}"
    m["glare_fraction"] = round(m["glare_fraction"], 4)
    m["mean_brightness"], m["contrast_std"] = round(m["mean_brightness"], 1), round(m["contrast_std"], 1)
    m["moire_peak_ratio"] = round(moire_score(scan_gray), 1)
    _, glare_box = glare_score(scan_gray)

    if m["min_side"] < TH["min_side_retake"]:
        reasons.append(f"Image too small ({W}x{H}); likely forwarded/downsized. Send the original photo or retake closer.")
    elif m["min_side"] < TH["min_side_warn"]:
        warnings.append("Low resolution; fine detail may be lost.")
    if m["blur_laplacian_var"] < TH["blur_retake"]:
        reasons.append("Image is blurry. Hold the phone steady, tap to focus on the film, retake.")
    if m["glare_fraction"] > TH["glare_retake"]:
        reasons.append("Glare/reflection hides part of the image. Tilt the phone or move away from the window/lightbox hotspot.")
    elif m["glare_fraction"] > TH["glare_warn"]:
        warnings.append("Minor glare spots present.")
    if m["jpeg_blockiness"] > TH["block_retake"]:
        reasons.append("Heavy compression artefacts (e.g. forwarded on WhatsApp). Send as 'Document' or share the original.")
    elif m["jpeg_blockiness"] > TH["block_warn"]:
        warnings.append("Visible compression artefacts.")
    if m["mean_brightness"] < TH["dark_retake"]:
        reasons.append("Too dark. Photograph the film against a lightbox/bright window.")
    if m["mean_brightness"] > TH["bright_retake"]:
        reasons.append("Over-exposed / washed out.")
    if m["contrast_std"] < TH["contrast_retake"]:
        reasons.append("Very low contrast; anatomy not distinguishable.")
    if geom:
        m.update({k: round(v, 2) for k, v in geom.items()})
        if abs(geom["tilt_deg"]) > TH["tilt_warn"]:
            warnings.append(f"Film rotated {geom['tilt_deg']:.0f} deg (auto-corrected).")
        if geom["perspective"] > TH["persp_retake"]:
            reasons.append("Strong perspective skew. Hold the phone parallel to the film.")
    elif not film_found:
        warnings.append("No film/screen border detected; treating upload as a digital image or tight crop.")
    if m["moire_peak_ratio"] > 400:
        warnings.append("Screen moire pattern detected; prefer exporting the image or photographing film.")

    # 0-100 score
    s = 100.0
    s -= min(40, max(0, (2 * TH["blur_retake"] - m["blur_laplacian_var"]) / (2 * TH["blur_retake"]) * 40))
    s -= min(35, m["glare_fraction"] / TH["glare_retake"] * 25)
    s -= min(25, max(0, m["jpeg_blockiness"] - 1.1) / max(0.1, TH["block_retake"] - 1.1) * 15)
    s -= 15 * len(reasons) / max(1, len(reasons)) if reasons else 0
    s = float(np.clip(s, 0, 100))
    return {
        "verdict": "RETAKE" if reasons else "PASS",
        "score": round(s, 1),
        "reasons": reasons,
        "warnings": warnings,
        "metrics": m,
        "glare_box": glare_box,
    }
