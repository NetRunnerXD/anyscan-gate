"""End-to-end AnyScan Gate pipeline (deterministic, CPU, offline once models are present)."""
import time
import cv2
import numpy as np
from . import DISCLAIMER, __version__
from .io import load_any
from .rectify import find_film, quad_geometry, warp, content_crop
from .quality import assess
from .classify import classify
from .redact import detect_text, redact
from .plugins import plugin_for
from .router import route


def frontend(img, inmeta=None):
    """Find + rectify the film/screen, then crop to displayed content. Returns (scan_bgr, quad, finfo, geom)."""
    inmeta = inmeta or {}
    quad, finfo = (None, {"film_found": False, "skipped": "digital DICOM"}) if inmeta.get("dicom") else find_film(img)
    if quad is not None and finfo.get("film_area_frac", 0) > 0.9 and abs(quad_geometry(quad, img.shape)["tilt_deg"]) < 3:
        # quad ~ whole frame and square to it: a digital image / tight crop, not a photographed film - don't warp
        quad, finfo = None, {"film_found": False, "ignored_full_frame_quad": True}
    if quad is not None:
        geom = quad_geometry(quad, img.shape)
        scan, _ = warp(img, quad)
    else:
        geom, scan = None, img.copy()
    if not inmeta.get("dicom"):
        scan, crop_box = content_crop(scan)
        finfo["content_crop"] = crop_box
    return scan, quad, finfo, geom


def run(src, red_flags=False, classical_redaction=False):
    t0 = time.time()
    img, inmeta = load_any(src)
    scan, quad, finfo, geom = frontend(img, inmeta)
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # PHI redaction: every exported/forwarded image is the redacted one. The on-device AI reads the
    # un-redacted pixels (nothing leaves the device) so a false-positive text box can never hide a lesion.
    scan_boxes = detect_text(scan, classical_redaction)
    scan_red = redact(scan, scan_boxes)
    photo_boxes = detect_text(img, classical_redaction) if not inmeta.get("dicom") else scan_boxes
    photo_red = redact(img, photo_boxes)
    scan_red_gray = cv2.cvtColor(scan_red, cv2.COLOR_BGR2GRAY)

    cls = classify(scan)
    q = assess(img, scan_gray, geom, finfo["film_found"], digital=bool(inmeta.get("dicom")))
    plugin = plugin_for(cls["label"])
    pres = None
    if plugin is not None:
        ctx = {"modality_conf": cls["confidence"], "quality_score": q["score"], "quality_verdict": q["verdict"]}
        pres = plugin.run(scan_gray, ctx)
    decision, why = route(q, cls, pres, red_flags)

    annotated = img.copy()
    if quad is not None:
        cv2.polylines(annotated, [quad.astype(np.int32)], True, (0, 255, 0), max(2, img.shape[1] // 300))
    for x, y, w, h in photo_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    report = {
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "input": inmeta,
        "modality": cls,
        "quality": {k: q[k] for k in ("verdict", "score", "reasons", "warnings", "metrics")},
        "film_detection": finfo,
        "redaction": {"scan_text_regions": len(scan_boxes), "photo_text_regions": len(photo_boxes),
                      "scan_boxes": scan_boxes, "photo_boxes": photo_boxes,
                      "engine": "PP-OCRv3 DB text detector (ONNX, offline)" + (" + classical text-line detector" if classical_redaction else "")},
        "plugin": None if pres is None else {k: v for k, v in pres.items() if k != "overlay"},
        "routing": {"decision": decision, "why": why},
        "runtime_s": round(time.time() - t0, 2),
    }
    images = {"annotated": annotated, "rectified": scan, "redacted_scan": scan_red,
              "redacted_photo": photo_red, "overlay": None if pres is None else pres["overlay"]}
    return report, images
