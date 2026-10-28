"""Find the film / screen inside a phone photo and rectify it (perspective warp)."""
import cv2
import numpy as np
from .utils import order_quad, resize_max


def _quad_candidates(mask, H, W):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
        area = cv2.contourArea(c)
        if area < 0.10 * H * W or area > 0.96 * H * W:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                q = order_quad(ap.reshape(4, 2).astype(np.float32))
                fill = area / max(cv2.contourArea(q), 1)
                ang = []
                for i in range(4):
                    v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
                    ang.append(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))))
                if fill > 0.85 and max(abs(a_ - 90) for a_ in ang) < 25:
                    out.append((q, area))
                break
    return out


def find_film(img_bgr):
    """Return (quad in original coords or None, info dict).
    Candidates: low-saturation region, edge contours, and multi-threshold brightness masks
    (image on a dark monitor bezel, or dark film on a bright lightbox)."""
    small, s = resize_max(img_bgr, 800)
    H, W = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = max(9, int(0.03 * min(H, W)) | 1)
    masks = []
    m1 = (hsv[..., 1] < 40).astype(np.uint8) * 255
    masks.append(("lowsat", cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    masks.append(("edges", cv2.dilate(edges, np.ones((5, 5), np.uint8))))
    g = cv2.GaussianBlur(gray, (7, 7), 0)
    for t in (25, 40, 60, 80):
        m = (g > t).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        masks.append((f"bright>{t}", m))
    for t in (200, 225):
        m = (g < t).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        masks.append((f"dark<{t}", m))
    # image displayed on a dark monitor: it is a bright "hole" inside the dark screen/bezel region
    for t in (30, 45, 65):
        dark = (g < t).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hole = np.zeros((H, W), np.uint8)
        for c, h in zip(cnts, hier[0]):
            if h[3] >= 0 and cv2.contourArea(c) > 0.08 * H * W:
                cv2.drawContours(hole, [c], -1, 255, -1)
        if hole.any():
            masks.insert(0, (f"screen-hole<{t}", hole))
    cands = []
    for name, m in masks:
        for q, area in _quad_candidates(m, H, W):
            inner = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(inner, q.astype(np.int32), 1)
            sat = float(hsv[..., 1][inner > 0].mean())
            tex = float(cv2.Laplacian(gray, cv2.CV_32F)[inner > 0].std())
            if sat < 45:
                cands.append((q, area, name, sat, tex))
    if not cands:
        return None, {"film_found": False}
    # prefer the largest grey textured quad that is not (almost) the whole frame
    holes = [c for c in cands if c[2].startswith("screen-hole")]
    pool = holes if holes else cands
    q, area, name, sat, tex = max(pool, key=lambda c: c[1] * (0.6 if c[1] > 0.9 * H * W else 1.0))
    return order_quad(q) / s, {"film_found": True, "film_area_frac": round(float(area / (H * W)), 3), "detector": name}


def quad_geometry(quad, shape):
    """Tilt (deg), perspective ratio, and margin to frame for a quad."""
    tl, tr, br, bl = quad
    ang_top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    ang_bot = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    tilt = float((ang_top + ang_bot) / 2)
    wt, wb = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    hl, hr = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    persp = float(max(abs(wt - wb) / max(wt, wb), abs(hl - hr) / max(hl, hr)))
    H, W = shape[:2]
    margin = float(min(quad[:, 0].min(), quad[:, 1].min(), W - quad[:, 0].max(), H - quad[:, 1].max()))
    return {"tilt_deg": tilt, "perspective": persp, "edge_margin_px": margin}


def warp(img, quad):
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img, M, (w, h))
    # trim a thin border (film edge / table bleed)
    b = max(2, int(0.01 * min(w, h)))
    return out[b:h - b, b:w - b], M


def content_crop(img):
    """Crop a rectangular image displayed inside a dark (monitor) or bright (lightbox) surround.
    Only fires when the content is clearly rectangular, so head-CT ellipses on black are untouched."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small, s = resize_max(gray, 600)
    H, W = small.shape
    best = None
    for m in ((small > 35), (small < 225)):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h, a = st[i]
        frac, fill = w * h / (H * W), a / max(1, w * h)
        if 0.2 < frac < 0.9 and fill > 0.9 and (best is None or frac > best[1]):
            best = ((x, y, w, h), frac)
    if best is None:
        return img, None
    x, y, w, h = (np.array(best[0]) / s).astype(int)
    return img[y:y + h, x:x + w], [int(x), int(y), int(w), int(h)]
