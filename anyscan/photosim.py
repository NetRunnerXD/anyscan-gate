"""PhotoShift: render a REAL digital scan as a realistic phone photo of a film/monitor.

Used (a) as training augmentation for photo robustness and (b) to build the PhotoShift-IN
test set from held-out real images. Every rendering returns its ground truth (film quad,
applied degradations, expected quality verdict) so the gate can be scored objectively.
These are simulated captures of real images - NOT real phone photos (see docs/EVAL.md)."""
import cv2
import numpy as np

RETAKE_KINDS = ("blur", "glare", "jpeg", "lowres", "dark")


def _background(rng, H, W, kind):
    if kind == "lightbox":
        bg = np.full((H, W, 3), rng.integers(215, 250), np.uint8)
        bg = cv2.GaussianBlur((bg + rng.normal(0, 4, bg.shape)).clip(0, 255).astype(np.uint8), (0, 0), 3)
    elif kind == "table":
        base = np.array(rng.integers(60, 200, 3), np.float32)
        noise = cv2.GaussianBlur(rng.normal(0, 1, (H // 8 + 1, W // 8 + 1)).astype(np.float32), (0, 0), 2)
        noise = cv2.resize(noise, (W, H))[..., None]
        bg = (base + 25 * noise).clip(0, 255).astype(np.uint8)
    else:  # monitor bezel / dark room
        bg = np.full((H, W, 3), rng.integers(5, 40), np.uint8)
    return bg


def render(scan_gray, rng=None, severity=None, kind=None, out_side=None):
    """scan_gray: uint8 2D real image. severity: None (random), 'clean', or a dict of forced degradations.
    Returns (photo_bgr, info)."""
    rng = rng or np.random.default_rng()
    h0, w0 = scan_gray.shape
    out_side = out_side or int(rng.integers(1200, 2000))
    kind = kind or rng.choice(["lightbox", "table", "monitor"])
    H = out_side
    W = int(out_side * rng.uniform(0.7, 1.0)) if rng.random() < 0.6 else int(out_side * rng.uniform(1.0, 1.4))
    bg = _background(rng, H, W, kind)
    # film placement
    s = min(H, W) * rng.uniform(0.65, 0.88) / max(h0, w0)
    fw, fh = w0 * s, h0 * s
    cx, cy = W / 2 + rng.uniform(-0.06, 0.06) * W, H / 2 + rng.uniform(-0.06, 0.06) * H
    deg = {} if severity in (None, "clean") else dict(severity)
    if severity is None:  # random mix, mostly mild
        for k, p in (("blur", 0.2), ("glare", 0.2), ("tilt", 0.15), ("jpeg", 0.2), ("lowres", 0.1), ("dark", 0.08), ("moire", 0.1)):
            if rng.random() < p:
                deg[k] = rng.choice(["mild", "severe"], p=[0.6, 0.4])
    tilt = rng.uniform(-4, 4)
    if "tilt" in deg:
        tilt = rng.choice([-1, 1]) * (rng.uniform(9, 16) if deg["tilt"] == "mild" else rng.uniform(22, 35))
    persp = rng.uniform(0, 0.06)
    a = np.deg2rad(tilt)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-fw / 2, -fh / 2], [fw / 2, -fh / 2], [fw / 2, fh / 2], [-fw / 2, fh / 2]])
    corners[0, 0] += persp * fw; corners[1, 0] -= persp * fw * rng.uniform(0, 1)
    quad = (corners @ R.T + [cx, cy]).astype(np.float32)
    src = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])
    M = cv2.getPerspectiveTransform(src, quad)
    film = cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR)
    if kind == "lightbox":  # transmitted light: slight blue-ish tint, lifted blacks
        film = (film.astype(np.float32) * 0.92 + 12).clip(0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(film, M, (W, H))
    mask = cv2.warpPerspective(np.full((h0, w0), 255, np.uint8), M, (W, H))
    img = np.where(mask[..., None] > 0, warped, bg)
    # vignetting + colour cast (phone white balance)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    cast = rng.uniform(0.92, 1.08, 3)
    img = (img.astype(np.float32) * vig[..., None] * cast)
    if "dark" in deg:
        img *= 0.35 if deg["dark"] == "mild" else 0.18
    img = img.clip(0, 255)
    if "glare" in deg:
        n = 1 if deg["glare"] == "mild" else 2
        for _ in range(n):
            gx, gy = cx + rng.uniform(-0.3, 0.3) * fw, cy + rng.uniform(-0.3, 0.3) * fh
            r = min(fw, fh) * (rng.uniform(0.05, 0.08) if deg["glare"] == "mild" else rng.uniform(0.12, 0.2))
            d = np.hypot(xx - gx, (yy - gy) * rng.uniform(0.6, 1.4))
            g = np.clip(1.6 - d / r, 0, 1)[..., None]
            img = img * (1 - g) + 255 * g
    if "moire" in deg:
        f = rng.uniform(0.6, 1.2)
        amp = 6 if deg["moire"] == "mild" else 16
        img += amp * np.sin(f * xx + 0.3 * yy)[..., None]
    img += rng.normal(0, rng.uniform(1.5, 4), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    if "blur" in deg:
        if deg["blur"] == "mild":
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(1.2, 2.0))
        else:
            k = int(rng.integers(15, 31)); ker = np.zeros((k, k), np.float32)
            ker[k // 2] = 1; ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1), (k, k))
            img = cv2.filter2D(cv2.GaussianBlur(img, (0, 0), 2.5), -1, ker / ker.sum())
    if "lowres" in deg:
        side = 600 if deg["lowres"] == "mild" else 380
        sc = side / min(H, W)
        img = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
        quad = quad * sc
    q = 90
    if "jpeg" in deg:
        # WhatsApp-like: downscale to 1600 long side + aggressive JPEG, possibly re-forwarded
        q = int(rng.integers(35, 55)) if deg["jpeg"] == "mild" else int(rng.integers(5, 12))
        sc = min(1.0, 1600 / max(img.shape[:2]))
        if sc < 1:
            img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA); quad = quad * sc
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # tilt / perspective are auto-corrected and moire only warns, so they do not make a capture unreadable
    severe = [k for k, v in deg.items() if v == "severe" and k in RETAKE_KINDS]
    info = {"quad": quad.tolist(), "background": str(kind), "degradations": {k: str(v) for k, v in deg.items()},
            "jpeg_quality": q, "tilt_deg": float(tilt),
            "expected_quality": "RETAKE" if severe else "PASS"}
    return img, info


def augment_for_training(gray, rng):
    """Cheap photo-domain augmentation applied directly to a scan crop (no background)."""
    g = gray.astype(np.float32)
    if rng.random() < 0.5:
        g = 255 * (g / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        g = g * rng.uniform(0.75, 1.1) + rng.uniform(-15, 20)
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.5, 1.8))
    if rng.random() < 0.3:
        h, w = g.shape; yy, xx = np.mgrid[0:h, 0:w]
        gx, gy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.05, 0.15) * w
        gl = np.clip(1.5 - np.hypot(xx - gx, yy - gy) / r, 0, 1) * rng.uniform(0.3, 0.8)
        g = g * (1 - gl) + 255 * gl
    if rng.random() < 0.4:
        g = g + rng.normal(0, rng.uniform(1, 6), g.shape)
    g = g.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, e = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 80))])
        g = cv2.imdecode(e, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.4:  # small rotation / perspective residual after rectification
        h, w = g.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
        g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=int(g[:5, :5].mean()))
    return g
