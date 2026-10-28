import cv2
import numpy as np


def load_image(src):
    """Accept path, bytes, or ndarray -> BGR uint8."""
    if isinstance(src, np.ndarray):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resize_max(img, max_side=1024):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img, s
    return img, 1.0


def norm_gray(gray, lo=1, hi=99.5):
    a, b = np.percentile(gray, [lo, hi])
    if b - a < 1:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
