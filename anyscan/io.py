"""Input loading: phone photos / screenshots (JPEG/PNG/...) and digital DICOM files."""
import io
import cv2
import numpy as np


MAX_SIDE = 2400


def _is_dicom(b):
    return len(b) > 132 and b[128:132] == b"DICM"


def dicom_to_bgr(b):
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    d = pydicom.dcmread(io.BytesIO(b), force=True)
    a = d.pixel_array.astype(np.float32)
    if a.ndim == 3 and a.shape[-1] != 3:
        a = a[a.shape[0] // 2]
    mod = str(getattr(d, "Modality", "")).upper()
    meta = {"dicom": True, "modality_tag": mod, "body_part_tag": str(getattr(d, "BodyPartExamined", ""))}
    if mod == "CT":
        hu = a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        g = np.clip(hu / 80.0 * 255, 0, 255)          # brain window W80/L40, same as plug-in training
        meta["window"] = "brain W80/L40"
    else:
        try:
            a = apply_voi_lut(d.pixel_array, d).astype(np.float32)
        except Exception:
            pass
        lo, hi = np.percentile(a, [0.5, 99.5])
        g = np.clip((a - lo) / max(1e-6, hi - lo) * 255, 0, 255)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            g = 255 - g
    g = g.astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), meta


def load_any(src):
    """path | bytes | ndarray -> (BGR uint8, meta). DICOM header PHI never leaves this function."""
    if isinstance(src, np.ndarray):
        img = src
        meta = {"dicom": False}
    else:
        b = bytes(src) if isinstance(src, (bytes, bytearray)) else open(str(src), "rb").read()
        if _is_dicom(b) or str(src).lower().endswith(".dcm"):
            return dicom_to_bgr(b)
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        meta = {"dicom": False}
        if img is not None and img.dtype == np.uint16:
            lo, hi = np.percentile(img, [0.5, 99.5]); img = np.clip((img - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
    if img is None:
        raise ValueError("Could not decode image (supported: JPEG/PNG/BMP/WEBP/TIFF/DICOM)")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if max(img.shape[:2]) > MAX_SIDE:   # 12-48 MP phone photos: nothing downstream needs more than this
        s = MAX_SIDE / max(img.shape[:2])
        meta["downscaled_from"] = f"{img.shape[1]}x{img.shape[0]}"
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img, meta
