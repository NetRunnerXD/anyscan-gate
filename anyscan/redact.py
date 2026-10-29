"""On-device PHI redaction: detect burnt-in text (names, IDs, dates, hospital) and blur it.

Primary detector: PP-OCRv3 DB text detector (ONNX, bundled with rapidocr-onnxruntime, runs offline on CPU).
Optional (off by default): classical top-hat/black-hat character-line detector - it raises recall slightly
but fires on brain sulci / bone edges, so it is disabled to avoid blurring anatomy.
All detected text lines are blurred - we deliberately do not try to decide which text is "safe".
Single-character laterality markers (L/R) may be left visible."""
import shutil
import cv2
import numpy as np


def _char_lines(gray, polarity):
    H, W = gray.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT if polarity == "bright" else cv2.MORPH_BLACKHAT, k)
    _, bw = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.bitwise_and(bw, (hat > 40).astype(np.uint8) * 255)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    chars = np.zeros_like(bw)
    hs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 6 <= h <= 60 and 1 <= w <= 2.2 * h and 0.08 < a / (w * h) < 0.95 and a >= 8:
            chars[lab == i] = 255
            hs.append(h)
    if not hs:
        return []
    mh = float(np.median(hs))
    joined = cv2.dilate(chars, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(mh * 1.2)), max(1, int(mh * 0.25)))))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(joined, 8)
    boxes = []
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        sub = chars[y:y + h, x:x + w]
        ncomp = cv2.connectedComponents(sub)[0] - 1
        if ncomp >= 3 and w > 1.8 * h and h < 0.15 * H:
            # text should have roughly constant height components; ribs/edges fail this
            boxes.append([int(x), int(y), int(w), int(h)])
    return boxes


def _tesseract_boxes(gray):
    if not shutil.which("tesseract"):
        return []
    try:
        import pytesseract
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 11")
        out = []
        for i, t in enumerate(d["text"]):
            if t.strip() and len(t.strip()) >= 2 and float(d["conf"][i]) > 50:
                out.append([d["left"][i], d["top"][i], d["width"][i], d["height"][i]])
        return out
    except Exception:
        return []


_ocr = None


def _dnn_boxes(small_bgr):
    global _ocr
    try:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        res, _ = _ocr(small_bgr, use_det=True, use_cls=False, use_rec=True)
    except Exception:
        return []
    out = []
    for quad, text, score in res or []:
        t = str(text).strip()
        if len(t) < 2:          # keep lone L / R markers
            continue
        q = np.array(quad)
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return out


def detect_text(img_bgr, classical=False):
    H, W = img_bgr.shape[:2]
    s = 1400 / max(H, W)
    small = cv2.resize(img_bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    boxes = _dnn_boxes(small)
    if classical:
        boxes += _char_lines(gray, "bright") + _char_lines(gray, "dark")
    out = []
    for x, y, w, h in boxes:
        pad = int(0.35 * h) + 3
        out.append([int((x - pad) / s), int((y - pad) / s), int((w + 2 * pad) / s), int((h + 2 * pad) / s)])
    return _merge(out, img_bgr.shape)


def _merge(boxes, shape):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(m, (max(0, x), max(0, y)), (min(W - 1, x + w), min(H - 1, y + h)), 255, -1)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return [list(map(int, st[i, :4])) for i in range(1, n)]


def redact(img_bgr, boxes):
    out = img_bgr.copy()
    for x, y, w, h in boxes:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), max(2, h / 4))
    return out


def recall(pred, truth, thr=0.6):
    """Fraction of ground-truth text boxes whose area is >= thr covered by predictions."""
    if not truth:
        return None
    H = max(y + h for x, y, w, h in truth + pred) + 5
    W = max(x + w for x, y, w, h in truth + pred) + 5
    m = np.zeros((H, W), bool)
    for x, y, w, h in pred:
        m[max(0, y):y + h, max(0, x):x + w] = True
    hit = 0
    for x, y, w, h in truth:
        reg = m[max(0, y):y + h, max(0, x):x + w]
        if reg.size and reg.mean() >= thr:
            hit += 1
    return hit / len(truth)
