"""Modality / region / view classifier.

Frozen ImageNet ResNet-18 embedding (512-d) -> multinomial logistic-regression head trained on
REAL public images (CXR: NIH/Shenzhen/Montgomery/CheXphoto photos; head CT: RSNA ICH slices;
brain MRI: public brain-tumour MRI set; other X-ray: public extremity fracture set; non-medical:
Imagenette) with phone-photo renderings + montage grids. See scripts/train_modality.py.
"""
import numpy as np
from .models import embed, MODEL_DIR

CLASSES = ["chest_xray", "head_ct", "brain_mri", "other_xray", "non_medical"]
META = {
    "chest_xray": ("X-ray", "chest", "frontal (PA/AP)"),
    "head_ct": ("CT", "head", "axial"),
    "brain_mri": ("MRI", "head / brain", "axial/sagittal/coronal (not resolved)"),
    "other_xray": ("X-ray", "musculoskeletal / other", "unknown"),
    "non_medical": ("not a medical image", "-", "-"),
}
_head = None


def _load():
    global _head
    if _head is None:
        d = np.load(MODEL_DIR / "modality_head.npz", allow_pickle=False)
        _head = {k: d[k] for k in d.files}
        assert list(_head["classes"]) == CLASSES, "modality head class order mismatch"
    return _head


def predict_proba_from_features(F):
    h = _load()
    z = ((F - h["mean"]) / h["scale"]) @ h["coef"].T + h["intercept"]
    z = z / float(h["temperature"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


def classify(img):
    """img: uint8 gray or BGR (rectified scan or whole photo)."""
    p = predict_proba_from_features(embed([img]))[0]
    i = int(np.argmax(p)); top = CLASSES[i]
    modality, region, view = META[top]
    return {"label": top, "modality": modality, "region": region, "view": view,
            "confidence": float(p[i]), "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
            "model": "resnet18-imagenet-frozen + logistic head (models/modality_head.npz)"}
