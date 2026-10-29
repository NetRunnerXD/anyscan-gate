import numpy as np


def abstain_gate(p, ctx, lo, hi, min_modality_conf=0.80):
    """Uncertainty logic shared by plug-ins.
    lo / hi are operating points chosen on REAL validation data (see models/*.json):
      p < lo  -> 'no finding' side (lo targets high sensitivity),
      p >= hi -> 'finding' side   (hi targets high specificity),
      lo <= p < hi -> ABSTAIN (model is not reliable enough here).
    Also abstains when the image type is uncertain or capture quality failed."""
    reasons = []
    if lo <= p < hi:
        reasons.append(f"Model probability {p:.2f} lies in the validated uncertain band [{lo:.2f}, {hi:.2f}).")
    if ctx["modality_conf"] < min_modality_conf:
        reasons.append(f"Unsure this is the right image type (confidence {ctx['modality_conf']:.2f} < {min_modality_conf}).")
    if ctx["quality_verdict"] != "PASS":
        reasons.append("Capture quality failed (RETAKE); AI output not trusted.")
    if ctx.get("ood"):
        reasons.append(ctx["ood"])
    margin = (p - hi) / max(1e-6, 1 - hi) if p >= hi else (lo - p) / max(1e-6, lo) if p < lo else 0.0
    return bool(reasons), reasons, round(float(np.clip(margin, 0, 1)), 3)


def colorize(gray_bgr, cam, alpha=0.4):
    import cv2
    heat = cv2.applyColorMap((255 * np.clip(cam, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gray_bgr, 1 - alpha, heat, alpha, 0)
