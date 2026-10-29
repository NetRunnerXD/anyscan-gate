"""Plugin registry. Each plugin: name, applies_to(label), run(scan_gray, ctx) -> dict with keys
status ('result'|'abstain'), finding, p_abnormal, confidence, abstain_reasons, severity, overlay, notes."""
from .chest_xray import ChestXrayAbnormal
from .head_ct import HeadCTICH

REGISTRY = [ChestXrayAbnormal(), HeadCTICH()]


def plugin_for(label):
    for p in REGISTRY:
        if label in p.applies_to:
            return p
    return None
