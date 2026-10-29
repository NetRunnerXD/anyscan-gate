"""Smoke test on REAL public-dataset samples (samples/, built by scripts/make_samples.py).
Asserts: no crash, full report schema, valid enums, expected modality / quality verdicts, abstention on RETAKE,
non-medical rejection, PHI redaction of synthetic identifiers on a real CXR, DICOM input path, determinism.
Run: python tests/smoke_test.py   (exit code 0 = pass)"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anyscan.pipeline import run
from anyscan.redact import recall

ROUTES = {"refer_now", "urgent_telerad", "routine", "rejected_not_medical"}
man = json.load(open(os.path.join(ROOT, "samples", "manifest.json")))
fails = 0


def check_schema(rep):
    bad = []
    for k in ("version", "disclaimer", "input", "modality", "quality", "film_detection", "redaction", "plugin", "routing", "runtime_s"):
        if k not in rep:
            bad.append(f"missing {k}")
    if "NOT FOR CLINICAL USE" not in rep.get("disclaimer", ""):
        bad.append("disclaimer missing")
    m = rep["modality"]
    if not (0 <= m["confidence"] <= 1) or abs(sum(m["probs"].values()) - 1) > 0.02:
        bad.append("bad modality probs")
    if rep["quality"]["verdict"] not in ("PASS", "RETAKE"):
        bad.append("bad verdict")
    if rep["routing"]["decision"] not in ROUTES:
        bad.append("bad route")
    p = rep["plugin"]
    if p is not None:
        for k in ("plugin", "status", "p_abnormal", "positive", "abstain_reasons", "operating_points"):
            if k not in p:
                bad.append(f"plugin missing {k}")
        if p["status"] not in ("result", "abstain") or not (0 <= p["p_abnormal"] <= 1):
            bad.append("bad plugin output")
    return bad


for name, info in sorted(man.items()):
    path = os.path.join(ROOT, "samples", name)
    try:
        rep, ims = run(path)
    except Exception as e:
        fails += 1; print(f"FAIL {name}: crashed: {e!r}"); continue
    bad = check_schema(rep)
    e = info["expect"]
    got = {"modality": rep["modality"]["label"], "quality": rep["quality"]["verdict"]}
    for k in ("modality", "quality"):
        if k in e and got[k] != e[k]:
            bad.append(f"{k}: expected {e[k]} got {got[k]}")
    p = rep["plugin"]
    if got["modality"] in ("chest_xray", "head_ct") and p is None:
        bad.append("no plug-in ran")
    if got["quality"] == "RETAKE" and p is not None and p["status"] != "abstain":
        bad.append("plug-in must abstain on RETAKE")
    if got["modality"] == "non_medical" and rep["routing"]["decision"] != "rejected_not_medical":
        bad.append("non-medical not rejected")
    if "phi_boxes" in e:
        r = recall(rep["redaction"]["photo_boxes"], e["phi_boxes"])   # boxes are in original-image coordinates
        got["phi_recall"] = r
        if r < 0.75:
            bad.append(f"PHI recall {r:.2f} < 0.75")
    if name.endswith(".dcm") and not rep["input"].get("dicom"):
        bad.append("DICOM not detected")
    pstr = "-" if p is None else f"{p['status']}:{p['finding']} p={p['p_abnormal']:.3f}"
    fails += bool(bad)
    print(f"{'FAIL' if bad else 'ok  '} {name:40s} {got} plugin={pstr} route={rep['routing']['decision']} "
          f"| truth: {info['ground_truth']} {bad if bad else ''}")

# determinism: same input -> identical decision/probabilities
n1 = sorted(man)[0]
a, _ = run(os.path.join(ROOT, "samples", n1)); b, _ = run(os.path.join(ROOT, "samples", n1))
strip = lambda r: json.dumps({k: r[k] for k in ("modality", "quality", "plugin", "routing")}, sort_keys=True, default=str)
if strip(a) != strip(b):
    fails += 1; print("FAIL non-deterministic output")
# red-flag override
r, _ = run(os.path.join(ROOT, "samples", n1), red_flags=True)
if r["routing"]["decision"] != "refer_now":
    fails += 1; print("FAIL red-flag override")
print("SMOKE TEST:", "PASSED" if not fails else f"FAILED ({fails})")
sys.exit(1 if fails else 0)
