"""AnyScan Gate - Streamlit demo UI.  Run: streamlit run app.py"""
import json, os
import cv2
import numpy as np
import streamlit as st
from anyscan import DISCLAIMER
from anyscan.pipeline import run
from anyscan.plugins import REGISTRY

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
st.set_page_config(page_title="AnyScan Gate", page_icon="🩻", layout="wide")

ROUTE_STYLE = {
    "refer_now": ("🔴 REFER NOW", "error"),
    "urgent_telerad": ("🟠 URGENT TELERADIOLOGY READ", "warning"),
    "routine": ("🟢 ROUTINE READ", "success"),
    "rejected_not_medical": ("⚪ NOT A MEDICAL IMAGE - nothing triaged", "info"),
}


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.title("🩻 AnyScan Gate")
st.caption("Offline safety layer for phone photos (and DICOM files) of X-ray / CT / MRI: "
           "film finding → PHI redaction → modality model → quality gate → trained AI plug-in with abstention → routing.")
st.error("**⚠️ NOT FOR CLINICAL USE — DECISION AID ONLY.** " + DISCLAIMER)
MAN = json.load(open(os.path.join(SAMPLES, "manifest.json"))) if os.path.exists(os.path.join(SAMPLES, "manifest.json")) else {}

with st.sidebar:
    st.header("Input")
    src = st.radio("Source", ["Sample image", "Upload", "Camera"], index=0)
    data = None
    if src == "Sample image":
        files = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".png", ".dcm")))
        if not files:
            st.warning("No samples. Run: python scripts/make_samples.py")
        else:
            pick = st.selectbox("Real public-dataset sample", files)
            data = open(os.path.join(SAMPLES, pick), "rb").read()
            if pick in MAN:
                st.caption(f"**Source:** {MAN[pick]['source']}  \n**Ground truth:** {MAN[pick]['ground_truth']}")
    elif src == "Upload":
        up = st.file_uploader("Photo of film / scan screen, or a DICOM file", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "dcm"])
        data = up.getvalue() if up else None
    else:
        cam = st.camera_input("Take a photo of the film")
        data = cam.getvalue() if cam else None
    red_flags = st.checkbox("Clinical red flags present (e.g. GCS drop, severe breathlessness, trauma)",
                            help="Health-worker override: always routes REFER NOW.")
    st.divider()
    st.subheader("Installed plug-ins")
    for p in REGISTRY:
        st.markdown(f"- `{p.name}` - {p.description}")
    st.caption("All processing runs locally on this machine. No image leaves the device.")

if data is None:
    st.info("Choose a sample, upload a photo, or use the camera.")
    st.stop()

with st.spinner("Analysing locally…"):
    try:
        rep, ims = run(data, red_flags)
    except Exception as e:
        st.exception(e)
        st.stop()

dec = rep["routing"]["decision"]
label, kind = ROUTE_STYLE[dec]
getattr(st, kind)(f"### {label}\n" + "\n".join(f"- {w}" for w in rep["routing"]["why"]))

c1, c2, c3, c4 = st.columns(4)
m = rep["modality"]; q = rep["quality"]; p = rep["plugin"]
c1.metric("Image type", f"{m['modality']} · {m['region']}", f"conf {m['confidence']:.2f}")
c2.metric("Capture quality", q["verdict"], f"score {q['score']:.0f}/100", delta_color="normal" if q["verdict"] == "PASS" else "inverse")
c3.metric("PHI regions blurred", rep["redaction"]["scan_text_regions"] + rep["redaction"]["photo_text_regions"])
if p is None:
    c4.metric("AI plug-in", "none for this type")
elif p["status"] == "abstain":
    c4.metric("AI plug-in", "ABSTAINED", f"p={p['p_abnormal']:.2f}", delta_color="off")
else:
    c4.metric("AI plug-in", "FLAGGED" if p["positive"] else "no finding", f"p={p['p_abnormal']:.2f}", delta_color="off")

t1, t2, t3, t4 = st.tabs(["Images", "Quality gate", "Modality & plug-in", "Raw JSON report"])
with t1:
    a, b = st.columns(2)
    a.image(rgb(ims["annotated"]), caption="Upload (green = detected film, red = text to redact)", width="stretch")
    b.image(rgb(ims["redacted_scan"]), caption="Rectified + PHI-redacted scan (safe to forward)", width="stretch")
    a2, b2 = st.columns(2)
    a2.image(rgb(ims["redacted_photo"]), caption="Redacted original photo", width="stretch")
    if ims["overlay"] is not None:
        b2.image(rgb(ims["overlay"]), caption="Plug-in Grad-CAM attention (where the model looked; not a lesion outline)", width="stretch")
    ok, enc = cv2.imencode(".png", ims["redacted_scan"])
    st.download_button("⬇️ Download redacted scan (PNG)", enc.tobytes(), "anyscan_redacted_scan.png", "image/png")
    ok, enc2 = cv2.imencode(".png", ims["redacted_photo"])
    st.download_button("⬇️ Download redacted photo (PNG)", enc2.tobytes(), "anyscan_redacted_photo.png", "image/png")
with t2:
    if q["reasons"]:
        st.error("**RETAKE** - " + " ".join(q["reasons"]))
    else:
        st.success("Capture PASSED the quality gate.")
    for w in q["warnings"]:
        st.warning(w)
    st.json(q["metrics"])
with t3:
    st.write(f"**Classification:** `{m['label']}` · view: {m['view']}")
    st.bar_chart(m["probs"])
    if p is None:
        st.info("No plug-in registered for this image type - routed for human read.")
    else:
        st.write(f"**Plug-in:** `{p['plugin']}` → status **{p['status']}**")
        if p["status"] == "abstain":
            st.warning("Abstained because: " + " ".join(p["abstain_reasons"]))
        else:
            st.write(f"Finding: **{p['finding']}** (p_abnormal={p['p_abnormal']}, confidence={p['confidence']})")
        st.caption(p["notes"])
        st.json(p["features"])
        st.json({"operating_points": p.get("operating_points")})
with t4:
    st.download_button("⬇️ Download JSON report", json.dumps(rep, indent=2, default=str), "anyscan_report.json", "application/json")
    st.json(rep)

st.caption("Limitations: research prototype; models trained/evaluated on public datasets only (RSNA ICH, CQ500, NIH, "
           "Shenzhen/Montgomery, CheXpert/CheXphoto) - see docs/EVAL.md. No clinical validation; redaction can miss text "
           "(always check before sharing); a film photo shows only some CT slices.")
