"""Build samples/ from REAL public datasets (run after scripts/download_data.py).
Every sample records its source dataset, licence and ground-truth label in samples/manifest.json."""
import glob, hashlib, json, os, shutil, sys
import numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from anyscan.photosim import render
from evaluate import cq500_scans, montage_of, overlay_phi, hpct

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
S = os.path.join(os.path.dirname(HERE), "samples"); os.makedirs(S, exist_ok=True)
CQ = "CQ500 (Chilamkurthy et al., Lancet 2018; qure.ai) via HF neurosentry/cq500 – CC BY-NC-SA 4.0"
NLM = "NLM Shenzhen/Montgomery TB CXR sets (Jaeger et al. 2014) – public, cite source"
CXP = "CheXphoto (Phillips et al. 2020) via Zenodo 12723955 (CC BY 4.0 rehost); labels: CheXpert validation radiologist consensus"
man = {}
rng = np.random.default_rng(42)


def put(name, src, gt, expect):
    man[name] = {"source": src, "ground_truth": gt, "expect": expect}


# --- CQ500 DICOM slices (digital input path)
import pydicom
for sid, lab in (("CQ500-CT-118", 1), ("CQ500-CT-131", 0)):
    fs = sorted(glob.glob(f"{ROOT}/dicom_samples/{sid}/*.dcm"))
    ds = sorted(((float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]), f) for f in fs))
    f = ds[int(len(ds) * 0.55)][1]
    name = f"cq500_{sid[-3:]}_{'ich' if lab else 'no_ich'}_slice.dcm"
    shutil.copy(f, f"{S}/{name}")
    put(name, CQ + " – single DICOM slice (55% height)", ("ICH present in scan (3/3 readers); this slice may or may not show it" if lab else "no ICH (3/3 readers)"),
        {"modality": "head_ct"})

# --- CQ500 film montages: digital + simulated phone photo
scans = cq500_scans()
pos = [s for s in scans if s["y"] == 1]; neg = [s for s in scans if s["y"] == 0]
for tag, s in (("ich", pos[3]), ("no_ich", neg[3])):
    m = montage_of(s["files"])
    cv2.imwrite(f"{S}/cq500_film_{tag}_{s['sid'][-3:]}.png", m)
    put(f"cq500_film_{tag}_{s['sid'][-3:]}.png", CQ + " – 4x4 film montage (brain window)", f"{'ICH' if s['y'] else 'no ICH'} (scan-level majority of 3 radiologists)", {"modality": "head_ct"})
    ph, info = render(m, rng, severity="clean", kind="lightbox")
    cv2.imwrite(f"{S}/phone_sim_cq500_film_{tag}_{s['sid'][-3:]}.jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 90])
    put(f"phone_sim_cq500_film_{tag}_{s['sid'][-3:]}.jpg", CQ + " – PhotoShift-IN simulated phone photo of the film on a lightbox",
        f"{'ICH' if s['y'] else 'no ICH'} (scan-level)", {"modality": "head_ct", "quality": "PASS"})

# --- CheXphoto REAL smartphone photos (held-out patients)
va = pd.read_parquet(f"{ROOT}/chexphoto/chexpert_valid.parquet"); va = va[va["Frontal/Lateral"] == 0]
picked = {}
for _, r in va.iterrows():
    pid, st, vw = r.Path.split("/")[-3:]
    if hpct(pid) < 30:
        continue
    y = int(r["No Finding"] != 3)
    f = f"{ROOT}/chexphoto/valid/natural/oneplus/{pid}/{st}/{vw}"
    if os.path.exists(f) and y not in picked:
        picked[y] = (f, r)
for y, (f, r) in picked.items():
    im = cv2.imread(f); sc = 1800 / max(im.shape[:2]); im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    pos_labels = [c for c in ["Cardiomegaly", "Lung Opacity", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion", "Pneumothorax", "Pneumonia", "Lung Lesion", "Enlarged Cardiomediastinum", "Fracture", "Pleural Other"] if r[c] == 3]
    name = f"real_phone_photo_cxr_{'abnormal' if y else 'normal'}.jpg"
    cv2.imwrite(f"{S}/{name}", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    put(name, CXP + " – REAL OnePlus photo of a monitor", ("abnormal: " + ", ".join(pos_labels)) if y else "No Finding", {"modality": "chest_xray"})

# --- Shenzhen digital CXRs (held-out half)
tab = pd.read_csv(f"{ROOT}/cache/cxr_table.csv")
te = tab[(tab.split == "test") & (tab.dataset == "shenzhen_tb")]
for y in (0, 1):
    f = te[te.y == y].path.iloc[2]
    name = f"shenzhen_cxr_{'tb' if y else 'normal'}.png"
    shutil.copy(f, f"{S}/{name}")
    put(name, NLM + " – digital PNG (downscaled to 1024 px)", "TB" if y else "normal", {"modality": "chest_xray", "quality": "PASS"})
    if y == 0:
        g = cv2.imread(f, 0)
        ph, info = render(g, rng, severity={"blur": "severe"}, kind="table")
        cv2.imwrite(f"{S}/phone_sim_blurry_cxr.jpg", ph)
        put("phone_sim_blurry_cxr.jpg", NLM + " – PhotoShift-IN render with SEVERE motion blur", "normal; capture should be rejected", {"quality": "RETAKE"})
        img, boxes = overlay_phi(g, rng)
        cv2.imwrite(f"{S}/cxr_with_synthetic_phi.png", img)
        put("cxr_with_synthetic_phi.png", NLM + " – with SYNTHETIC name/UHID/date/hospital text burnt in (fake identities)", "normal; 4 PHI lines must be blurred",
            {"modality": "chest_xray", "phi_boxes": boxes})

# --- brain MRI (held-out) and non-medical
mri = sorted(f for f in glob.glob(f"{ROOT}/misc/brain_mri/*") if hpct(os.path.basename(f)) >= 70)[5]
shutil.copy(mri, f"{S}/brain_mri.jpg")
put("brain_mri.jpg", "Brain Tumor MRI Dataset (Nickparvar, Kaggle; HF mirror BJyotibrat/...) – held-out", os.path.basename(mri).split("__")[1], {"modality": "brain_mri"})
nm = sorted(f for f in glob.glob(f"{ROOT}/misc/non_medical/*") if hpct(os.path.basename(f)) >= 70)[3]
shutil.copy(nm, f"{S}/not_medical.jpg")
put("not_medical.jpg", "Imagenette (fast.ai, ImageNet subset) – held-out", "not a medical image", {"modality": "non_medical"})
json.dump(man, open(f"{S}/manifest.json", "w"), indent=1)
print(json.dumps({k: v["ground_truth"] for k, v in man.items()}, indent=1))
