"""Download REAL public datasets used for training/evaluation (no login required).

  python scripts/download_data.py cxr        # Shenzhen + Montgomery TB sets (NLM), NIH CXR14 batch 001 (HF mirror)
  python scripts/download_data.py cq500 N    # first N of 19 CQ500 tars (HF neurosentry/cq500), converted to brain-window PNG slices
  python scripts/download_data.py rsna GB    # stream the first GB gigabytes of the RSNA ICH 16-bit PNG tar (HF ianpan/...)
  python scripts/download_data.py mri        # brain MRI slices (HF)  [modality classifier only]
  python scripts/download_data.py misc       # non-medical + other X-ray images [modality classifier only]

Data root: $ANYSCAN_DATA (default /workspace/data). Licences: see docs/DATA.md.
"""
import concurrent.futures as cf
import io, os, shutil, subprocess, sys, tarfile, glob
import numpy as np
import cv2

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
NLM = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
HF = "https://huggingface.co/datasets"


def fetch(url, timeout=120):
    r = subprocess.run(["curl", "-sfL", "--max-time", str(timeout), url], capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else None


def save_small(data, out, max_side=1024):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    s = max_side / max(img.shape)
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, img)
    return True


def cxr():
    out = os.path.join(ROOT, "cxr"); os.makedirs(f"{out}/shenzhen", exist_ok=True); os.makedirs(f"{out}/montgomery", exist_ok=True)
    jobs = []
    for i in range(1, 663):
        lab = 0 if i <= 326 else 1
        jobs.append((f"{NLM}/Shenzhen-Hospital-CXR-Set/CXR_png/CHNCXR_{i:04d}_{lab}.png", f"{out}/shenzhen/CHNCXR_{i:04d}_{lab}.png"))
    for i in range(1, 420):
        for lab in (0, 1):
            jobs.append((f"{NLM}/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_{i:04d}_{lab}.png", f"{out}/montgomery/MCUCXR_{i:04d}_{lab}.png"))

    def go(j):
        url, dst = j
        if os.path.exists(dst):
            return 1
        d = fetch(url)
        return int(bool(d) and save_small(d, dst))
    with cf.ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(go, jobs))
    print("NLM TB CXR images:", n)
    # NIH ChestX-ray14 batch 001 (~5k images) + labels
    nih = f"{out}/nih"; os.makedirs(nih, exist_ok=True)
    base = f"{HF}/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
    if not os.path.exists(f"{nih}/Data_Entry_2017_v2020.csv"):
        open(f"{nih}/Data_Entry_2017_v2020.csv", "wb").write(fetch(f"{base}/Data_Entry_2017_v2020.csv", 600))
    if len(glob.glob(f"{nih}/images/*.png")) < 4000:
        z = f"{nih}/images_001.zip"
        subprocess.run(["curl", "-sfL", "-o", z, f"{base}/images/images_001.zip"], check=True)
        os.makedirs(f"{nih}/images", exist_ok=True)
        import zipfile
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith(".png"):
                    save_small(zf.read(m), f"{nih}/images/{os.path.basename(m)}")
        os.remove(z)
    print("NIH images:", len(glob.glob(f"{nih}/images/*.png")))


def _series_score(name, n):
    u = name.upper()
    s = 0
    if "PLAIN" in u: s += 3
    if "THIN" in u or "0.625" in u: s -= 2
    if "BONE" in u: s -= 5
    if "POST" in u or "CONTRAST" in u: s -= 3
    if 18 <= n <= 60: s += 2
    return s


def window(hu, c=40, w=80):
    lo = c - w / 2
    return np.clip((hu - lo) / w * 255, 0, 255).astype(np.uint8)


def convert_cq500_dir(scan_dir, out_dir):
    import pydicom
    series = {}
    for f in glob.glob(f"{scan_dir}/**/*.dcm", recursive=True):
        series.setdefault(os.path.dirname(f), []).append(f)
    if not series:
        return 0
    ranked = sorted(series, key=lambda k: -_series_score(os.path.basename(k), len(series[k])))
    for best in ranked:
        sl = _read_series(series[best])
        if len(sl) >= 10:
            break
    os.makedirs(out_dir, exist_ok=True)
    for k, (z, hu) in enumerate(sl):
        cv2.imwrite(f"{out_dir}/{k:03d}.png", cv2.resize(window(hu), (256, 256), interpolation=cv2.INTER_AREA))
    with open(f"{out_dir}/series.txt", "w") as fh:
        fh.write(os.path.basename(best))
    return len(sl)


def _read_series(files):
    import pydicom
    sl = []
    for f in files:
        try:
            d = pydicom.dcmread(f)
            z = float(d.ImagePositionPatient[2])
            hu = d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
            sl.append((z, hu))
        except Exception:
            pass
    sl.sort(key=lambda t: t[0])
    return sl


def cq500(n):
    out = os.path.join(ROOT, "cq500"); os.makedirs(f"{out}/slices", exist_ok=True)
    if not os.path.exists(f"{out}/reads.csv"):
        open(f"{out}/reads.csv", "wb").write(fetch(f"{HF}/neurosentry/cq500/resolve/main/reads.csv"))
    for i in range(1, n + 1):
        tag = f"qct{i:02d}"
        if os.path.exists(f"{out}/{tag}.done"):
            continue
        tmp = f"{out}/tmp_{tag}"; os.makedirs(tmp, exist_ok=True)
        p1 = subprocess.Popen(["curl", "-sfL", f"{HF}/neurosentry/cq500/resolve/main/{tag}.tar"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "xf", "-", "-C", tmp], stdin=p1.stdout, check=True); p1.wait()
        for scan in sorted(glob.glob(f"{tmp}/{tag}/*")):
            sid = os.path.basename(scan).split()[0].replace("CQ500CT", "CQ500-CT-")
            if len(glob.glob(f"{out}/slices/{sid}/*.png")) >= 10:
                continue
            k = convert_cq500_dir(scan, f"{out}/slices/{sid}")
            print(tag, sid, k, flush=True)
        shutil.rmtree(tmp); open(f"{out}/{tag}.done", "w").close()


def rsna(gb):
    out = os.path.join(ROOT, "rsna"); os.makedirs(out, exist_ok=True)
    base = f"{HF}/ianpan/rsna-intracranial-hemorrhage-16bit-png/resolve/main"
    for f in ("slice_labels.csv", "rescale_values.csv"):
        if not os.path.exists(f"{out}/{f}"):
            open(f"{out}/{f}", "wb").write(fetch(f"{base}/{f}", 600))
    cmd = f"curl -sL -r 0-{int(gb * 1e9) - 1} {base}/images.tar | tar xf - -C {out} --ignore-zeros"
    subprocess.run(cmd, shell=True)


def _hf_list(repo, prefix=""):
    import json, re, urllib.request
    out, url = [], f"https://huggingface.co/api/datasets/{repo}/tree/main/{prefix}?recursive=true"
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            out += [x["path"] for x in json.load(r) if x["type"] == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
            url = m.group(1) if m else None
    return out


def _grab(repo, paths, dst, max_side=512):
    os.makedirs(dst, exist_ok=True)
    def go(p):
        o = f"{dst}/{p.replace('/', '__')}"
        if os.path.exists(o):
            return 1
        d = fetch(f"{HF}/{repo}/resolve/main/{p}")
        return int(bool(d) and save_small(d, o, max_side))
    with cf.ThreadPoolExecutor(16) as ex:
        return sum(ex.map(go, paths))


def misc():
    """Modality-classifier-only sets: brain MRI, extremity X-ray, non-medical photos."""
    import random
    rnd = random.Random(0)
    mri = [p for p in _hf_list("BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset") if p.lower().endswith(".jpg")]
    rnd.shuffle(mri)
    print("brain MRI", _grab("BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset", mri[:1200], f"{ROOT}/misc/brain_mri"))
    bone = [p for p in _hf_list("Mahadih534/x-ray_bone-fracture-Dataset") if p.lower().endswith(".jpg")]
    print("bone xray", _grab("Mahadih534/x-ray_bone-fracture-Dataset", bone, f"{ROOT}/misc/other_xray"))
    # non-medical: imagenette parquet
    import pandas as pd
    pq = f"{ROOT}/misc/imagenette.parquet"
    if not os.path.exists(pq):
        subprocess.run(["curl", "-sfL", "-o", pq, f"{HF}/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet"], check=True)
    df = pd.read_parquet(pq).sample(1200, random_state=0)
    os.makedirs(f"{ROOT}/misc/non_medical", exist_ok=True)
    col = [c for c in df.columns if "image" in c.lower()][0]
    for i, v in enumerate(df[col]):
        b = v["bytes"] if isinstance(v, dict) else v
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f"{ROOT}/misc/non_medical/in_{i:04d}.jpg", img)
    print("non-medical", len(df))


if __name__ == "__main__":
    what = sys.argv[1]
    {"cxr": cxr, "cq500": lambda: cq500(int(sys.argv[2]) if len(sys.argv) > 2 else 8),
     "misc": misc, "rsna": lambda: rsna(float(sys.argv[2]) if len(sys.argv) > 2 else 12)}[what]()
