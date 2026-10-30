"""Evaluate AnyScan Gate on REAL held-out data and write docs/EVAL.md + docs/eval.json.

  python scripts/evaluate.py [section ...]     sections: ich cxr modality quality phi film runtime  (default: all)

Nothing here is used for training. Data provenance per section is written into the report.
"""
import glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from sklearn.metrics import roc_auc_score, confusion_matrix
from anyscan.models import MODEL_DIR, load_json
from anyscan.photosim import render
from anyscan.pipeline import frontend, run
from anyscan.quality import assess, measure
from anyscan.classify import classify, CLASSES, predict_proba_from_features
from anyscan.models import embed
from anyscan.plugins.head_ct import HeadCTICH, split_montage
from anyscan.plugins.chest_xray import ChestXrayAbnormal
from anyscan.redact import detect_text, recall

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
DOCS = os.path.join(os.path.dirname(HERE), "docs"); os.makedirs(DOCS, exist_ok=True)
OUT = os.path.join(DOCS, "eval.json")
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
PS = f"{ROOT}/photoshift_in"; os.makedirs(PS, exist_ok=True)


def hpct(k):
    return int(hashlib.md5(k.encode()).hexdigest(), 16) % 100


def bin_metrics(y, p, lo, hi):
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None}
    for name, t in (("at_hi", hi), ("at_lo", lo)):
        pr = p >= t
        out[f"sens_{name}"] = round(float(pr[y == 1].mean()), 4) if y.sum() else None
        out[f"spec_{name}"] = round(float((~pr[y == 0]).mean()), 4) if (y == 0).sum() else None
    dec = (p < lo) | (p >= hi)
    out["coverage_non_abstained"] = round(float(dec.mean()), 4)
    if dec.any():
        yy, pp = y[dec], p[dec] >= hi
        out["sens_non_abstained"] = round(float(pp[yy == 1].mean()), 4) if (yy == 1).any() else None
        out["spec_non_abstained"] = round(float((~pp[yy == 0]).mean()), 4) if (yy == 0).any() else None
        out["acc_non_abstained"] = round(float((pp == yy).mean()), 4)
    return out


def boot_ci(y, p, n=500, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


# ------------------------------------------------------------------ head CT / ICH
def cq500_scans():
    reads = pd.read_csv(f"{ROOT}/cq500/reads.csv")
    reads["sid"] = reads.name.str.replace("CQ500-CT-", "CQ500-CT-")
    reads["y"] = (reads[["R1:ICH", "R2:ICH", "R3:ICH"]].sum(1) >= 2).astype(int)
    rows = []
    for _, r in reads.iterrows():
        fs = sorted(glob.glob(f"{ROOT}/cq500/slices/{r['name']}/*.png"))
        if len(fs) >= 10:
            rows.append({"sid": r["name"], "y": int(r.y), "files": fs, "category": r.Category})
    return rows


def montage_of(files, n=4, side=256):
    k = len(files); idx = np.linspace(int(0.25 * k), int(0.85 * k), n * n).astype(int)   # supratentorial-to-vertex band, like a printed film
    tiles = [cv2.resize(cv2.imread(files[i], cv2.IMREAD_GRAYSCALE), (side, side)) for i in idx]
    pad = 6
    tiles = [cv2.copyMakeBorder(t, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) for t in tiles]
    return np.vstack([np.hstack(tiles[r * n:(r + 1) * n]) for r in range(n)])


def eval_ich():
    plug = HeadCTICH(); cfg = plug.cfg; lo, hi = cfg["lo"], cfg["hi"]
    _, meta = __import__("anyscan.models", fromlist=["x"]).get_ich()
    res = {"model": meta, "operating_points": {"lo": lo, "hi": hi, "rule": cfg.get("rule")}}
    va = pd.read_csv(MODEL_DIR / "ich_val_predictions.csv")
    res["rsna_heldout_slices"] = bin_metrics(va["any"], va.p, lo, hi)
    ser = va.groupby("series_ID").agg(y=("any", "max"), p=("p", "max"))
    res["rsna_heldout_series"] = bin_metrics(ser.y, ser.p, lo, hi)
    scans = cq500_scans()
    ys, p_all, p_mont, p_photo, q_photo = [], [], [], [], []
    rng = np.random.default_rng(7)
    t0 = time.time()
    for i, s in enumerate(scans):
        gs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in s["files"]]
        ps = np.concatenate([plug.slice_probs(gs[j:j + 64]) for j in range(0, len(gs), 64)])
        mont = montage_of(s["files"])
        pm = float(plug.slice_probs([mont[y:y + h, x:x + w] for x, y, w, h in split_montage(mont)]).max())
        ph, info = render(mont, rng, severity="clean" if i % 2 else None)
        scan, quad, finfo, geom = frontend(ph)
        sg = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        tiles = [sg[y:y + h, x:x + w] for x, y, w, h in split_montage(sg)]
        pp = float(plug.slice_probs(tiles).max())
        q = assess(ph, sg, None, finfo["film_found"])["verdict"]
        ys.append(s["y"]); p_all.append(float(ps.max())); p_mont.append(pm); p_photo.append(pp); q_photo.append(q)
        if i % 25 == 0:
            print("cq500", i, len(scans), f"{time.time() - t0:.0f}s", flush=True)
        if i < 3:
            cv2.imwrite(f"{PS}/cq500_film_{s['sid']}.jpg", ph)
    ys = np.array(ys)
    res["cq500_all_slices"] = {**bin_metrics(ys, p_all, lo, hi), "auc_95ci": boot_ci(ys, p_all)}
    res["cq500_film_montage_digital"] = {**bin_metrics(ys, p_mont, lo, hi), "auc_95ci": boot_ci(ys, p_mont)}
    res["cq500_film_montage_phone_sim"] = {**bin_metrics(ys, p_photo, lo, hi), "auc_95ci": boot_ci(ys, p_photo)}
    pf = np.array(p_photo); qp = np.array(q_photo) == "PASS"
    res["cq500_phone_sim_pass_only"] = bin_metrics(ys[qp], pf[qp], lo, hi) if qp.sum() > 10 else None
    res["cq500_flip_rate_digital_vs_phone"] = round(float(((np.array(p_mont) >= hi) != (pf >= hi)).mean()), 4)
    res["cq500_n_scans"] = len(scans)
    pd.DataFrame({"sid": [s["sid"] for s in scans], "y": ys, "p_all_slices": p_all, "p_montage": p_mont, "p_phone": p_photo, "quality": q_photo}).to_csv(f"{DOCS}/cq500_predictions.csv", index=False)
    R["ich"] = res


# ------------------------------------------------------------------ chest X-ray
def eval_cxr():
    plug = ChestXrayAbnormal(); cfg = plug.cfg; lo, hi = cfg["lo"], cfg["hi"]
    res = {"operating_points": {"lo": lo, "hi": hi, "rule": cfg["threshold_rule"]}}
    pr = pd.read_csv(f"{ROOT}/cache/cxr_predictions.csv")
    te = pr[pr.split == "test"]
    for ds in te.dataset.unique():
        d = te[te.dataset == ds]
        res[f"{ds}_test"] = {**bin_metrics(d.y, d.p, lo, hi), "auc_95ci": boot_ci(d.y.values, d.p.values)}
    tb = te[te.dataset.str.endswith("_tb")]
    res["tb_sets_combined_test"] = {**bin_metrics(tb.y, tb.p, lo, hi), "auc_95ci": boot_ci(tb.y.values, tb.p.values)}
    # CheXpert validation (radiologist consensus labels): digital originals vs CheXphoto REAL smartphone photos
    va = pd.read_parquet(f"{ROOT}/chexphoto/chexpert_valid.parquet")
    va = va[va["Frontal/Lateral"] == 0]
    rows = []
    for _, r in va.iterrows():
        pid, st, vw = r.Path.split("/")[-3:]
        if hpct(pid) < 30:      # 30% of CheXphoto patients were used to train the modality classifier
            continue
        nat = f"{ROOT}/chexphoto/valid/natural/oneplus/{pid}/{st}/{vw}"
        syn = f"{ROOT}/chexphoto/valid/synthetic/photographic/{pid}/{st}/{vw}"
        if not os.path.exists(nat):
            continue
        img = r.image; b = img["bytes"] if isinstance(img, dict) else img
        dig = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        rows.append({"pid": pid, "y": int(r["No Finding"] != 3), "dig": dig, "nat": nat, "syn": syn})
    yd, pdg, pn, ps_, qn, mod_ok = [], [], [], [], [], []
    for r in rows:
        yd.append(r["y"])
        pdg.append(float(plug.prob(plug.features([r["dig"]]))[0]))
        for path, acc in ((r["nat"], pn), (r["syn"], ps_)):
            ph = cv2.imread(path)
            sc = 2000 / max(ph.shape[:2]); ph = cv2.resize(ph, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else ph
            scan, quad, finfo, geom = frontend(ph)
            sg = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
            acc.append(float(plug.prob(plug.features([sg]))[0]))
            if acc is pn:
                qn.append(assess(ph, sg, geom, finfo["film_found"])["verdict"])
                mod_ok.append(classify(scan)["label"] == "chest_xray")
    yd = np.array(yd); pn_a = np.array(pn); qn = np.array(qn) == "PASS"
    res["chexpert_valid_digital"] = {**bin_metrics(yd, pdg, lo, hi), "auc_95ci": boot_ci(yd, pdg)}
    res["chexphoto_real_phone_photos"] = {**bin_metrics(yd, pn, lo, hi), "auc_95ci": boot_ci(yd, pn)}
    res["chexphoto_synthetic_photographic"] = {**bin_metrics(yd, ps_, lo, hi), "auc_95ci": boot_ci(yd, ps_)}
    res["chexphoto_flip_rate_digital_vs_real_photo"] = round(float(((np.array(pdg) >= hi) != (pn_a >= hi)).mean()), 4)
    res["chexphoto_real_photo_quality_pass_rate"] = round(float(qn.mean()), 4)
    res["chexphoto_real_photo_modality_correct"] = round(float(np.mean(mod_ok)), 4)
    if qn.sum() > 10 and (~qn).sum() > 3:
        res["chexphoto_real_photos_PASS_only"] = bin_metrics(yd[qn], pn_a[qn], lo, hi)
        res["chexphoto_real_photos_RETAKE_only"] = bin_metrics(yd[~qn], pn_a[~qn], lo, hi)
    R["cxr"] = res


# ------------------------------------------------------------------ modality
def eval_modality():
    from train_modality import pools, montage
    P = pools("test"); rng = np.random.default_rng(3)
    sets = {}
    for c in ("chest_xray", "brain_mri", "other_xray", "non_medical"):
        f = list(P[c]); rng.shuffle(f); sets[c] = f[:200]
    cq = cq500_scans()
    sets["head_ct"] = [s["files"][len(s["files"]) // 2] for s in cq[:200]]      # CQ500 = external (India)
    res = {}
    for mode in ("digital", "phone_sim"):
        y, X = [], []
        for ci, c in enumerate(CLASSES):
            for f in sets[c]:
                img = cv2.imread(f, cv2.IMREAD_COLOR if c == "non_medical" else cv2.IMREAD_GRAYSCALE)
                if mode == "phone_sim":
                    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if c == "non_medical":
                        img = img  # natural photos are already photos
                    else:
                        ph, _ = render(g if g.shape[0] >= 400 else cv2.resize(g, None, fx=2, fy=2), rng)
                        img = frontend(ph)[0]
                X.append(img); y.append(ci)
        F = np.concatenate([embed(X[i:i + 64]) for i in range(0, len(X), 64)])
        pr = predict_proba_from_features(F); yh = pr.argmax(1); y = np.array(y)
        res[mode] = {"n": int(len(y)), "accuracy": round(float((yh == y).mean()), 4),
                     "per_class_recall": {c: round(float((yh[y == i] == i).mean()), 4) for i, c in enumerate(CLASSES)},
                     "confusion": confusion_matrix(y, yh, labels=range(len(CLASSES))).tolist(),
                     "mean_conf_correct": round(float(pr.max(1)[yh == y].mean()), 4),
                     "frac_below_0.8_conf": round(float((pr.max(1) < 0.8).mean()), 4)}
    # CQ500 film montages photographed (simulated) + CheXphoto real photos
    mont = [frontend(render(montage_of(s["files"]), rng)[0])[0] for s in cq[:60]]
    pr = predict_proba_from_features(embed(mont))
    res["cq500_montage_phone_sim_head_ct_recall"] = round(float((pr.argmax(1) == CLASSES.index("head_ct")).mean()), 4)
    res["classes"] = CLASSES
    res["sources"] = {"chest_xray": "NIH/Shenzhen/Montgomery held-out half", "head_ct": "CQ500 (external, not used in training)",
                      "brain_mri": "brain-tumour MRI set, 30% held-out", "other_xray": "fracture X-ray set, 30% held-out",
                      "non_medical": "Imagenette, 30% held-out"}
    R["modality"] = res


# ------------------------------------------------------------------ quality gate on PhotoShift-IN test split
def eval_quality():
    from calibrate_quality import sample_set
    rows = sample_set("test", 280, 11)
    man = []
    y, yh, per = [], [], {}
    for i, r in enumerate(rows):
        scan, quad, finfo, geom = frontend(r["photo"])
        q = assess(r["photo"], cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY), None, finfo["film_found"])
        exp = r["info"]["expected_quality"]; y.append(exp); yh.append(q["verdict"])
        for k, v in (r["info"]["degradations"] or {"none": "clean"}).items():
            per.setdefault(f"{k}:{v}", []).append(q["verdict"])
        name = f"ps_{i:04d}.jpg"; cv2.imwrite(f"{PS}/{name}", r["photo"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        man.append({"file": name, "source_class": r["cls"], "source": os.path.relpath(r["src"], ROOT), **r["info"], "gate": q["verdict"]})
    json.dump(man, open(f"{PS}/manifest.json", "w"), indent=1)
    y, yh = np.array(y), np.array(yh)
    R["quality"] = {"n": int(len(y)), "agreement": round(float((y == yh).mean()), 4),
                    "retake_recall": round(float((yh[y == "RETAKE"] == "RETAKE").mean()), 4),
                    "pass_recall": round(float((yh[y == "PASS"] == "PASS").mean()), 4),
                    "retake_rate_by_degradation": {k: round(float(np.mean(np.array(v) == "RETAKE")), 3) for k, v in sorted(per.items())},
                    "thresholds": load_json("quality_config.json")}


# ------------------------------------------------------------------ PHI redaction
NAMES = ["RAMESH KUMAR", "SUNITA DEVI", "MOHD IRFAN", "PRIYA SHARMA", "ANIL YADAV", "LAKSHMI N", "ARJUN SINGH", "FATIMA BEGUM"]


def overlay_phi(g, rng):
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); H, W = g.shape; boxes = []
    texts = [f"{rng.choice(NAMES)} {rng.integers(18, 85)}Y/{rng.choice(['M', 'F'])}",
             f"UHID {rng.integers(1000000, 9999999)}", f"{rng.integers(1, 28):02d}/{rng.integers(1, 12):02d}/2025 {rng.integers(0, 23):02d}:{rng.integers(0, 59):02d}",
             f"DIST HOSP {rng.choice(['SEHORE', 'VIDISHA', 'RAISEN', 'BHOPAL'])}"]
    corners = [(0.03, 0.06), (0.03, 0.97), (0.55, 0.06), (0.5, 0.97)]
    for t, (fx, fy) in zip(texts, corners):
        fs = W / 1100 * rng.uniform(0.7, 1.1); th = max(1, int(fs * 2))
        (tw, tht), bl = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y0 = int(fx * W), int(fy * H)
        x = min(x, W - tw - 2)
        col = 255 if rng.random() < 0.8 else 0
        cv2.putText(img, t, (x, y0), cv2.FONT_HERSHEY_SIMPLEX, fs, (col, col, col), th, cv2.LINE_AA)
        boxes.append([x, y0 - tht, tw, tht + bl])
    return img, boxes


def eval_phi():
    from train_modality import pools
    P = pools("test"); rng = np.random.default_rng(5)
    srcs = []
    for c in ("chest_xray", "other_xray", "brain_mri"):
        f = list(P[c]); rng.shuffle(f); srcs += f[:40]
    cq = cq500_scans(); srcs += [s["files"][len(s["files"]) // 2] for s in cq[:40]]
    rec_d, rec_p, fp_area, clean_fp = [], [], [], []
    for i, f in enumerate(srcs):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if max(g.shape) < 800:
            g = cv2.resize(g, None, fx=1024 / max(g.shape), fy=1024 / max(g.shape), interpolation=cv2.INTER_CUBIC)
        img, gt = overlay_phi(g, rng)
        pb = detect_text(img); rec_d.append(recall(pb, gt))
        m = np.zeros(g.shape, bool)
        for x, y, w, h in pb: m[max(0, y):y + h, max(0, x):x + w] = True
        for x, y, w, h in gt: m[max(0, y):y + h, max(0, x):x + w] = False
        fp_area.append(float(m.mean()))
        # photographed version: transform GT boxes through the film quad, then score on the photo
        ph, info = render(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rng, severity="clean")
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [g.shape[1], 0], [g.shape[1], g.shape[0]], [0, g.shape[0]]]), np.float32(info["quad"]))
        gtp = []
        for x, y, w, h in gt:
            pts = cv2.perspectiveTransform(np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]]), Hm)[0]
            x0, y0 = pts.min(0); x1, y1 = pts.max(0); gtp.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
        rec_p.append(recall(detect_text(ph), gtp))
        # false positives on the clean (no overlay) real image
        cb = detect_text(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        cm = np.zeros(g.shape, bool)
        for x, y, w, h in cb: cm[max(0, y):y + h, max(0, x):x + w] = True
        clean_fp.append(float(cm.mean()))
    R["phi"] = {"n_images": len(srcs), "n_text_lines": 4 * len(srcs),
                "recall_digital": round(float(np.mean(rec_d)), 4), "recall_phone_sim": round(float(np.mean(rec_p)), 4),
                "images_with_all_lines_redacted_digital": round(float(np.mean(np.array(rec_d) == 1.0)), 4),
                "images_with_all_lines_redacted_phone_sim": round(float(np.mean(np.array(rec_p) == 1.0)), 4),
                "over_redaction_area_frac_mean": round(float(np.mean(fp_area)), 4),
                "blurred_area_frac_on_clean_images_mean": round(float(np.mean(clean_fp)), 4),
                "note": "Real held-out images with SYNTHETIC rendered PHI text (names/UHID/date/hospital); public data contains no real PHI."}
    cv2.imwrite(f"{PS}/phi_example.jpg", img)


# ------------------------------------------------------------------ film detection
def eval_film():
    from train_modality import pools
    P = pools("test"); rng = np.random.default_rng(9); ious = []
    srcs = []
    for c in ("chest_xray", "head_ct", "brain_mri", "other_xray"):
        f = list(P[c]); rng.shuffle(f); srcs += f[:40]
    from anyscan.rectify import find_film
    for f in srcs:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ph, info = render(g if g.shape[0] >= 400 else cv2.resize(g, None, fx=2, fy=2), rng, severity="clean")
        q, fi = find_film(ph)
        if q is None:
            ious.append(0.0); continue
        a = np.zeros(ph.shape[:2], np.uint8); b = a.copy()
        cv2.fillConvexPoly(a, np.int32(info["quad"]), 1); cv2.fillConvexPoly(b, np.int32(q), 1)
        ious.append(float((a & b).sum() / max(1, (a | b).sum())))
    ious = np.array(ious)
    R["film"] = {"n": int(len(ious)), "detected": round(float((ious > 0).mean()), 4), "iou_mean": round(float(ious.mean()), 4),
                 "iou_gt_0.8": round(float((ious > 0.8).mean()), 4),
                 "note": "Simulated captures (known film quad) of real held-out images on lightbox/table/monitor backgrounds."}


def eval_runtime():
    fs = sorted(glob.glob(f"{ROOT}/chexphoto/valid/natural/oneplus/*/*/*.jpg"))[:3] + [montage_of(cq500_scans()[0]["files"])]
    ts = []
    for f in fs:
        src = f if isinstance(f, str) else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        t = time.time(); run(src); ts.append(time.time() - t)
    R["runtime"] = {"seconds_per_image_mean": round(float(np.mean(ts[1:])), 2), "cpu": os.cpu_count(), "note": "end-to-end incl. OCR redaction + Grad-CAM, warm models"}


SECTIONS = {"ich": eval_ich, "cxr": eval_cxr, "modality": eval_modality, "quality": eval_quality, "phi": eval_phi, "film": eval_film, "runtime": eval_runtime}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        t = time.time(); print("==", s, flush=True); SECTIONS[s]()
        cur = json.load(open(OUT)) if os.path.exists(OUT) else {}
        cur.update({s: R[s]}); R.update(cur)                 # merge with sections written by other runs
        json.dump(R, open(OUT, "w"), indent=1, default=float); print(s, "done", f"{time.time() - t:.0f}s", flush=True)
    from write_eval_md import write
    write(R, os.path.join(DOCS, "EVAL.md"))
