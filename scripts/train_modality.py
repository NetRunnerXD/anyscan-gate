"""Train the modality/region classifier head on REAL public images (+ phone-photo renderings & film montages).

Train pools (disjoint from the evaluation pools used in scripts/evaluate.py):
  chest_xray : NIH CXR14 + Shenzhen/Montgomery 'calib' half (cache/cxr_table.csv) + 30% of CheXphoto real phone photos (by patient)
  head_ct    : RSNA ICH brain-window slices, train-split patients only   (CQ500 is kept for external test)
  brain_mri  : public brain-tumour MRI slices, 70% by filename hash
  other_xray : public extremity-fracture X-rays, 70%
  non_medical: Imagenette photos (half converted to greyscale), 70%
"""
import glob, hashlib, os, sys, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyscan.models import embed, MODEL_DIR
from anyscan.photosim import render, augment_for_training
from anyscan.pipeline import frontend
from anyscan.classify import CLASSES
from sklearn.linear_model import LogisticRegression

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 900))


def hpct(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


def montage(paths, rng):
    n = int(rng.choice([2, 3, 4]))
    picks = rng.choice(paths, n * n, replace=len(paths) < n * n)
    tiles = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (256, 256)) for p in picks]
    rows = [np.hstack(tiles[r * n:(r + 1) * n]) for r in range(n)]
    g = np.vstack(rows)
    return cv2.resize(g, (768, 768), interpolation=cv2.INTER_AREA)


def pools(split):
    """split='train' or 'test'. Returns {class: [paths]} (+ montage source groups for CT/MRI)."""
    tr = split == "train"
    P = {}
    cx = pd.read_csv(f"{ROOT}/cache/cxr_table.csv")
    P["chest_xray"] = cx[cx.split == ("calib" if tr else "test")].path.tolist()
    photos = sorted(glob.glob(f"{ROOT}/chexphoto/valid/natural/*/*/*/*.jpg"))
    P["chest_xray_photo"] = [p for p in photos if (hpct(p.split("/")[-3]) < 30) == tr]
    lab = pd.read_csv(f"{ROOT}/rsna_bw/labels.csv")
    lab = lab[lab.patient_ID.map(lambda x: hpct(x) >= 15)] if tr else lab[lab.patient_ID.map(lambda x: hpct(x) < 15)]
    P["head_ct"] = lab.path.tolist()
    P["head_ct_series"] = lab.groupby("series_ID").path.apply(list).tolist()
    for c in ("brain_mri", "other_xray", "non_medical"):
        fs = sorted(glob.glob(f"{ROOT}/misc/{c}/*"))
        P[c] = [f for f in fs if (hpct(os.path.basename(f)) < 70) == tr]
    return P


def variants(path, cls, rng, photo_real=False):
    """Yield images for one source file: digital, augmented, simulated phone photo (through the real front-end)."""
    if photo_real:
        img = cv2.imread(path)
        img = cv2.resize(img, None, fx=1600 / max(img.shape[:2]), fy=1600 / max(img.shape[:2]), interpolation=cv2.INTER_AREA)
        return [frontend(img)[0]]
    flag = cv2.IMREAD_COLOR if cls == "non_medical" else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if cls == "non_medical" and rng.random() < 0.5:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = [img]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cls != "non_medical":
        out.append(augment_for_training(g, rng))
        ph, _ = render(g, rng, out_side=900)
        out.append(frontend(ph)[0])
    return out


def build(split, n_per_class, seed):
    rng = np.random.default_rng(seed)
    P = pools(split)
    X, y, src = [], [], []
    for ci, c in enumerate(CLASSES):
        files = list(P[c])
        rng.shuffle(files)
        imgs = []
        for f in files[:n_per_class]:
            imgs += [(im, f) for im in variants(f, c, rng)]
        if c == "chest_xray":
            for f in P["chest_xray_photo"]:
                imgs += [(im, f) for im in variants(f, c, rng, photo_real=True)]
        if c == "head_ct":
            for k in range(n_per_class // 4):
                ser = P["head_ct_series"][int(rng.integers(len(P["head_ct_series"])))]
                imgs.append((montage(ser, rng), "montage"))
        if c == "brain_mri":
            for k in range(n_per_class // 6):
                imgs.append((montage(P["brain_mri"], rng), "montage"))
        print(split, c, len(imgs), flush=True)
        for i in range(0, len(imgs), 64):
            X.append(embed([im for im, _ in imgs[i:i + 64]]))
        y += [ci] * len(imgs); src += [s for _, s in imgs]
    return np.concatenate(X), np.array(y), src


def main():
    X, y, _ = build("train", N_PER_CLASS, 0)
    rng = np.random.default_rng(1); idx = rng.permutation(len(y)); cut = int(0.85 * len(y))
    tr, ca = idx[:cut], idx[cut:]
    mean, scale = X[tr].mean(0), X[tr].std(0) + 1e-6
    lr = LogisticRegression(C=0.05, max_iter=3000).fit((X[tr] - mean) / scale, y[tr])
    # temperature scaling on the held-out 15% of the training pool
    z = ((X[ca] - mean) / scale) @ lr.coef_.T + lr.intercept_
    best = (1e9, 1.0)
    for T in np.linspace(0.5, 3, 26):
        zz = z / T; zz -= zz.max(1, keepdims=True); lp = zz - np.log(np.exp(zz).sum(1, keepdims=True))
        nll = -lp[np.arange(len(ca)), y[ca]].mean(); best = min(best, (nll, T))
    acc = (z.argmax(1) == y[ca]).mean()
    lr = LogisticRegression(C=0.05, max_iter=3000).fit((X - mean) / scale, y)
    np.savez(MODEL_DIR / "modality_head.npz", classes=np.array(CLASSES), mean=mean, scale=scale,
             coef=lr.coef_, intercept=lr.intercept_, temperature=np.array(best[1]))
    print(json.dumps({"n_train": int(len(y)), "inner_holdout_acc": float(acc), "temperature": float(best[1])}))


if __name__ == "__main__":
    main()
