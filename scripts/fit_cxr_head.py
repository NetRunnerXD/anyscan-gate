"""Fit the chest-X-ray abnormal-vs-normal head on REAL labelled CXRs and choose abstention thresholds.

Features = 18 TorchXRayVision DenseNet-121 ('all') pathology outputs.
Calibration split (50% by patient/image hash): NIH ChestX-ray14 batch 001 (abnormal = any finding)
  + Shenzhen & Montgomery TB sets (abnormal = TB).  The other 50% is held out for docs/EVAL.md.
Thresholds: lo = highest p keeping sensitivity >= 0.90 on calib; hi = lowest p giving specificity >= 0.90.
"""
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THREADS", 8)))
from anyscan.plugins.chest_xray import ChestXrayAbnormal
from anyscan.models import MODEL_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
CACHE = f"{ROOT}/cache"; os.makedirs(CACHE, exist_ok=True)


def half(key):
    return "calib" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def cxr_table():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/cxr/shenzhen/*.png")) + sorted(glob.glob(f"{ROOT}/cxr/montgomery/*.png")):
        b = os.path.basename(f)
        rows.append({"path": f, "dataset": "shenzhen_tb" if b.startswith("CHN") else "montgomery_tb",
                     "y": int(b[-5]), "split": half(b)})
    nih = pd.read_csv(f"{ROOT}/cxr/nih/Data_Entry_2017_v2020.csv")
    have = {os.path.basename(f): f for f in glob.glob(f"{ROOT}/cxr/nih/images/*.png")}
    nih = nih[nih["Image Index"].isin(have)]
    for _, r in nih.iterrows():
        rows.append({"path": have[r["Image Index"]], "dataset": "nih_cxr14", "y": int(r["Finding Labels"] != "No Finding"),
                     "split": half(str(r["Patient ID"]))})
    return pd.DataFrame(rows)


def features(df, plug, name):
    cf = f"{CACHE}/{name}.npy"
    if os.path.exists(cf):
        F = np.load(cf)
        if len(F) == len(df):
            return F
    out = []
    for i in range(0, len(df), 32):
        gs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in df.path.values[i:i + 32]]
        out.append(plug.features(gs))
        if i % 640 == 0:
            print(name, i, len(df), flush=True)
    F = np.concatenate(out); np.save(cf, F); return F


def thresholds(y, p, sens=0.90, spec=0.90):
    ps, ns = np.sort(p[y == 1]), np.sort(p[y == 0])
    lo = float(ps[int(np.floor((1 - sens) * len(ps)))])          # >= sens of positives have p >= lo
    hi = float(ns[min(len(ns) - 1, int(np.ceil(spec * len(ns))))])  # spec of negatives have p < hi
    return lo, max(hi, lo)


def main():
    df = cxr_table()
    df.to_csv(f"{CACHE}/cxr_table.csv", index=False)
    plug = ChestXrayAbnormal()
    F = features(df, plug, "cxr_xrv_feats")
    cal = (df.split == "calib").values
    mean, scale = F[cal].mean(0), F[cal].std(0) + 1e-6
    # balance datasets so the TB sets are not drowned by NIH
    w = df.loc[cal].groupby("dataset").y.transform("size").rdiv(1.0).values
    lr = LogisticRegression(C=0.5, max_iter=2000).fit((F[cal] - mean) / scale, df.y[cal], sample_weight=w * cal.sum() / w.sum())
    p = 1 / (1 + np.exp(-(((F - mean) / scale) @ lr.coef_[0] + lr.intercept_[0])))
    lo, hi = thresholds(df.y[cal].values, p[cal])
    vm = {}
    for ds in df.dataset.unique():
        m = (df.dataset == ds).values
        vm[ds] = {s: round(float(roc_auc_score(df.y[m & (df.split == s).values], p[m & (df.split == s).values])), 4) for s in ("calib", "test")}
    cfg_path = MODEL_DIR / "plugin_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    cfg["chest_xray"] = {"backbone": "torchxrayvision densenet121-res224-all", "pathologies": list(plug.features.__self__ and __import__("anyscan.models", fromlist=["x"]).get_xrv().pathologies),
                         "mean": mean.tolist(), "scale": scale.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                         "lo": round(lo, 4), "hi": round(hi, 4),
                         "threshold_rule": "lo: sensitivity>=0.90, hi: specificity>=0.90 on calibration split",
                         "val_metrics": {"auc_by_dataset": vm}}
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    df.assign(p=p).to_csv(f"{CACHE}/cxr_predictions.csv", index=False)
    print(json.dumps(cfg["chest_xray"]["val_metrics"], indent=1), "lo", lo, "hi", hi)


if __name__ == "__main__":
    main()
