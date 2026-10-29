"""Train the head-CT intracranial-haemorrhage slice classifier on REAL RSNA ICH slices.

Input representation = 8-bit brain window (W80/L40), i.e. what is printed on film / shown on
a monitor and therefore what a phone photo can capture. ImageNet ResNet-18, fine-tuned on CPU.
Patient-level split (hash of patient ID): 85% train / 15% validation. External test = CQ500 (scripts/evaluate.py).
  python scripts/train_ich.py [epochs] [neg_per_pos]
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyscan.models import resnet18, to_tensor_rgb, MODEL_DIR
from anyscan.photosim import augment_for_training
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
torch.manual_seed(0); np.random.seed(0)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NEG_PER_POS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
SIDE = 224
LR = float(os.environ.get("ICH_LR", 3e-4))
OUT_NAME = os.environ.get("ICH_OUT", "ich_resnet18.pt")
torch.set_num_threads(int(os.environ.get("THREADS", 8)))


def split_of(pid):
    return "val" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100 < 15 else "train"


class DS(Dataset):
    def __init__(self, df, train):
        self.p, self.y, self.train = df.path.values, df["any"].values.astype(np.float32), train

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        g = cv2.imread(self.p[i], cv2.IMREAD_GRAYSCALE)
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                g = g[:, ::-1].copy()
            s = rng.uniform(0.85, 1.0); h = int(256 * s); y0, x0 = rng.integers(0, 257 - h, 2)
            g = augment_for_training(g[y0:y0 + h, x0:x0 + h], rng)
        return to_tensor_rgb(g, SIDE)[0], self.y[i]


def predict(model, df, bs=64):
    dl = DataLoader(DS(df, False), batch_size=bs, num_workers=4)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x).squeeze(1).numpy())
    return np.concatenate(out)


def main():
    lab = pd.read_csv(f"{ROOT}/rsna_bw/labels.csv")
    lab["split"] = lab.patient_ID.map(split_of)
    tr, va = lab[lab.split == "train"], lab[lab.split == "val"]
    print(f"train {len(tr)} slices ({tr['any'].mean():.3f} pos, {tr.patient_ID.nunique()} patients); "
          f"val {len(va)} ({va['any'].mean():.3f} pos, {va.patient_ID.nunique()} patients)", flush=True)
    rng = np.random.default_rng(0)
    va_eval = pd.concat([va[va["any"] == 1], va[va["any"] == 0].sample(min(len(va[va["any"] == 0]), 3 * (va["any"] == 1).sum()), random_state=0)])
    model = resnet18(1)
    if os.environ.get("ICH_INIT"):
        model.load_state_dict(torch.load(os.environ["ICH_INIT"], map_location="cpu", weights_only=False)["state_dict"])
        print("init from", os.environ["ICH_INIT"], flush=True)
    for p in list(model.conv1.parameters()) + list(model.bn1.parameters()):
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    pos, neg = tr[tr["any"] == 1], tr[tr["any"] == 0]
    n_ep = len(pos) + int(NEG_PER_POS * len(pos))
    steps = EPOCHS * (n_ep // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps + 5, pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, step = -1, 0
    for ep in range(EPOCHS):
        ep_df = pd.concat([pos, neg.sample(int(NEG_PER_POS * len(pos)), random_state=ep)]).sample(frac=1, random_state=ep)
        dl = DataLoader(DS(ep_df, True), batch_size=32, shuffle=True, num_workers=5, drop_last=True, persistent_workers=False)
        model.train(); t0 = time.time(); tl = 0
        for i, (x, y) in enumerate(dl):
            loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            tl += loss.item()
            if i % 50 == 0:
                print(f"ep{ep} it{i}/{len(dl)} loss {tl / (i + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        lv = predict(model, va_eval)
        auc = roc_auc_score(va_eval["any"], lv)
        print(f"== epoch {ep} val AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), f"/tmp/{OUT_NAME}.best")
    model.load_state_dict(torch.load(f"/tmp/{OUT_NAME}.best"))
    model.eval()
    # Platt calibration on the FULL validation patients (true prevalence)
    lv = predict(model, va)
    from sklearn.linear_model import LogisticRegression
    pl = LogisticRegression(C=1e6).fit(lv[:, None], va["any"].values)
    a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
    p = 1 / (1 + np.exp(-(a * lv + b)))
    va = va.assign(p=p)
    ser = va.groupby("series_ID").agg(y=("any", "max"), p=("p", "max"))
    meta = {"arch": "resnet18", "input": "8-bit brain window W80/L40, letterboxed 224, ImageNet norm, gray->RGB",
            "train_data": "RSNA Intracranial Hemorrhage Detection (HF ianpan/rsna-intracranial-hemorrhage-16bit-png subset)",
            "train_slices": int(len(tr)), "val_slices": int(len(va)), "val_patients": int(va.patient_ID.nunique()),
            "val_slice_auc": float(roc_auc_score(va["any"], p)), "val_series_auc": float(roc_auc_score(ser.y, ser.p)),
            "platt": [a, b], "epochs": EPOCHS, "init": os.environ.get("ICH_INIT", "imagenet"), "lr": LR, "neg_per_pos": NEG_PER_POS}
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, MODEL_DIR / OUT_NAME)
    va[["patient_ID", "series_ID", "filename", "any", "p"]].to_csv(MODEL_DIR / OUT_NAME.replace(".pt", "_val_predictions.csv").replace("ich_resnet18_val", "ich_val"), index=False)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
