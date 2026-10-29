"""Choose head-CT operating points on the RSNA *validation* patients (never CQ500).
Image-level use is "max over the slices visible on the image", so thresholds are set at series level:
  lo = sensitivity >= 0.95 (series), hi = specificity >= 0.90 (series)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyscan.models import MODEL_DIR

SENS, SPEC = float(os.environ.get("ICH_SENS", 0.95)), float(os.environ.get("ICH_SPEC", 0.90))
va = pd.read_csv(MODEL_DIR / "ich_val_predictions.csv")
ser = va.groupby("series_ID").agg(y=("any", "max"), p=("p", "max"))
ps, ns = np.sort(ser.p[ser.y == 1].values), np.sort(ser.p[ser.y == 0].values)
lo = float(ps[int(np.floor((1 - SENS) * len(ps)))])
hi = float(ns[min(len(ns) - 1, int(np.ceil(SPEC * len(ns))))])
hi = max(hi, lo)
cfg_path = MODEL_DIR / "plugin_config.json"
cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
cfg["head_ct"] = {"lo": round(lo, 4), "hi": round(hi, 4),
                  "rule": f"RSNA validation series-level: lo = sensitivity>={SENS}, hi = specificity>={SPEC}",
                  "val_metrics": {"n_series": int(len(ser)), "abstain_rate_val_series": round(float(((ser.p >= lo) & (ser.p < hi)).mean()), 3)}}
json.dump(cfg, open(cfg_path, "w"), indent=1)
print(cfg["head_ct"])
