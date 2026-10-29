"""Convert downloaded RSNA ICH 16-bit PNG slices -> 8-bit brain-window (40/80 HU) 256px PNGs + labels csv.
Brain window is what is printed on film / shown on a monitor, i.e. what a phone photo captures."""
import glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2

ROOT = os.environ.get("ANYSCAN_DATA", "/workspace/data")
SRC, DST = f"{ROOT}/rsna/images", f"{ROOT}/rsna_bw"
resc = pd.read_csv(f"{ROOT}/rsna/rescale_values.csv").set_index("series_id")


def conv(p):
    rel = os.path.relpath(p, SRC)
    pid, stid, ser, fn = rel.split(os.sep)
    out = f"{DST}/{ser}/{fn}"
    if os.path.exists(out):
        return rel
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None or ser not in resc.index:
        return None
    r = resc.loc[ser]
    hu = a.astype(np.float32) * float(r.rescale_slope) + float(r.rescale_intercept)
    bw = np.clip((hu - 0) / 80 * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.resize(bw, (256, 256), interpolation=cv2.INTER_AREA))
    return rel


if __name__ == "__main__":
    files = glob.glob(f"{SRC}/*/*/*/*.png")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        done = set(x for x in pool.imap_unordered(conv, files, chunksize=64) if x)
    lab = pd.read_csv(f"{ROOT}/rsna/slice_labels.csv")
    lab["rel"] = lab.patient_ID + "/" + lab.study_ID + "/" + lab.series_ID + "/" + lab.filename
    lab = lab[lab.rel.isin(done)]
    lab["path"] = DST + "/" + lab.series_ID + "/" + lab.filename
    lab[["patient_ID", "series_ID", "filename", "path", "any"]].to_csv(f"{ROOT}/rsna_bw/labels.csv", index=False)
    print(len(lab), "slices,", lab["any"].mean().round(3), "positive,", lab.patient_ID.nunique(), "patients")
