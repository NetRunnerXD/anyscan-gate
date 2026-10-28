"""AnyScan Gate CLI (deterministic, offline, CPU).

  python cli.py IMAGE [IMAGE ...] [--out DIR] [--red-flags]

IMAGE may be a phone photo / screenshot (JPEG/PNG/...) or a DICOM file. For each input writes
<stem>_report.json and annotated / rectified / redacted / overlay PNGs to --out, and prints a summary.
Exit code 0 on success, 2 if any input failed to process.
"""
import argparse, json, os, sys
import cv2
from anyscan.pipeline import run

ap = argparse.ArgumentParser(description="AnyScan Gate - NOT FOR CLINICAL USE")
ap.add_argument("images", nargs="+"); ap.add_argument("--out", default="out"); ap.add_argument("--red-flags", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
bad = 0
for path in a.images:
    try:
        rep, ims = run(path, a.red_flags)
    except Exception as e:  # report and continue with the next file
        bad += 1; print(json.dumps({"file": path, "error": str(e)})); continue
    stem = os.path.splitext(os.path.basename(path))[0]
    for k, im in ims.items():
        if im is not None:
            cv2.imwrite(os.path.join(a.out, f"{stem}_{k}.png"), im)
    with open(os.path.join(a.out, f"{stem}_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    p = rep["plugin"]
    print(json.dumps({"file": path, "modality": rep["modality"]["label"], "modality_conf": round(rep["modality"]["confidence"], 3),
                      "quality": rep["quality"]["verdict"], "quality_reasons": rep["quality"]["reasons"],
                      "plugin": None if not p else {"name": p["plugin"], "status": p["status"], "finding": p["finding"], "p": p["p_abnormal"]},
                      "route": rep["routing"]["decision"], "phi_regions_blurred": rep["redaction"]["scan_text_regions"],
                      "runtime_s": rep["runtime_s"]}))
print("NOT FOR CLINICAL USE - decision aid only.", file=sys.stderr)
sys.exit(2 if bad else 0)
