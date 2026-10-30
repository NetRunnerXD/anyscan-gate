# AnyScan Gate 🩻 — v1.0 (research prototype)

> ## ⚠️ NOT FOR CLINICAL USE — DECISION AID ONLY
> Research/hackathon prototype for **VIT Bhopal Health Hackathon 3.0**. Not a medical device, not clinically validated,
> not approved by CDSCO/FDA/CE. It must not be used to diagnose, treat or rule out disease. Every image must be read by a
> qualified clinician/radiologist.

AnyScan Gate is an **offline, CPU-only safety layer for medical images that arrive as phone photos (WhatsApp-style) or DICOM files**:

```
photo / DICOM ─► find & rectify film/screen ─► modality model ─► quality gate (PASS/RETAKE)
             ─► trained AI plug-in (abstains when unsure) ─► route: refer_now | urgent_telerad | routine | rejected_not_medical
             └► PHI text redaction on every exported / forwarded image (AI itself runs on-device on the original pixels)
```

| Stage | What runs | Trained / fitted on (real public data) |
|---|---|---|
| Film / screen finder | Classical CV: monitor "screen-hole", low-saturation, brightness & edge quads → perspective warp → content crop | Rules; scored on PhotoShift-IN + CheXphoto real photos |
| PHI redaction | **PP-OCRv3 DB text detector** (ONNX via `rapidocr-onnxruntime`, offline) → pixelate+blur every text line (≥2 chars). Classical detector available but off (it blurred brain sulci) | Pretrained OCR; scored on real scans with synthetic identifiers |
| Modality / region | Frozen ImageNet ResNet-18 → logistic head (chest X-ray, head CT, brain MRI, other X-ray, non-medical) | NIH, Shenzhen, Montgomery, CheXphoto photos, RSNA ICH, brain-MRI set, fracture X-rays, Imagenette (+ simulated photos, film montages) |
| Quality gate | Blur, glare, JPEG blockiness, resolution, exposure, perspective → PASS/RETAKE with reasons | Thresholds fitted on PhotoShift-IN calibration split |
| **Plug-in: head CT haemorrhage** | ResNet-18 fine-tuned on brain-window slices; splits film montages into slices (gutter-profile grid splitter); image p = max over slices; Grad-CAM | **RSNA ICH** (44k slices) → **external test on CQ500 (India)** |
| **Plug-in: chest X-ray abnormal** | TorchXRayVision DenseNet-121 (18 pathologies) → logistic head; Grad-CAM | NIH + Shenzhen + Montgomery (calibration halves) → tested on held-out halves, CheXpert-valid, **CheXphoto real phone photos** |
| Abstention + routing | Two validated operating points per plug-in (`lo` high-sensitivity, `hi` high-specificity); between → abstain; RETAKE / unsure modality → abstain | Thresholds from validation splits only |

Headline results (full tables + caveats in **[docs/EVAL.md](docs/EVAL.md)**, all on held-out real images):

| Test | n | AUC (95% CI) |
|---|---|---|
| Head-CT ICH, **CQ500 (India, external)**, full scan | 472 scans (197 ICH) | **0.898** (0.868–0.927) |
| Head-CT ICH, CQ500 4×4 film, digital | 472 | 0.900 (0.870–0.930) |
| Head-CT ICH, same CQ500 films, *simulated* phone photo | 472 | 0.856 (0.821–0.891) |
| Head-CT ICH, RSNA held-out patients (slice / series) | 6,935 / 201 | 0.941 / 0.931 |
| CXR abnormal, Shenzhen + Montgomery TB held-out | 392 | 0.886 (0.852–0.918) |
| CXR abnormal, **CheXphoto REAL smartphone photos** (CheXpert-valid labels) | 134 (114 abn.) | 0.900 (0.831–0.952) |
| CXR abnormal, NIH CXR14 held-out (backbone saw NIH) | 2,275 | 0.706 |
| Modality, digital / simulated photo (head-CT test = CQ500) | 959 | acc 0.996 / 0.960 |
| PHI text-line recall, digital / simulated photo (synthetic PHI on real scans) | 640 lines | 0.894 / 0.870 |
| Quality gate vs expected verdict (PhotoShift-IN sim) | 280 | agreement 0.768, RETAKE recall 1.00, PASS recall 0.63 |

Abstention: at the validated operating points the ICH plug-in decides on ~75% of CQ500 scans (sens 0.90 / spec 0.81 on those) and abstains on the rest (→ urgent human read).

## Quick start (CPU, Linux/macOS, Python 3.10–3.13)

```bash
cd anyscan-gate
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
python scripts/download_models.py      # no-op if models/third_party is already present (it is in the release zip)
python tests/smoke_test.py             # runs the full pipeline on the real samples in samples/
streamlit run app.py                   # web UI on http://localhost:8501
python cli.py samples/real_phone_photo_cxr_abnormal.jpg samples/cq500_118_ich_slice.dcm --out out/
```

After installation **no network is needed**: all weights live in `models/` (trained heads) and `models/third_party/`
(ImageNet ResNet-18, TorchXRayVision DenseNet-121, OCR model inside the `rapidocr-onnxruntime` wheel).
Typical latency ≈ 1–3 s per image on an 8-core CPU (see EVAL.md). Set `ANYSCAN_THREADS` to limit CPU threads.

### CLI
`python cli.py IMAGE [IMAGE ...] [--out DIR] [--red-flags]` — accepts JPEG/PNG/BMP/WEBP/TIFF/DICOM. Writes
`<stem>_report.json` plus annotated, rectified, redacted and Grad-CAM PNGs; prints a one-line JSON summary per file.
`--red-flags` = health-worker override (always `refer_now`). Deterministic: same input → same output.

### Web app
`streamlit run app.py` — pick a real sample, upload a photo/DICOM, or use the camera. Shows routing banner, quality
reasons, redacted image (download), Grad-CAM, modality probabilities, plug-in operating points and the raw JSON report.

## Reproduce training & evaluation from scratch (~2–3 h on 8 CPU cores, ~45 GB disk peak)

```bash
export ANYSCAN_DATA=/workspace/data
python scripts/download_data.py cxr          # Shenzhen, Montgomery (NLM), NIH batch 001 (HF)
python scripts/download_data.py misc         # brain MRI, fracture X-ray, Imagenette (modality classes only)
python scripts/download_data.py rsna 9       # first ~9 GB of RSNA ICH (random patients) ; then:
python scripts/prep_rsna.py                  # -> brain-window PNGs + labels
python scripts/download_data.py cq500 19     # all CQ500 (external test) -> brain-window PNGs
mkdir -p $ANYSCAN_DATA/chexphoto && cd $ANYSCAN_DATA/chexphoto && \
  curl -L -o chexphoto_test.zip https://zenodo.org/api/records/12723955/files/chexphoto_test.zip/content && unzip -q chexphoto_test.zip && \
  curl -L -o chexpert_valid.parquet https://huggingface.co/datasets/danjacobellis/chexpert/resolve/main/data/validation-00000-of-00001.parquet && cd -
python scripts/train_ich.py 3 1.5            # ICH ResNet-18 (RSNA) -> models/ich_resnet18.pt
python scripts/set_ich_thresholds.py         # lo/hi from RSNA validation series
python scripts/fit_cxr_head.py               # CXR head + lo/hi -> models/plugin_config.json
python scripts/train_modality.py             # -> models/modality_head.npz
python scripts/calibrate_quality.py          # -> models/quality_config.json
python scripts/evaluate.py                   # -> docs/EVAL.md, docs/eval.json, PhotoShift-IN
python scripts/make_samples.py               # -> samples/ (real images + manifest with provenance)
```
Dataset provenance, licences and what could not be obtained: **[docs/DATA.md](docs/DATA.md)**.

## What is real vs. still heuristic

**Real (trained/fitted on real labelled images, evaluated on held-out real images):** modality classifier, head-CT
haemorrhage plug-in (external test on CQ500), chest-X-ray abnormal plug-in (incl. real smartphone photos from CheXphoto),
abstention thresholds, OCR text detector (pretrained PP-OCRv3).

**Classical CV / rule-based (not learned):** film/screen finder, film-montage splitter, content crop, quality-gate
metrics (only their thresholds are fitted, on simulated captures), routing rules.
**Simulated:** PhotoShift-IN degradations (used for quality-gate calibration, robustness tests and modality augmentation)
and the synthetic PHI text used to score redaction.

## Honest limitations
- **No clinical validation.** Retrospective metrics on public research datasets only; no Indian PHC/district-hospital phone photos exist publicly.
- **Head CT:** a phone photo of a film shows only some slices — bleeds on unphotographed slices cannot be detected. Brain window only (no bone window → skull fractures not assessed). Trained on a 9 GB RSNA subset with a small CPU model.
- **Chest X-ray:** "abnormal" is generic (NIH findings + TB). Not validated for silicosis, paediatric films, or laterals. The backbone saw NIH in pre-training, so NIH numbers are optimistic (flagged in EVAL.md).
- **Quality gate** thresholds come from simulated degradations; it is conservative (≈37% of clean/mild simulated captures and ≈28% of real CheXphoto photos get RETAKE) and needs field calibration.
- **Film finder** works on most monitor/lightbox photos but can include the room (window glare) in the crop; mean IoU 0.67 on simulated captures.
- **Redaction** can miss text (stylised fonts, handwriting, text over bright anatomy). Always check the redacted image before sharing. Single-letter L/R markers are intentionally kept. DICOM header PHI is never exported (only pixels).
- Modality model knows 5 classes; other CT regions (chest/abdomen CT), ultrasound, ECGs etc. will be forced into the nearest class — low confidence triggers abstention, but not always.
- No MRI plug-in; MRI images are recognised and routed to a human read.

## Layout
```
anyscan/            pipeline.py, io.py (photo+DICOM), rectify.py, redact.py, classify.py, quality.py, router.py,
                    photosim.py (PhotoShift renderer), models.py, plugins/{chest_xray,head_ct,base}.py
models/             ich_resnet18.pt, modality_head.npz, plugin_config.json, quality_config.json, third_party/
scripts/            download_data, prep_rsna, train_ich, set_ich_thresholds, fit_cxr_head, train_modality,
                    calibrate_quality, evaluate, write_eval_md, make_samples, download_models
samples/            real public-dataset samples + manifest.json (source, licence, ground truth)
tests/              smoke_test.py (real samples), ui_test.py (Streamlit AppTest)
docs/               EVAL.md, eval.json, DATA.md, cq500_predictions.csv
app.py, cli.py
```
