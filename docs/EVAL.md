# AnyScan Gate – Evaluation on real held-out data

_Generated 2026-10-30 by `python scripts/evaluate.py`. Raw numbers: `docs/eval.json`._

> **NOT FOR CLINICAL USE.** These are retrospective research metrics on public datasets. They are not a clinical validation, and performance on Indian district-hospital films/phones is unknown until prospectively tested under ethics approval.

**How to read the plug-in tables.** Each plug-in has two operating points chosen on *validation* data (never on the test sets): `lo` (high-sensitivity point) and `hi` (high-specificity point). p ≥ hi → flagged; p < lo → "no finding" (still routine human read); lo ≤ p < hi → **abstain** (routed to urgent human read). *Coverage* = share of images where the plug-in did not abstain.

## 1. Head CT – intracranial haemorrhage plug-in

Model: ResNet-18 (ImageNet init) fine-tuned on **44,182 real RSNA ICH slices** (brain window), 3 epochs on CPU. Operating points lo=0.230, hi=0.556 (RSNA validation series-level: lo = sensitivity>=0.95, hi = specificity>=0.9).

Labels: RSNA = per-slice expert labels (any haemorrhage); **CQ500 (Qure.ai / CARING, New Delhi, India) = scan-level majority vote of 3 radiologists (R1–R3 `ICH`)**. CQ500 was never used for training or threshold selection → external test.

| Test set | n | AUC (95% CI) | Sens / Spec @ hi (flag) | Sens / Spec @ lo | Coverage (not abstained) | Sens / Spec when not abstaining | Notes |
|---|---|---|---|---|---|---|---|
| RSNA held-out patients – per slice | 6935 (986 pos) | 0.941 | 0.580 / 0.983 | 0.799 / 0.921 | 0.916 | 0.743 / 0.982 | internal, patient-level split |
| RSNA held-out patients – per series (max) | 201 (81 pos) | 0.931 | 0.790 / 0.908 | 0.951 / 0.675 | 0.796 | 0.941 / 0.880 | internal |
| **CQ500 – full scan (max over all slices)** | 472 (197 pos) | 0.898 (0.868–0.927) | 0.777 / 0.873 | 0.914 / 0.549 | 0.754 | 0.900 / 0.812 | external, India |
| CQ500 – 4×4 film montage, digital | 472 (197 pos) | 0.900 (0.870–0.930) | 0.766 / 0.894 | 0.919 / 0.575 | 0.750 | 0.904 / 0.845 | 16 slices like a printed film |
| CQ500 – same film, simulated phone photo | 472 (197 pos) | 0.856 (0.821–0.891) | 0.706 / 0.866 | 0.898 / 0.545 | 0.733 | 0.874 / 0.802 | PhotoShift-IN rendering → full front-end |
| CQ500 – phone sim, quality-gate PASS only | 371 (153 pos) | 0.873 | 0.752 / 0.872 | 0.908 / 0.550 | 0.747 | 0.891 / 0.811 |  |

Digital→phone decision flip rate on CQ500 films: **0.102**. Scans evaluated: 472.

## 2. Chest X-ray – abnormal vs normal plug-in

Backbone: TorchXRayVision DenseNet-121 `densenet121-res224-all` (public weights; trained on NIH, PadChest, CheXpert-train, MIMIC, Google, OpenI, RSNA-pneumonia). Head: logistic regression on its 18 outputs, fitted on the *calibration half* of NIH-batch-001 + Shenzhen + Montgomery. Operating points lo=0.202, hi=0.768 (lo: sensitivity>=0.90, hi: specificity>=0.90 on calibration split).

| Test set | n | AUC (95% CI) | Sens / Spec @ hi (flag) | Sens / Spec @ lo | Coverage (not abstained) | Sens / Spec when not abstaining | Notes |
|---|---|---|---|---|---|---|---|
| Shenzhen TB set (China) – held-out half | 332 (167 pos) | 0.886 (0.848–0.920) | 0.431 / 0.988 | 0.934 / 0.558 | 0.533 | 0.868 / 0.979 | external to backbone; abnormal = TB |
| Montgomery TB set (USA) – held-out half | 60 (26 pos) | 0.900 (0.797–0.970) | 0.654 / 0.941 | 0.846 / 0.735 | 0.800 | 0.809 / 0.926 | external to backbone; abnormal = TB |
| Shenzhen + Montgomery combined | 392 (193 pos) | 0.886 (0.852–0.918) | 0.461 / 0.980 | 0.922 / 0.588 | 0.574 | 0.856 / 0.967 | external to backbone |
| NIH ChestX-ray14 batch 001 – held-out patients | 2275 (996 pos) | 0.706 (0.684–0.728) | 0.299 / 0.901 | 0.841 / 0.383 | 0.472 | 0.653 / 0.794 | ⚠ backbone saw NIH in pre-training → optimistic |
| **CheXpert validation – digital originals** | 134 (114 pos) | 0.743 (0.620–0.862) | 0.842 / 0.550 | 0.974 / 0.000 | 0.806 | 0.970 / 0.000 | radiologist-consensus labels; external |
| **CheXphoto – REAL smartphone photos of the same films** | 134 (114 pos) | 0.900 (0.831–0.952) | 0.623 / 1.000 | 0.833 / 0.750 | 0.784 | 0.789 / 1.000 | OnePlus photos of monitor → full front-end |
| CheXphoto – synthetic photographic transforms | 134 (114 pos) | 0.654 (0.497–0.808) | 0.921 / 0.250 | 0.991 / 0.000 | 0.903 | 0.991 / 0.000 |  |
| CheXphoto real photos – gate PASS | 96 (89 pos) | 0.929 | 0.764 / 1.000 | 0.966 / 0.286 | 0.760 | 0.958 / 1.000 |  |
| CheXphoto real photos – gate RETAKE | 38 (25 pos) | 0.803 | 0.120 / 1.000 | 0.360 / 1.000 | 0.842 | 0.158 / 1.000 |  |

- Digital→real-phone-photo decision flip rate (CheXpert valid vs CheXphoto): **0.358**
- ⚠ CheXpert-valid frontal films are mostly abnormal (only ~20 'No Finding' cases in the 70% held-out patients), so CIs are wide; the higher AUC on photos than on the 320-px digital originals is within overlapping CIs and should not be read as photos being better.
- Real phone photos passing the quality gate: 0.716; recognised as chest X-ray by the modality model: 1.000

## 3. Modality / region classifier

Frozen ImageNet ResNet-18 + logistic head. Test images are disjoint from training; head-CT test images come from **CQ500 (external)**.

| Condition | n | Accuracy | recall chest_xray | recall head_ct | recall brain_mri | recall other_xray | recall non_medical | share with conf < 0.8 |
|---|---|---|---|---|---|---|---|---|
| digital | 959 | 0.996 | 1.000 | 1.000 | 0.985 | 0.994 | 1.000 | 0.007 |
| phone_sim | 959 | 0.960 | 1.000 | 0.945 | 0.895 | 0.962 | 1.000 | 0.140 |

CQ500 film montages, simulated phone photo → classified head CT: 0.933

Sources: chest_xray: NIH/Shenzhen/Montgomery held-out half; head_ct: CQ500 (external, not used in training); brain_mri: brain-tumour MRI set, 30% held-out; other_xray: fracture X-ray set, 30% held-out; non_medical: Imagenette, 30% held-out

## 4. Capture-quality gate (PASS / RETAKE)

Scored on the **PhotoShift-IN test split**: simulated phone captures (lightbox / table / monitor; blur, glare, WhatsApp-style JPEG, low-res, under-exposure) of *real* held-out scans, each with a known expected verdict. Thresholds were fitted on a disjoint calibration split. No public dataset has human PASS/RETAKE labels for real phone photos, so this is a *simulation* benchmark.

- Agreement with expected verdict: **0.768** (n=280); RETAKE recall 1.000; PASS recall 0.629
- RETAKE rate by injected degradation: `blur:mild` 0.63, `blur:severe` 1.00, `dark:mild` 0.88, `dark:severe` 1.00, `glare:mild` 0.37, `glare:severe` 1.00, `jpeg:mild` 0.46, `jpeg:severe` 1.00, `lowres:mild` 0.19, `lowres:severe` 1.00, `moire:mild` 0.00, `none:clean` 0.11, `tilt:mild` 0.00, `tilt:severe` 0.00

## 5. PHI redaction

Real held-out images with SYNTHETIC rendered PHI text (names/UHID/date/hospital); public data contains no real PHI.

- Text-line recall (≥60% of line area blurred), digital: **0.894**; simulated phone photo: **0.870** (640 lines on 160 images)
- Images with *every* PHI line redacted: digital 0.700; phone 0.637
- Over-redaction (blurred area outside PHI): 0.017 of image; blurred area on clean images: 0.005

## 6. Film / screen detection

Simulated captures (known film quad) of real held-out images on lightbox/table/monitor backgrounds.

- Detected: 0.969; mean IoU 0.673; IoU > 0.8: 0.581 (n=160)

## 7. Runtime

- 1.1 s per image on 8 CPU cores, no GPU (end-to-end incl. OCR redaction + Grad-CAM, warm models).

## What these numbers do *not* show

- No prospective or Indian-PHC phone-photo validation; CheXphoto photos are of a US monitor with one phone model.
- CQ500 labels are scan-level; a film photo shows only some slices, so a bleed outside the photographed slices cannot be detected.
- The CXR head's 'abnormal' mixes TB (Shenzhen/Montgomery) and NIH findings; disease-specific performance (e.g. silicosis) is untested.
- Quality-gate and PHI metrics use simulated degradations / synthetic text on real images.
