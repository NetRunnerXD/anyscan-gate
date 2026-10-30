# AnyScan Gate — Expansion plan (v1.0 → "fully fledged" v2)

> Planning document, written 2026-10-30. **No new metrics are claimed here.** Every number quoted is already in
> [EVAL.md](EVAL.md); everything else is a proposal that must be trained and evaluated on held-out real data before it
> ships. Product stays what it is: an **offline, privacy-first safety layer** for phone/WhatsApp photos (and digital files)
> of medical images — *not* a radiology AI suite and *not* a diagnostic device.

---

## 1. Current scope (facts from the v1.0 code)

### Inputs accepted today (`anyscan/io.py`, `app.py`, `cli.py`)
| Input | How it is handled |
|---|---|
| JPEG / PNG / BMP / WEBP / TIFF (incl. 16-bit PNG/TIFF) | `cv2.imdecode`; 16-bit → percentile-scaled 8-bit; alpha dropped; downscaled to ≤2400 px long side |
| Phone photo of a film on a lightbox / table, or of a monitor | Classical film/screen finder → perspective warp → content crop (`rectify.py`) |
| WhatsApp-forwarded JPEG | Accepted; quality gate detects heavy JPEG blockiness → RETAKE with "send as Document" advice |
| Screenshot / tight digital export | Full-frame quad is ignored (no warp), content crop only |
| Film montage (e.g. 4×4 head-CT film) | Head-CT plug-in splits the grid into slices (gutter-profile splitter, connected-component fallback); image p = max over slices |
| Single DICOM file (`.dcm` or `DICM` magic) | CT → HU → brain window W80/L40; other modalities → VOI LUT / percentile window; MONOCHROME1 inverted. **Header PHI never leaves `load_any`** |
| Multi-frame DICOM | Accepted, but **only the middle frame** is analysed |
| Camera (Streamlit `camera_input`) | Same as photo |
| **Not accepted:** DICOM series/zip, PDF, video, HEIC (OpenCV cannot decode it), text/report input | — |

### What the models cover today
| Component | Output | Trained / fitted on (real public data) |
|---|---|---|
| Modality/region classifier | 5 classes: `chest_xray`, `head_ct`, `brain_mri`, `other_xray`, `non_medical` | Frozen ImageNet ResNet-18 + logistic head |
| Quality gate | PASS / RETAKE + reasons (blur, glare, JPEG blockiness, resolution, exposure, perspective) | Thresholds fitted on PhotoShift-IN (simulated captures) |
| PHI redaction | Blurs every OCR-detected text line (≥2 chars) on the scan and the photo | PP-OCRv3 via `rapidocr-onnxruntime` |
| Plug-in `head_ct_ich_resnet18_rsna_v1` | ICH present/absent (any subtype) + Grad-CAM; severity `high` | RSNA ICH → external test CQ500 (India) |
| Plug-in `cxr_abnormal_xrv_densenet_v1` | Generic "abnormal chest X-ray" + top-3 of 18 TorchXRayVision pathology scores + Grad-CAM; severity `moderate` | XRV DenseNet-121 + logistic head on NIH/Shenzhen/Montgomery; tested incl. CheXphoto real phone photos |
| `brain_mri`, `other_xray` | **No plug-in** → routed `routine` for a human read | — |

### Routing / abstention (`router.py`, `plugins/base.py`)
Order: red-flag override → `refer_now`; `non_medical` → `rejected_not_medical`; quality RETAKE → `urgent_telerad` (AI not applied);
no plug-in → `routine`; plug-in abstains → `urgent_telerad`; positive & severity high → `refer_now`, else positive → `urgent_telerad`;
negative → `routine` (never "discharge"; adds a warning when < 4 CT slices were visible).
A plug-in abstains when p is inside its validated band `[lo, hi)`, when modality confidence < 0.80, when quality ≠ PASS,
or when `ctx["ood"]` is set (hook exists, nothing sets it yet).

### Existing hooks for text / LLM / reports
- **Structured JSON report** (`pipeline.run` → `report`): modality, quality reasons, plug-in result, routing + reasons, disclaimer. This is the natural, safe LLM input.
- **Plug-in registry** (`plugins/__init__.py`): any object with `name`, `applies_to`, `description`, `run(scan_gray, ctx)` returning the documented dict is a plug-in.
- **OCR recognition is already run** (`use_rec=True` in `redact._dnn_boxes`) but the recognised text is discarded — only boxes are kept. That text is a ready hook for "what text was on the image" (e.g. for PHI auditing or report-PDF reading) but must stay on-device.
- **No LLM, chat, or report-drafting code exists.** No WhatsApp integration exists (only WhatsApp-style JPEG handling).

---

## 2. Target inputs (MVP+)

| Input | Feasible for the hackathon? | Plan |
|---|---|---|
| WhatsApp-compressed JPEG ("photo" not "document") | ✅ already | Keep; add a real WhatsApp-recompressed test set if a labelled one becomes obtainable (CheXwhatsApp/WaCXR blocked — see DATA.md) |
| HEIC/HEIF (iPhone default) | ✅ trivial | `pillow-heif` → BGR in `load_any` |
| Multiple photos of one study (e.g. 3 photos of a CT film) | ✅ small | Accept a list; per-image gate; study-level p = max; route by worst case; report "slices seen" total |
| DICOM series (folder or `.zip`) | ✅ small/medium | Group by SeriesInstanceUID, sort by InstanceNumber/ImagePosition, run ICH on all slices (that is how CQ500 "full scan" was evaluated in EVAL.md); header PHI still dropped |
| Multi-frame DICOM | ✅ small | Iterate frames instead of middle-only |
| PDF radiology/lab report (scanned or digital) | ⚠️ text-only, no "diagnosis" | `pypdfium2` render → existing OCR (rec on) → on-device PHI scrub → optional local LLM structured extraction (values, units, flags vs printed reference range). No disease inference |
| Short phone video panning across a film | ⚠️ stretch | Sample frames, pick sharpest per film region via quality gate; *no* new model |
| ECG paper/photo | ⚠️ needs a new plug-in (see §3) | Modality class + quality gate + optional rhythm plug-in |
| Ultrasound stills | ⚠️ recognise-and-route only | Add `ultrasound` modality class so it is not forced into the wrong class; no finding plug-in in MVP |
| Chest/abdomen CT, mammography, fundus, dermatology, histopathology | ❌ out of MVP scope | Add as modality classes later so they **abstain cleanly** rather than mis-route |

Also needed for any new input: an **OOD guard** (set `ctx["ood"]` when the image is far from the plug-in's training distribution, e.g. Mahalanobis distance on the existing embeddings) so new/unknown inputs abstain.

---

## 3. Target plug-ins (public data, same gate, abstain-first)

All follow the existing contract: `lo`/`hi` from a validation split, abstain in between, external test set kept untouched,
Grad-CAM overlay, results written to EVAL.md only after evaluation.

| # | Plug-in | Modality | Public dataset(s) | Openness / licence (verify before redistribution) | Offline CPU? | Fit for rural India |
|---|---|---|---|---|---|---|
| 1 | **ICH subtype + mass effect / midline shift** (extends head-CT) | head_ct | RSNA ICH subtype labels (EDH/IPH/IVH/SAH/SDH — **already in `$ANYSCAN_DATA/rsna/slice_labels.csv`**); CQ500 `reads.csv` has subtypes, MassEffect, MidlineShift, Fracture (**already on disk**) | RSNA research terms; CQ500 CC BY-NC-SA 4.0 | ✅ same ResNet-18, multi-label head | ✅ trauma/stroke referral |
| 2 | **TB-focused CXR flag** | chest_xray | Shenzhen + Montgomery (on disk), TBX11K | NLM public; TBX11K research use | ✅ new head on existing XRV features | ✅✅ national TB programme priority |
| 3 | **ECG photo → rhythm triage (AF / STEMI-pattern flag)** | new `ecg` class | PTB-XL (21.8k 12-lead, signals) + `ecg-image-kit` (PhysioNet Challenge 2024) to render realistic paper/photo images | PTB-XL CC BY 4.0; ecg-image-kit open source | ✅ small CNN | ✅✅ ECGs are photographed and WhatsApp'd constantly |
| 4 | **Paediatric pneumonia flag** | chest_xray (paediatric) | Kermany et al. 2018 paediatric CXR (Mendeley) | CC BY 4.0 | ✅ | ✅ — but known shortcut-learning issues; must abstain outside paediatric |
| 5 | **Pneumothorax flag** (severity high → refer_now) | chest_xray | SIIM-ACR Pneumothorax | Kaggle competition / research terms | ✅ | ✅ trauma |
| 6 | **Extremity fracture flag** | other_xray | FracAtlas; GRAZPEDWRI-DX (paediatric wrist) | both CC BY 4.0 | ✅ | ✅ (MURA blocked — see DATA.md) |
| 7 | **Knee OA grade (KL) — routine-only** | other_xray | Knee OA severity dataset (Chen 2018, OAI-derived, Mendeley) | CC BY 4.0 (verify) | ✅ | ✅ but low urgency |
| 8 | **Brain MRI mass flag** | brain_mri | Nickparvar brain-tumour MRI (already used for modality) | CC0 per Kaggle page | ✅ | ⚠️ dataset has duplicates/leakage risk; must dedupe and keep "flag for read", not tumour typing |
| 9 | **Ultrasound: fetal head circumference / view check** | new `ultrasound` class | HC18 (Zenodo) | CC BY 4.0 (verify) | ✅ | ✅ antenatal, but stretch goal |
| 10 | **Dental panoramic findings** | new `dental_opg` class | DENTEX (MICCAI 2023) | CC BY-NC-SA 4.0 | ✅ | ⚠️ low triage value |

**Recommended next three:** (1) ICH subtype + mass effect/midline shift (data already on disk, same model family, raises clinical value of `refer_now`), (2) ECG photo plug-in (new, high-impact WhatsApp use case, open data), (3) TB-focused CXR flag (India priority, data already on disk). Fracture (6) is the best fourth.

Caveat for #1: CQ500 is currently the *external* ICH test set. Train subtype heads on RSNA only and keep CQ500 as the external test; mass-effect/midline-shift labels exist only in CQ500, so those would need patient-level cross-validation inside CQ500 and must be reported as *internal* results.

---

## 4. LLM / vLLM architecture

**Yes, integrate — but as a language layer on top of the gate, never as the diagnostician.**

```
image ─► AnyScan Gate (existing, deterministic) ─► structured JSON report (no PHI, no pixels)
                                                        │
              protocol docs (RAG: referral SOPs,       ▼
              national TB / stroke / trauma ──► local LLM (Ollama / llama.cpp on CPU; vLLM on GPU server)
              guidelines, local hospital list)          │  JSON-schema-constrained output
                                                        ▼
                         guardrail validator (facts must match JSON; banned claims; disclaimer)
                                                        ▼
         drafts: ① tele-radiology referral note  ② WhatsApp reply to health worker (English/Hindi/regional)
                 ③ "why retake" coaching text    ④ lab-PDF value extraction (text only)
                                       ── human reviews & sends ──
```

Rules
1. **Input = the JSON report** (+ optional on-device-scrubbed OCR text / clinician's typed notes). Not raw pixels, not DICOM headers, not un-redacted images.
2. **The LLM may not change** routing decision, p-values, findings or abstention. A validator rejects any draft whose routing/finding differs from the JSON or that contains phrases like "rules out", "normal study", "no need to refer", a diagnosis not in the JSON, or a missing disclaimer — and falls back to a deterministic template.
3. **Local by default.** Cloud APIs are opt-in, off by default, and only ever receive the PHI-free JSON.
4. **Temperature 0, JSON-schema / grammar-constrained decoding** (supported by llama.cpp grammars, Ollama `format`, vLLM guided decoding).
5. **Template fallback always exists**, so the product still works with no LLM installed (current behaviour).
6. Vision-language models are optional *second-reader captioners* only — their output is shown as "VLM caption (unvalidated)" and can only push a case toward human review, never toward `routine`.

Candidate open models (all run offline; sizes are parameter counts, not measured latency on this box)
| Role | Model | Licence | Where it runs |
|---|---|---|---|
| Report/WhatsApp drafting, RAG | Qwen2.5-3B/7B-Instruct | Apache-2.0 (3B: Qwen research licence — verify) | CPU via Ollama/llama.cpp (GGUF Q4); GPU via vLLM |
| Same, smallest | Llama 3.2 3B Instruct / Phi-4-mini / Gemma 3 4B | Llama community / MIT / Gemma terms | CPU |
| Indic-language replies | Sarvam-1 (2B) or translate output with IndicTrans2 | verify each | CPU/GPU |
| Medical VLM second reader, lab-report extraction | **MedGemma 1.5 4B-it** (released 2026-01-13; CXR/CT/MRI interpretation, lab-report extraction) | Health AI Developer Foundations terms (gated, research + commercial allowed with use restrictions) | GPU recommended (bf16); CPU only with quantisation and long latency |
| Embeddings for RAG | bge-small / multilingual-e5-small | MIT | CPU |

Deployment options
- **Edge / laptop / PHC PC (default):** Ollama or `llama-server` (llama.cpp) on CPU, OpenAI-compatible localhost API. Matches the current CPU-only design (this box: 8 cores, ~15 GB RAM, no GPU).
- **District hub with a GPU:** vLLM serving the same OpenAI-compatible API (higher throughput, guided JSON decoding, MedGemma in bf16). The app only needs a configurable `ANYSCAN_LLM_URL`; nothing else changes.
- **Cloud API:** not recommended; if ever enabled, only the PHI-free JSON is sent and the UI must say so.

---

## 5. Phased build order (hackathon demo first)

**Phase 0 – hygiene (½ day)**
- HEIC support; multi-frame / series-zip DICOM; multi-photo-per-study; OOD guard setting `ctx["ood"]`.
- Add `ecg`, `ultrasound` (and optionally `chest_ct`) as modality classes so they route/abstain instead of being forced into the nearest class.

**Phase 1 – language layer (1 day, highest demo value)**
- `anyscan/narrate.py`: deterministic template report + optional local LLM (Ollama/llama.cpp, OpenAI-compatible) with JSON schema + validator.
- Streamlit tab "Referral note / WhatsApp reply" (English + Hindi), copy-to-clipboard; human sends it. No auto-send.
- CLI `--narrate`. Tests: validator rejects contradicting drafts; app works with LLM absent.

**Phase 2 – plug-ins (1–2 days each)**
1. ICH subtypes (RSNA) + mass effect/midline shift (CQ500 CV) → richer `refer_now` reasons.
2. ECG photo plug-in (PTB-XL + ecg-image-kit renders + PhotoShift-style phone simulation).
3. TB-focused CXR head (Shenzhen/Montgomery/TBX11K).
Each: train → thresholds on validation → external test → append to EVAL.md (real numbers only) → sample + smoke test.

**Phase 3 – documents & stretch**
- PDF/lab-report text extraction (OCR + local LLM/MedGemma, values only, no interpretation beyond "outside printed reference range").
- Fracture plug-in; video frame picker; RAG over referral protocols.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| LLM hallucinates findings / reassures ("normal") | LLM sees only JSON; validator + banned-phrase list; template fallback; "draft — clinician must review" watermark |
| Users treat output as diagnosis | Keep NOT-FOR-CLINICAL-USE banner; router never outputs "discharge"/"normal"; language says "flag", "abstained", "needs read" |
| PHI leakage to cloud LLMs | Local-only default; only PHI-free JSON ever leaves the model boundary; OCR text scrubbed on-device; DICOM headers never read into reports |
| PHI leakage via recognised OCR text | Keep recognised text in memory only; never log/export unless scrubbed |
| Shortcut learning in public datasets (e.g. Kermany paediatric, brain-tumour MRI duplicates) | Dedupe by hash/patient; external test sets; abstain outside training distribution |
| Distribution shift to Indian phones/films | No public Indian phone-photo data exists (DATA.md); conservative quality gate; prospective, ethics-approved field calibration before any real use |
| Licence limits (CQ500, DENTEX = non-commercial; MedGemma HAI-DEF terms; RSNA research terms) | Track per-plug-in licence in `plugin_config.json` and README; non-commercial plug-ins flagged |
| Scope creep into "radiology suite" | Every new modality first ships as *recognise + quality-gate + route*; a finding plug-in is added only with a validated abstention band |
| Regulatory (CDSCO SaMD) | Remains a research prototype; any clinical pilot needs ethics approval and regulatory pathway |
