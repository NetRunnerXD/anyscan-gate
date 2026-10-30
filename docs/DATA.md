# Data sources (all public; no patient data collected by this project)

| Dataset | Used for | How obtained | Licence / terms |
|---|---|---|---|
| **RSNA Intracranial Hemorrhage Detection** (2019 Kaggle challenge; 16-bit PNG re-packaging on HF `ianpan/rsna-intracranial-hemorrhage-16bit-png`) | ICH plug-in training (~44k slices, 1,096 patients) + validation (~6.9k slices, 179 patients). Head-CT class of modality model. | `scripts/download_data.py rsna 9` streams the first ~9 GB of the 144 GB tar (random subset of patients); `scripts/prep_rsna.py` converts to brain-window PNG | RSNA challenge data terms (research/non-commercial); HF card: MIT |
| **CQ500** (Chilamkurthy et al., *Lancet* 2018; Qure.ai / CARING, New Delhi) via HF `neurosentry/cq500` | **External test** for ICH (scan-level, 3-radiologist majority) and for head-CT modality recognition. Never used for training or threshold choice. | `scripts/download_data.py cq500 19` | CC BY-NC-SA 4.0 |
| **NIH ChestX-ray14**, batch `images_001` (4,999 images) via HF `alkzar90/NIH-Chest-X-ray-dataset` | CXR head calibration (50%) / test (50%, by patient); chest-X-ray class of modality model | `download_data.py cxr` | NIH CC0-like public release (cite Wang et al. 2017) |
| **Shenzhen** (662) and **Montgomery** (138) TB CXR sets, NLM | CXR head calibration / external test (50/50 by image) | `download_data.py cxr` (direct NLM PNG URLs) | Public, cite Jaeger et al. 2014 |
| **CheXphoto** validation (Phillips et al. 2020): 234 real OnePlus smartphone photos of CXRs on a monitor + synthetic transforms; Zenodo `12723955` (CheXwhatsApp team rehost) | **Real phone-photo test** for the CXR plug-in, modality model, quality gate; 30% of patients used to teach the modality model what monitor photos look like | `curl` from Zenodo (see README) | Rehost marked CC BY 4.0; original CheXpert/CheXphoto Stanford research-use terms |
| **CheXpert validation labels + digital originals** (HF `danjacobellis/chexpert`, validation parquet) | Ground truth (radiologist consensus) and digital counterpart for the CheXphoto photos | `curl` (see README) | Stanford CheXpert research-use agreement |
| Brain Tumor MRI Dataset (Nickparvar; HF `BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset`) | Brain-MRI class of the modality model only | `download_data.py misc` | CC0 per Kaggle page |
| Bone-fracture X-ray set (HF `Mahadih534/x-ray_bone-fracture-Dataset`, Roboflow export) | "other X-ray" class of the modality model only | `download_data.py misc` | per dataset card |
| Imagenette (fast.ai) via HF `johnowhitaker/imagenette2-320` | "non-medical" class of the modality model only | `download_data.py misc` | ImageNet terms (research) |
| TorchXRayVision `densenet121-res224-all` weights | CXR backbone (frozen) | auto-download on first use / `scripts/download_models.py` | Apache-2.0 code; weights trained on public CXR sets |
| ImageNet ResNet-18 (torchvision) | modality backbone (frozen) + ICH initialisation | auto-download | BSD |

## Not obtained / blocked

| Dataset | Why not used | Substitute |
|---|---|---|
| CheXphoto *train* (Stanford AIMI, ~43 GB split zip) | Size; the full set needs the Stanford research-use agreement | CheXphoto validation (234 real photos) used for testing only |
| CheXwhatsApp / WaCXR WhatsApp-compressed CXRs (Zenodo) | The CheXphoto-WhatsApp zip has no filename→label mapping (images renamed `IMG-…-WA….jpg`); WaCXR asks users to email the authors | WhatsApp-style JPEG recompression simulated in PhotoShift-IN |
| MURA (Stanford) | Requires Stanford registration | Public fracture X-ray set for the modality class only; no bone plug-in shipped |
| Full RSNA ICH (144 GB) | Disk/time on a CPU box | ~9 GB random-patient subset (51k slices) |
| Real Indian phone photos of films | None public | PhotoShift-IN simulated captures (clearly labelled as simulated) |

## PhotoShift-IN (generated here)
`scripts/evaluate.py quality` writes `$ANYSCAN_DATA/photoshift_in/` – simulated phone captures (lightbox/table/monitor,
perspective, glare, blur, WhatsApp-like JPEG, low-res, under-exposure, moiré) of **real held-out** public scans with a
`manifest.json` giving the source image, injected degradations, film quad and expected PASS/RETAKE verdict. It is a
simulation benchmark; it is not a substitute for real phone photos.
