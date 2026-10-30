"""Render docs/eval.json -> docs/EVAL.md"""
import json, sys, datetime


def f(v):
    return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def row(name, m, note=""):
    ci = m.get("auc_95ci")
    auc = f(m.get("auc")) + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")
    return (f"| {name} | {m['n']} ({m['n_pos']} pos) | {auc} | {f(m.get('sens_at_hi'))} / {f(m.get('spec_at_hi'))} | "
            f"{f(m.get('sens_at_lo'))} / {f(m.get('spec_at_lo'))} | {f(m.get('coverage_non_abstained'))} | "
            f"{f(m.get('sens_non_abstained'))} / {f(m.get('spec_non_abstained'))} | {note} |")


HDR = ("| Test set | n | AUC (95% CI) | Sens / Spec @ hi (flag) | Sens / Spec @ lo | Coverage (not abstained) | Sens / Spec when not abstaining | Notes |\n"
       "|---|---|---|---|---|---|---|---|")


def write(R, path):
    L = ["# AnyScan Gate – Evaluation on real held-out data", "",
         f"_Generated {datetime.date.today().isoformat()} by `python scripts/evaluate.py`. Raw numbers: `docs/eval.json`._", "",
         "> **NOT FOR CLINICAL USE.** These are retrospective research metrics on public datasets. They are not a clinical validation, "
         "and performance on Indian district-hospital films/phones is unknown until prospectively tested under ethics approval.", "",
         "**How to read the plug-in tables.** Each plug-in has two operating points chosen on *validation* data (never on the test sets): "
         "`lo` (high-sensitivity point) and `hi` (high-specificity point). p ≥ hi → flagged; p < lo → \"no finding\" (still routine human read); "
         "lo ≤ p < hi → **abstain** (routed to urgent human read). *Coverage* = share of images where the plug-in did not abstain.", ""]
    if "ich" in R:
        I = R["ich"]; m = I["model"]
        L += ["## 1. Head CT – intracranial haemorrhage plug-in", "",
              f"Model: ResNet-18 (ImageNet init) fine-tuned on **{m['train_slices']:,} real RSNA ICH slices** (brain window), "
              f"{m['epochs']} epochs on CPU. Operating points lo={I['operating_points']['lo']:.3f}, hi={I['operating_points']['hi']:.3f} "
              f"({I['operating_points'].get('rule')}).", "",
              "Labels: RSNA = per-slice expert labels (any haemorrhage); **CQ500 (Qure.ai / CARING, New Delhi, India) = scan-level majority vote of 3 radiologists (R1–R3 `ICH`)**. "
              "CQ500 was never used for training or threshold selection → external test.", "", HDR,
              row("RSNA held-out patients – per slice", I["rsna_heldout_slices"], "internal, patient-level split"),
              row("RSNA held-out patients – per series (max)", I["rsna_heldout_series"], "internal"),
              row("**CQ500 – full scan (max over all slices)**", I["cq500_all_slices"], "external, India"),
              row("CQ500 – 4×4 film montage, digital", I["cq500_film_montage_digital"], "16 slices like a printed film"),
              row("CQ500 – same film, simulated phone photo", I["cq500_film_montage_phone_sim"], "PhotoShift-IN rendering → full front-end"),
              ]
        if I.get("cq500_phone_sim_pass_only"):
            L.append(row("CQ500 – phone sim, quality-gate PASS only", I["cq500_phone_sim_pass_only"], ""))
        L += ["", f"Digital→phone decision flip rate on CQ500 films: **{I['cq500_flip_rate_digital_vs_phone']:.3f}**. Scans evaluated: {I['cq500_n_scans']}.", ""]
    if "cxr" in R:
        C = R["cxr"]
        L += ["## 2. Chest X-ray – abnormal vs normal plug-in", "",
              "Backbone: TorchXRayVision DenseNet-121 `densenet121-res224-all` (public weights; trained on NIH, PadChest, CheXpert-train, MIMIC, Google, OpenI, RSNA-pneumonia). "
              "Head: logistic regression on its 18 outputs, fitted on the *calibration half* of NIH-batch-001 + Shenzhen + Montgomery. "
              f"Operating points lo={C['operating_points']['lo']:.3f}, hi={C['operating_points']['hi']:.3f} ({C['operating_points']['rule']}).", "", HDR]
        names = {"shenzhen_tb_test": ("Shenzhen TB set (China) – held-out half", "external to backbone; abnormal = TB"),
                 "montgomery_tb_test": ("Montgomery TB set (USA) – held-out half", "external to backbone; abnormal = TB"),
                 "tb_sets_combined_test": ("Shenzhen + Montgomery combined", "external to backbone"),
                 "nih_cxr14_test": ("NIH ChestX-ray14 batch 001 – held-out patients", "⚠ backbone saw NIH in pre-training → optimistic"),
                 "chexpert_valid_digital": ("**CheXpert validation – digital originals**", "radiologist-consensus labels; external"),
                 "chexphoto_real_phone_photos": ("**CheXphoto – REAL smartphone photos of the same films**", "OnePlus photos of monitor → full front-end"),
                 "chexphoto_synthetic_photographic": ("CheXphoto – synthetic photographic transforms", ""),
                 "chexphoto_real_photos_PASS_only": ("CheXphoto real photos – gate PASS", ""),
                 "chexphoto_real_photos_RETAKE_only": ("CheXphoto real photos – gate RETAKE", "")}
        for k, (n, note) in names.items():
            if k in C and C[k]:
                L.append(row(n, C[k], note))
        L += ["", f"- Digital→real-phone-photo decision flip rate (CheXpert valid vs CheXphoto): **{C['chexphoto_flip_rate_digital_vs_real_photo']:.3f}**",
              "- ⚠ CheXpert-valid frontal films are mostly abnormal (only ~20 'No Finding' cases in the 70% held-out patients), so CIs are wide; "
              "the higher AUC on photos than on the 320-px digital originals is within overlapping CIs and should not be read as photos being better.",
              f"- Real phone photos passing the quality gate: {C['chexphoto_real_photo_quality_pass_rate']:.3f}; recognised as chest X-ray by the modality model: {C['chexphoto_real_photo_modality_correct']:.3f}", ""]
    if "modality" in R:
        M = R["modality"]
        L += ["## 3. Modality / region classifier", "",
              "Frozen ImageNet ResNet-18 + logistic head. Test images are disjoint from training; head-CT test images come from **CQ500 (external)**.", "",
              "| Condition | n | Accuracy | " + " | ".join(f"recall {c}" for c in M["classes"]) + " | share with conf < 0.8 |",
              "|---|---|---|" + "---|" * len(M["classes"]) + "---|"]
        for mode in ("digital", "phone_sim"):
            d = M[mode]
            L.append(f"| {mode} | {d['n']} | {d['accuracy']:.3f} | " + " | ".join(f"{d['per_class_recall'][c]:.3f}" for c in M["classes"]) + f" | {d['frac_below_0.8_conf']:.3f} |")
        L += ["", f"CQ500 film montages, simulated phone photo → classified head CT: {M['cq500_montage_phone_sim_head_ct_recall']:.3f}", "",
              "Sources: " + "; ".join(f"{k}: {v}" for k, v in M["sources"].items()), ""]
    if "quality" in R:
        Q = R["quality"]
        L += ["## 4. Capture-quality gate (PASS / RETAKE)", "",
              "Scored on the **PhotoShift-IN test split**: simulated phone captures (lightbox / table / monitor; blur, glare, WhatsApp-style JPEG, low-res, under-exposure) "
              "of *real* held-out scans, each with a known expected verdict. Thresholds were fitted on a disjoint calibration split. "
              "No public dataset has human PASS/RETAKE labels for real phone photos, so this is a *simulation* benchmark.", "",
              f"- Agreement with expected verdict: **{Q['agreement']:.3f}** (n={Q['n']}); RETAKE recall {Q['retake_recall']:.3f}; PASS recall {Q['pass_recall']:.3f}",
              "- RETAKE rate by injected degradation: " + ", ".join(f"`{k}` {v:.2f}" for k, v in Q["retake_rate_by_degradation"].items()), ""]
    if "phi" in R:
        P = R["phi"]
        L += ["## 5. PHI redaction", "", P["note"], "",
              f"- Text-line recall (≥60% of line area blurred), digital: **{P['recall_digital']:.3f}**; simulated phone photo: **{P['recall_phone_sim']:.3f}** ({P['n_text_lines']} lines on {P['n_images']} images)",
              f"- Images with *every* PHI line redacted: digital {P['images_with_all_lines_redacted_digital']:.3f}; phone {P['images_with_all_lines_redacted_phone_sim']:.3f}",
              f"- Over-redaction (blurred area outside PHI): {P['over_redaction_area_frac_mean']:.3f} of image; blurred area on clean images: {P['blurred_area_frac_on_clean_images_mean']:.3f}", ""]
    if "film" in R:
        F = R["film"]
        L += ["## 6. Film / screen detection", "", F["note"], "",
              f"- Detected: {F['detected']:.3f}; mean IoU {F['iou_mean']:.3f}; IoU > 0.8: {F['iou_gt_0.8']:.3f} (n={F['n']})", ""]
    if "runtime" in R:
        T = R["runtime"]
        L += ["## 7. Runtime", "", f"- {T['seconds_per_image_mean']} s per image on {T['cpu']} CPU cores, no GPU ({T['note']}).", ""]
    L += ["## What these numbers do *not* show", "",
          "- No prospective or Indian-PHC phone-photo validation; CheXphoto photos are of a US monitor with one phone model.",
          "- CQ500 labels are scan-level; a film photo shows only some slices, so a bleed outside the photographed slices cannot be detected.",
          "- The CXR head's 'abnormal' mixes TB (Shenzhen/Montgomery) and NIH findings; disease-specific performance (e.g. silicosis) is untested.",
          "- Quality-gate and PHI metrics use simulated degradations / synthetic text on real images.", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    write(json.load(open(sys.argv[1])), sys.argv[2])
