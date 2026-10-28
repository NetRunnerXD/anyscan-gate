"""Routing: refer_now / urgent_telerad / routine. Conservative by design: the AI can escalate
a case, and a confident negative only keeps it in the routine human-read queue (never 'discharge')."""


def route(quality, cls, plugin_res, red_flags=False):
    if red_flags:
        return "refer_now", ["Clinical red flags reported by health worker override image triage."]
    if cls["label"] == "non_medical":
        return "rejected_not_medical", ["Not recognised as an X-ray/CT/MRI image; nothing triaged. Upload a photo of a film or scan screen."]
    if quality["verdict"] == "RETAKE":
        return "urgent_telerad", ["Capture failed quality checks: RETAKE first. If a retake is impossible, a human must read it urgently (AI not applied)."]
    if plugin_res is None:
        return "routine", [f"No AI plug-in for {cls['label']}; queued for standard teleradiology read."]
    if plugin_res["status"] == "abstain":
        return "urgent_telerad", ["AI abstained (uncertain) - needs a human read soon."] + plugin_res["abstain_reasons"]
    if plugin_res["positive"]:
        if plugin_res["severity"] == "high":
            return "refer_now", [f"{plugin_res['finding']} (p={plugin_res['p_abnormal']:.2f}) - time-critical; refer to a CT-capable/neurosurgical centre now and send for urgent read."]
        return "urgent_telerad", [f"{plugin_res['finding']} (p={plugin_res['p_abnormal']:.2f}) - urgent radiologist confirmation."]
    why = ["Good capture; plug-in found no abnormality with adequate confidence.",
           "Routine read still required - the AI cannot rule out disease."]
    n = (plugin_res.get("features") or {}).get("n_slices_on_image")
    if n is not None and n < 4:
        why.append(f"Only {n} CT slice(s) were assessed - a bleed on other slices cannot be excluded; send the full scan/film.")
    return "routine", why
