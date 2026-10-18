"""Draft impression: deterministic template, optionally rephrased by an LLM (never allowed to add facts)."""
from __future__ import annotations

import json
import os

import config


def template_impression(res: dict) -> str:
    cur = res["current"]
    s, liver, lesions = cur["summary"], cur["liver"], cur["lesions"]
    lines = []
    if not s["lesion_present"]:
        lines.append(
            "No focal liver lesion was segmented by the automated model on this CT. "
            f"Estimated liver volume {liver['liver_volume_mm3']:,.0f} mm³. Note: the reference model detected only about "
            f"{config.REFERENCE_SENSITIVITY_CT}% of lesions on CT, so small or subtle lesions may be missed; visual review is required."
        )
    else:
        n = s["lesion_count"]
        burden = f" ({s['lesion_burden_pct']:.2f}% of liver volume)" if s["lesion_burden_pct"] is not None else ""
        lines.append(
            f"Automated segmentation identified {n} liver lesion{'s' if n > 1 else ''} with a total volume of "
            f"{s['total_lesion_volume_mm3']:,.0f} mm³{burden}. Liver volume {liver['liver_volume_mm3']:,.0f} mm³."
        )
        from segmentation import SEGMENT_ROMAN
        for l in lesions[:5]:
            others = [SEGMENT_ROMAN[k] for k in l["segments_involved"] if k != l["primary_segment"]]
            span = f" (also involving {', '.join(others)})" if others else ""
            lines.append(
                f"L{l['id']}: {l['primary_segment_label']}{span}, {l['axial_long_mm']:.1f} × {l['axial_short_mm']:.1f} mm axial, "
                f"{l['volume_mm3']:,.0f} mm³, mean {l['mean_hu']:.0f} HU."
            )
        if n > 5:
            lines.append(f"{n - 5} additional smaller lesion(s) are listed in the table.")
    radar = res.get("radar")
    if radar and radar.get("findings"):
        pos = [f for f in radar["findings"] if f["probability"] >= config.RADAR_POSITIVE_THRESHOLD]
        if pos:
            lines.append("RADAR liver findings above threshold: " + ", ".join(f"{f['finding']} ({f['probability']:.2f})" for f in pos[:5]) + ".")
        else:
            lines.append("RADAR did not flag any liver finding above the probability threshold.")
    cmp_ = res.get("comparison")
    if cmp_:
        lines.append(cmp_["sentence"])
    return " ".join(lines)


def draft_impression(res: dict, use_llm: bool):
    base = template_impression(res)
    if not use_llm or not os.getenv("ANTHROPIC_API_KEY"):
        return base, "template"
    try:
        import anthropic
        data = {
            "summary": res["current"]["summary"],
            "liver_volume_mm3": res["current"]["liver"]["liver_volume_mm3"],
            "lesions": [{k: l[k] for k in ("id", "primary_segment_label", "segments_involved_label", "axial_long_mm", "axial_short_mm",
                                          "max_diameter_3d_mm", "volume_mm3", "mean_hu")} for l in res["current"]["lesions"][:15]],
            "radar_liver_findings": (res.get("radar") or {}).get("findings", [])[:8],
            "follow_up": (res.get("comparison") or {}).get("sentence"),
        }
        msg = anthropic.Anthropic().messages.create(
            model=config.LLM_MODEL, max_tokens=500,
            system=("You draft the IMPRESSION section of a liver CT report from structured AI-tool output. "
                    "Use ONLY the numbers and facts provided; never add findings, diagnoses, or recommendations. "
                    "Use hedged language ('automated analysis suggests'), state that RADAR values are model probabilities, "
                    "and keep it under 130 words. Plain text, no markdown."),
            messages=[{"role": "user", "content": json.dumps(data)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return (text or base), ("llm" if text else "template")
    except Exception as e:  # any failure -> safe fallback
        return base, f"template (LLM unavailable: {type(e).__name__})"
