"""Orchestrator ("agent"): runs the whole workflow step by step, logs progress, degrades gracefully per step."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import config
import measurements as M
import narrative
import radar_runner
import report
import segmentation as S
import viz


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def compare_studies(prior: dict, cur: dict) -> dict:
    ps, cs = prior["summary"], cur["summary"]

    def row(name, p, c, nd=1):
        d = None if (p is None or c is None) else c - p
        pct = None if (d is None or not p) else 100 * d / p
        return {"name": name, "prior": p, "current": c, "delta": d, "pct": pct, "nd": nd}

    rows = [
        row("Lesion count", ps["lesion_count"], cs["lesion_count"], 0),
        row("Total lesion volume (mm³)", ps["total_lesion_volume_mm3"], cs["total_lesion_volume_mm3"], 0),
        row("Largest lesion max diameter (mm)", ps["largest_max_diameter_mm"], cs["largest_max_diameter_mm"], 1),
        row("Largest axial long axis (mm)", ps["largest_axial_long_mm"], cs["largest_axial_long_mm"], 1),
        row("Mean lesion attenuation (HU, vol-weighted)", ps["mean_lesion_hu_weighted"], cs["mean_lesion_hu_weighted"], 1),
        row("Liver volume (mm³)", ps["liver_volume_mm3"], cs["liver_volume_mm3"], 0),
        row("Lesion burden (% of liver)", ps["lesion_burden_pct"], cs["lesion_burden_pct"], 2),
    ]
    v = rows[1]
    if v["pct"] is not None:
        word = "increased" if v["delta"] > 0 else "decreased" if v["delta"] < 0 else "was unchanged"
        sentence = (f"Compared with the prior study, total lesion volume {word}"
                    + ("" if v["delta"] == 0 else f" by {abs(v['pct']):.0f}% ({v['prior']:,.0f} → {v['current']:,.0f} mm³)")
                    + f"; lesion count {ps['lesion_count']} → {cs['lesion_count']}.")
    else:
        sentence = f"Prior study had {ps['lesion_count']} lesion(s) vs {cs['lesion_count']} now (volume change could not be computed)."
    return {"rows": rows, "sentence": sentence}


class LiverReportingAgent:
    """Events yielded by run(): ("log", str) ... then ("result", dict)."""

    def __init__(self):
        self.radar = radar_runner.RadarRunner()

    # ---------------------------------------------------------------- one study
    def _analyze_study(self, path, role, min_lesion_mm3):
        t0 = time.time()
        case_id = S.case_name_from_path(path)
        fp = S.fingerprint(path)
        case_dir = config.WORK_DIR / f"{case_id}_{fp}"
        case_dir.mkdir(parents=True, exist_ok=True)
        yield ("log", f"[{role}] Ingest: {case_id} (cache: {case_dir.name})")
        inp = S.prepare_input(path, case_dir)

        pre = S.find_precomputed(case_id)
        if pre:
            yield ("log", f"[{role}] Using precomputed TotalSegmentator outputs from {pre['segments'].parent}")
            seg_src, les_src = pre["segments"], pre["lesions"]
        else:
            yield ("log", f"[{role}] TotalSegmentator: liver_segments (Couinaud)... (minutes on CPU)")
            seg_src = S.run_totalsegmentator(inp, case_dir / "seg_liver_segments.nii.gz", "liver_segments", log=print)
            yield ("log", f"[{role}] TotalSegmentator: liver_lesions...")
            les_src = S.run_totalsegmentator(inp, case_dir / "seg_liver_lesions.nii.gz", "liver_lesions", log=print)

        yield ("log", f"[{role}] Loading volumes (RAS+ canonical orientation)")
        ct, zooms, affine = S.load_canonical(inp)
        ct = ct.astype(np.float32)
        seg = S.load_segment_labels(seg_src)
        lmask = S.load_lesion_mask(les_src)
        if lmask is None:
            lmask = np.zeros(ct.shape, bool)
        if not (ct.shape == seg.shape == lmask.shape):
            raise ValueError(f"Shape mismatch CT {ct.shape} / segments {seg.shape} / lesions {lmask.shape}")

        warn = []
        if ct.min() > -200:
            warn.append("CT intensity range does not look like Hounsfield units (min > -200); HU values may be unreliable.")
        if seg.max() == 0:
            warn.append("Liver segmentation is empty — the liver may be outside the field of view or the model failed.")

        yield ("log", f"[{role}] Measuring lesions (connected components, size, volume, HU, Couinaud location)")
        lesions, labels = M.analyze_lesions(ct, lmask, seg, zooms, min_lesion_mm3)
        liver = M.liver_statistics(ct, seg, lmask, zooms)
        summary = M.study_summary(lesions, liver)
        yield ("log", f"[{role}] → {summary['lesion_count']} lesion(s), total {summary['total_lesion_volume_mm3']:,.0f} mm³, liver {liver['liver_volume_mm3']:,.0f} mm³")

        viz.save_case_arrays(case_dir, ct, seg, labels, zooms)
        nib.save(nib.Nifti1Image(labels.astype(np.int16), affine), str(case_dir / "lesion_labels_RAS.nii.gz"))
        study = {
            "case_id": case_id, "fingerprint": fp, "case_dir": str(case_dir), "input": str(inp),
            "geometry": {"shape": list(ct.shape), "zooms": [float(z) for z in zooms]},
            "liver": liver, "lesions": lesions, "summary": summary, "warnings": warn,
            "seconds": round(time.time() - t0, 1),
        }
        return study

    # ---------------------------------------------------------------- full workflow
    def run(self, current_path, prior_path=None, run_radar=True, use_llm=False, min_lesion_mm3=config.MIN_LESION_MM3):
        t0 = time.time()
        notes = []
        try:
            cur = yield from self._analyze_study(current_path, "current", min_lesion_mm3)
        except Exception as ex:
            yield ("log", f"✗ Failed: {ex}\n{traceback.format_exc(limit=3)}")
            yield ("error", str(ex))
            return
        notes += cur["warnings"]

        prior = None
        if prior_path:
            try:
                prior = yield from self._analyze_study(prior_path, "prior", min_lesion_mm3)
            except Exception as ex:
                notes.append(f"Prior study could not be processed ({ex}); follow-up comparison skipped.")
                yield ("log", f"✗ Prior study failed: {ex}")

        # ---- RADAR
        radar = {"findings": None, "reason": "disabled by user"}
        if run_radar:
            cache = Path(cur["case_dir"]) / "radar_liver.json"
            try:
                if cache.exists():
                    radar = json.loads(cache.read_text())
                    yield ("log", "[RADAR] cached result")
                else:
                    findings = radar_runner.lookup_precomputed(cur["case_id"])
                    source = "precomputed CSV"
                    if findings is None:
                        ok, msg = self.radar.available()
                        if not ok:
                            raise RuntimeError(msg)
                        yield ("log", "[RADAR] running inference... (can take several minutes on CPU)")
                        findings = self.radar.run(Path(cur["input"]), Path(cur["case_dir"]), cur["case_id"],
                                                  log=print)
                        source = "RADAR inference"
                    radar = {"findings": findings, "source": source}
                    cache.write_text(json.dumps(radar))
                    yield ("log", f"[RADAR] done ({source})")
                msg = radar_runner.crosscheck(cur["summary"]["lesion_count"], radar["findings"])
                if msg:
                    notes.append(msg)
            except Exception as ex:
                if isinstance(ex, ModuleNotFoundError):
                    msg = (f"RADAR skipped: missing package '{ex.name}'. In the same environment as the app run: "
                           f"pip install -r \"{config.RADAR_REPO_DIR / 'requirements.txt'}\"")
                else:
                    msg = f"RADAR skipped: {ex}"
                radar = {"findings": None, "reason": msg}
                notes.append(msg)
                yield ("log", f"[RADAR] ✗ {msg}")

        res = {
            "case_id": cur["case_id"], "current": cur, "prior": prior, "radar": radar, "notes": notes,
            "params": {"min_lesion_mm3": min_lesion_mm3, "use_llm": use_llm, "run_radar": run_radar},
        }
        if prior:
            res["comparison"] = compare_studies(prior, cur)

        yield ("log", "[Report] drafting impression + rendering overlays")
        res["impression"], res["impression_source"] = narrative.draft_impression(res, use_llm)
        key_images = viz.key_lesion_images(cur["case_dir"], cur["lesions"])
        res["elapsed_s"] = time.time() - t0
        html_str = report.build_report_html(res, key_images)

        d = Path(cur["case_dir"])
        (d / "report.html").write_text(html_str, encoding="utf-8")
        (d / "report.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
        rows = [{k: (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in l.items()} for l in cur["lesions"]]
        pd.DataFrame(rows).to_csv(d / "lesions.csv", index=False)

        res["report_html"] = html_str
        res["key_images"] = key_images
        res["files"] = [str(d / n) for n in ("report.html", "report.json", "lesions.csv", "lesion_labels_RAS.nii.gz")
                        if (d / n).exists()]
        yield ("log", f"✓ Done in {res['elapsed_s']:.0f}s — outputs in {d}")
        yield ("result", res)
