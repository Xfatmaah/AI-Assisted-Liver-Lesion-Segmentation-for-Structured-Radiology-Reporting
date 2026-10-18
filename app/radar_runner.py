"""RADAR wrapper (same calls as RADAR_Pipeline notebook) + liver-only parsing."""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from pathlib import Path

import pandas as pd

import config

LESION_KEYWORDS = ("lesion", "mass", "metast", "cyst", "hemangioma", "abscess", "carcinoma", "nodule", "tumor")


def english_name(col: str) -> str:
    m = re.search(r"\(([^)]+)\)", col)
    return m.group(1) if m else col


def _liver_findings_from_row(row: pd.Series) -> list[dict]:
    out = []
    for col, val in row.items():
        if col == "file_name":
            continue
        name = english_name(str(col))
        if name.startswith("Liver_"):
            out.append({"finding": name.split("_", 1)[1], "probability": float(val)})
    return sorted(out, key=lambda d: -d["probability"])


def lookup_precomputed(case_name: str):
    """Reuse a row from CSVs you already generated with the RADAR notebook."""
    for csv in config.PRECOMPUTED_RADAR_CSVS:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        hit = df[df["file_name"].astype(str).str.replace(r"\.nii(\.gz)?$", "", regex=True) == case_name]
        if len(hit):
            return _liver_findings_from_row(hit.iloc[0])
    return None


class RadarRunner:
    _lock = threading.Lock()
    _init_returns = None
    _mod = None

    def available(self):
        inf = config.RADAR_REPO_DIR / "RADAR_inference"
        ckpt = config.RADAR_REPO_DIR / "ckpt"
        if not inf.is_dir():
            return False, f"RADAR repo not found at {config.RADAR_REPO_DIR} (set RADAR_REPO_DIR)"
        if not (ckpt / "checkpoint_radar_pretrain.pth").exists():
            return False, f"RADAR checkpoint missing in {ckpt}"
        return True, "ok"

    def _ensure_loaded(self, log):
        if RadarRunner._init_returns is not None:
            return
        ckpt = config.RADAR_REPO_DIR / "ckpt"
        inf = config.RADAR_REPO_DIR / "RADAR_inference"
        # MODEL_ROOT / CONFIGS_ROOT must be set BEFORE importing inference_demo
        os.environ["MODEL_ROOT"] = str(ckpt)
        os.environ["CONFIGS_ROOT"] = str(ckpt)
        if str(inf) not in sys.path:
            sys.path.insert(0, str(inf))
        log("   loading RADAR model into memory (first time only)...")
        import inference_demo  # noqa: E402
        RadarRunner._mod = inference_demo
        RadarRunner._init_returns = inference_demo.initialize()

    def run(self, input_nii_gz: Path, work_dir: Path, case_name: str, log=print) -> list[dict]:
        """Returns the ranked liver findings [{'finding','probability'}]."""
        with self._lock:
            self._ensure_loaded(log)
            in_dir = work_dir / "radar_in"
            res_dir = work_dir / "radar_out"
            in_dir.mkdir(parents=True, exist_ok=True)
            res_dir.mkdir(parents=True, exist_ok=True)
            dst = in_dir / "case.nii.gz"
            if not dst.exists():
                shutil.copyfile(input_nii_gz, dst)
            self._mod.inference(self._init_returns, str(in_dir), str(res_dir), "case")
            csv = res_dir / "RADAR_infer_results_case.csv"
            if not csv.exists():
                raise RuntimeError("RADAR finished but no result CSV was produced (check RADAR_errors_case.csv)")
            df = pd.read_csv(csv)
            return _liver_findings_from_row(df.iloc[0])


def crosscheck(n_lesions: int, findings: list[dict]) -> str | None:
    """Flag disagreement between segmentation (lesion count) and RADAR lesion-type probabilities."""
    lesion_like = [f["probability"] for f in findings if any(k in f["finding"].lower() for k in LESION_KEYWORDS)]
    if not lesion_like:
        return None
    mx = max(lesion_like)
    if n_lesions > 0 and mx < 0.3:
        return (f"Discordance: segmentation found {n_lesions} lesion(s) but all RADAR lesion-type liver findings "
                f"are low (max {mx:.2f}). Verify the segmentation overlay carefully.")
    if n_lesions == 0 and mx >= config.RADAR_POSITIVE_THRESHOLD:
        return (f"Discordance: no lesion segmented but RADAR lesion-type liver finding probability is {mx:.2f}. "
                f"Possible missed (small/subtle) lesion — review the images.")
    return None
