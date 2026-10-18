"""TotalSegmentator wrapper + NIfTI helpers (everything is converted to RAS+ canonical orientation)."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

import config

SEGMENT_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}


# ----------------------------------------------------------------------------- file helpers
def case_name_from_path(p) -> str:
    n = Path(p).name
    for ext in (".nii.gz", ".nii"):
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return Path(p).stem


def fingerprint(path) -> str:
    """Cheap content hash (first/last 8 MB + size) used as cache key."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(8 << 20))
        if size > (16 << 20):
            f.seek(-(8 << 20), 2)
            h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def prepare_input(src, case_dir: Path) -> Path:
    """Copy / convert the upload to <case_dir>/input.nii.gz and sanity-check it."""
    dst = case_dir / "input.nii.gz"
    if dst.exists():
        return dst
    src = Path(src)
    name = src.name.lower()
    if name.endswith(".nii.gz"):
        shutil.copyfile(src, dst)
    elif name.endswith(".nii"):
        nib.save(nib.load(str(src)), str(dst))
    else:
        raise ValueError("Unsupported file type. Please upload a NIfTI file (.nii or .nii.gz).")
    shape = tuple(int(s) for s in nib.load(str(dst)).shape if s != 1)
    if len(shape) != 3:
        dst.unlink(missing_ok=True)
        raise ValueError(f"Expected a 3D CT volume, got shape {nib.load(str(dst)).shape}.")
    return dst


def load_canonical(path):
    """Load NIfTI -> (data, zooms_mm, affine) in RAS+ orientation. 4D singleton dims are squeezed."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {data.shape}")
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    unit = img.header.get_xyzt_units()[0]
    zooms = zooms * {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(unit, 1.0)
    return data, zooms, img.affine


def _load_mask(path) -> np.ndarray:
    data, _, _ = load_canonical(path)
    return np.rint(data).astype(np.int16)


# ----------------------------------------------------------------------------- TotalSegmentator
def _label_to_segment_lut(task="liver_segments"):
    out = {}
    try:
        from totalsegmentator.map_to_binary import class_map
        for k, v in class_map[task].items():
            m = re.search(r"(\d+)$", str(v))
            if m:
                out[int(k)] = int(m.group(1))
    except Exception:
        pass
    return out or {i: i for i in range(1, 9)}


def run_totalsegmentator(input_path: Path, out_file: Path, task: str, log=print) -> Path:
    """Run one TotalSegmentator task as a single multilabel NIfTI. Skips if already computed."""
    if out_file.exists():
        log(f"   cached: {out_file.name}")
        return out_file
    try:  # workaround for "module 'sympy' has no attribute 'printing'" seen on some runs
        import sympy.printing  # noqa: F401
    except Exception:
        pass
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(
        input=str(input_path), output=str(out_file), task=task,
        ml=True, quiet=True, nr_thr_resamp=1, nr_thr_saving=1,
    )
    if not out_file.exists():
        raise RuntimeError(f"TotalSegmentator task '{task}' did not produce {out_file.name}")
    return out_file


def find_precomputed(case_name: str):
    """Look for outputs you already generated in the notebook (per-class files, statistics.json layout)."""
    root = config.PRECOMPUTED_SEG_ROOT
    if not root or not root.exists():
        return None
    for d in root.rglob(case_name):
        if d.is_dir() and (d / "liver_segments").is_dir() and (d / "liver_lesions").is_dir():
            return {"segments": d / "liver_segments", "lesions": d / "liver_lesions"}
    return None


def load_segment_labels(source: Path) -> np.ndarray:
    """-> int8 array, 0 = background, 1..8 = Couinaud segment."""
    source = Path(source)
    if source.is_file():
        raw = _load_mask(source)
        lut_map = _label_to_segment_lut()
        lut = np.zeros(max(int(raw.max()), 8) + 1, dtype=np.int8)
        for k, v in lut_map.items():
            if 0 <= k < len(lut):
                lut[k] = v
        return lut[np.clip(raw, 0, len(lut) - 1)]
    seg = None
    for k in range(1, 9):
        f = source / f"liver_segment_{k}.nii.gz"
        if f.exists():
            m = _load_mask(f) > 0
            if seg is None:
                seg = np.zeros(m.shape, dtype=np.int8)
            seg[m] = k
    if seg is None:
        raise FileNotFoundError(f"No liver_segment_*.nii.gz files in {source}")
    return seg


def load_lesion_mask(source: Path) -> np.ndarray:
    source = Path(source)
    if source.is_file():
        return _load_mask(source) > 0
    files = sorted(source.glob("*.nii.gz"))
    if not files:
        return None
    mask = None
    for f in files:
        m = _load_mask(f) > 0
        mask = m if mask is None else (mask | m)
    return mask
