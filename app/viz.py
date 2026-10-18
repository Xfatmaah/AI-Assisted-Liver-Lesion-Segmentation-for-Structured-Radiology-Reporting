"""Slice rendering with segmentation overlay (axial / coronal / sagittal), radiological display convention."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
SEG_COLORS = np.array([
    (0, 0, 0),
    (155, 89, 182), (52, 152, 219), (26, 188, 156), (46, 204, 113),
    (241, 196, 15), (230, 126, 34), (0, 188, 212), (121, 85, 72),
], dtype=np.float32)
LESION_COLOR = np.array((255, 30, 30), dtype=np.float32)
DEFAULT_WINDOW = "CT-Abdomen (W350 / L40)"   # same as the 3D Slicer "CT-Abdomen" window/level preset
WINDOWS = {
    "CT-Abdomen (W350 / L40)": (40, 350),
    "Liver (W250 / L60)": (60, 250),
    "Soft tissue (W400 / L40)": (40, 400),
    "Wide (W700 / L50)": (50, 700),
}

LEGEND_HTML = "<div style='display:flex;flex-wrap:wrap;gap:10px;font-size:13px'>" + "".join(
    f"<span><span style='display:inline-block;width:12px;height:12px;background:rgb{tuple(int(c) for c in SEG_COLORS[i+1])};"
    f"border-radius:2px;margin-right:4px'></span>Segment {ROMAN[i]}</span>" for i in range(8)
) + "<span><span style='display:inline-block;width:12px;height:12px;border:2px solid red;border-radius:2px;margin-right:4px'></span>Lesion (L#)</span></div>"


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------- case cache on disk
@lru_cache(maxsize=2)
def load_case(case_dir: str):
    d = Path(case_dir)
    meta = json.loads((d / "meta.json").read_text())
    return {
        "ct": np.load(d / "ct_i16.npy", mmap_mode="r"),
        "seg": np.load(d / "seg.npy", mmap_mode="r"),
        "les": np.load(d / "lesions.npy", mmap_mode="r"),
        "zooms": np.array(meta["zooms"], float),
        "crop": meta.get("crop"),
    }


def save_case_arrays(case_dir: Path, ct, seg, lesions, zooms, margin_mm=15):
    np.save(case_dir / "ct_i16.npy", np.clip(np.rint(ct), -32768, 32767).astype(np.int16))
    np.save(case_dir / "seg.npy", seg.astype(np.int8))
    np.save(case_dir / "lesions.npy", lesions.astype(np.int16))
    crop = None
    roi = (seg > 0) | (lesions > 0)
    if roi.any():
        crop = []
        for ax in range(3):
            other = tuple(a for a in range(3) if a != ax)
            idx = np.where(roi.any(axis=other))[0]
            m = int(round(margin_mm / zooms[ax]))
            crop.append([int(max(idx.min() - m, 0)), int(min(idx.max() + m + 1, roi.shape[ax]))])
    (case_dir / "meta.json").write_text(json.dumps({"zooms": list(map(float, zooms)), "shape": list(ct.shape), "crop": crop}))
    load_case.cache_clear()


def n_slices(case_dir: str, plane: str) -> int:
    ct = load_case(case_dir)["ct"]
    return ct.shape[{"axial": 2, "coronal": 1, "sagittal": 0}[plane]]


def default_index(case_dir: str, plane: str) -> int:
    c = load_case(case_dir)
    ax = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
    roi = (np.asarray(c["seg"]) > 0) | (np.asarray(c["les"]) > 0)
    if roi.any():
        other = tuple(a for a in range(3) if a != ax)
        idx = np.where(roi.any(axis=other))[0]
        return int((idx.min() + idx.max()) // 2)
    return c["ct"].shape[ax] // 2


# ----------------------------------------------------------------------------- rendering
def _plane2d(vol, plane, idx, crop):
    cx, cy, cz = (slice(*c) if crop else slice(None) for c in (crop or [None] * 3))
    if plane == "axial":
        a = vol[cx, cy, idx]
    elif plane == "coronal":
        a = vol[cx, idx, cz]
    else:
        a = vol[idx, cy, cz]
    return np.asarray(a).T[::-1, ::-1]


def render_slice(case_dir: str, plane="axial", idx=0, window=DEFAULT_WINDOW, show_seg=True, show_les=True,
                 zoom_liver=True, out_px=900, highlight=None, opacity=0.3) -> Image.Image:
    c = load_case(case_dir)
    ct, seg, les, z = c["ct"], c["seg"], c["les"], c["zooms"]
    crop = c["crop"] if zoom_liver else None
    ax = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
    idx = int(np.clip(idx, 0, ct.shape[ax] - 1))

    g = np.ascontiguousarray(_plane2d(ct, plane, idx, crop), dtype=np.float32)
    s = _plane2d(seg, plane, idx, crop).astype(np.uint8)
    l = _plane2d(les, plane, idx, crop).astype(np.int32)

    lvl, wid = WINDOWS.get(window, WINDOWS[DEFAULT_WINDOW])
    row_sp, col_sp = {"axial": (z[1], z[0]), "coronal": (z[2], z[0]), "sagittal": (z[2], z[1])}[plane]
    h, w = g.shape
    scale = out_px / max(h * row_sp, w * col_sp)
    W, H = max(int(round(w * col_sp * scale)), 2), max(int(round(h * row_sp * scale)), 2)

    # resample the raw HU values first (bicubic), THEN apply window/level -> smooth, sharp CT
    gi = np.asarray(Image.fromarray(g, "F").resize((W, H), Image.BICUBIC), dtype=np.float32)
    gi = np.clip((gi - (lvl - wid / 2)) / wid, 0, 1) * 255
    si = np.asarray(Image.fromarray(s, "L").resize((W, H), Image.NEAREST))
    li = np.asarray(Image.fromarray(l, "I").resize((W, H), Image.NEAREST))

    rgb = np.repeat(gi[..., None], 3, axis=2)
    if show_seg and opacity > 0:
        m = si > 0
        a = opacity * 0.6
        rgb[m] = rgb[m] * (1 - a) + SEG_COLORS[si[m]] * a
    if show_les:
        m = li > 0
        a = opacity * 0.8
        rgb[m] = rgb[m] * (1 - a) + LESION_COLOR * a
        edge = m & ~ndi.binary_erosion(m)
        rgb[edge] = LESION_COLOR
        if highlight:
            hm = li == highlight
            e2 = ndi.binary_dilation(hm & ~ndi.binary_erosion(hm), iterations=1)
            rgb[e2] = (255, 255, 0)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f = _font(15)
    if show_les:
        for lid in np.unique(li[li > 0]):
            rr, cc = np.nonzero(li == lid)
            d.text((cc.mean() + 8 + (cc.max() - cc.min()) / 2, rr.mean() - 8), f"L{lid}", fill=(255, 255, 0), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    tl, tr, tt = {"axial": ("R", "L", "A"), "coronal": ("R", "L", "S"), "sagittal": ("A", "P", "S")}[plane]
    for pos, txt in (((6, H // 2 - 8), tl), ((W - 16, H // 2 - 8), tr), ((W // 2 - 4, 4), tt)):
        d.text(pos, txt, fill=(255, 220, 80), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((6, H - 22), f"{plane.capitalize()}  {idx + 1}/{ct.shape[ax]}", fill=(255, 255, 255), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def key_lesion_images(case_dir: str, lesions: list[dict], max_n=8):
    out = []
    for l in lesions[:max_n]:
        img = render_slice(case_dir, "axial", l["axial_slice"], highlight=l["id"], out_px=700)
        cap = (f"L{l['id']} · {l['primary_segment_label']} · {l['axial_long_mm']:.1f}×{l['axial_short_mm']:.1f} mm · "
               f"{l['volume_mm3']:,.0f} mm³")
        out.append((img, cap))
    return out
