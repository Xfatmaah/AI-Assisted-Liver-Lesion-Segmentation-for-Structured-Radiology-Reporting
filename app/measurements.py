"""Per-lesion measurements: connected components, size, volume, HU, Couinaud location."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

import config
from segmentation import SEGMENT_ROMAN

LOBE = {1: "Caudate", 2: "Left lobe", 3: "Left lobe", 4: "Left lobe",
        5: "Right lobe", 6: "Right lobe", 7: "Right lobe", 8: "Right lobe"}


def _extent_along(pts: np.ndarray, u: np.ndarray, zooms: np.ndarray) -> float:
    """Centre-to-centre extent of the voxel cloud along unit vector u (PyRadiomics convention).
    Floored at the smallest voxel spacing so a 1-voxel-thick object never reports 0 mm."""
    proj = pts @ u
    return float(max(proj.max() - proj.min(), zooms.min()))


def _longest_axis(pts: np.ndarray, zooms: np.ndarray):
    """Maximum Feret diameter (mm, centre-to-centre) and its unit direction."""
    n = len(pts)
    if n < 2:
        return float(np.cbrt(np.prod(zooms))), np.eye(len(zooms))[0]
    cand = pts
    if n >= len(zooms) + 2:
        try:
            cand = pts[ConvexHull(pts).vertices]
        except Exception:
            cand = pts
    if len(cand) > 2000:
        cand = cand[np.random.default_rng(0).choice(len(cand), 2000, replace=False)]
    d = cdist(cand, cand)
    i, j = np.unravel_index(int(d.argmax()), d.shape)
    if d[i, j] == 0:
        return float(np.cbrt(np.prod(zooms))), np.eye(len(zooms))[0]
    u = (cand[j] - cand[i]) / d[i, j]
    return _extent_along(pts, u, zooms), u


def _lesion_metrics(sub, offset, ct_sub, seg_sub, zooms, vox_mm3):
    coords = np.argwhere(sub) + offset
    nvox = len(coords)
    pts = coords * zooms

    # --- axis-aligned bounding box (RAS axes: x=left-right, y=anterior-posterior, z=superior-inferior)
    bbox = (coords.max(0) - coords.min(0) + 1) * zooms

    # --- max 3D diameter (hull of boundary voxels)
    bnd = sub & ~ndi.binary_erosion(sub)
    bpts = (np.argwhere(bnd if bnd.any() else sub) + offset) * zooms
    d3, _ = _longest_axis(bpts, zooms)

    # --- axial long / short axis on the slice with the largest cross-section (RECIST-style)
    zs = coords[:, 2]
    counts = np.bincount(zs - zs.min())
    z_best = int(zs.min() + counts.argmax())
    sl_pts = coords[zs == z_best][:, :2] * zooms[:2]
    z2 = zooms[:2]
    long_ax, u = _longest_axis(sl_pts, z2)
    perp = np.array([-u[1], u[0]])
    short_ax = _extent_along(sl_pts, perp, z2)
    short_ax = min(short_ax, long_ax)

    vol_mm3 = nvox * vox_mm3
    cen = coords.mean(0)

    # --- attenuation
    hu = ct_sub[sub].astype(np.float32)

    # --- Couinaud location
    segvals = seg_sub[sub].astype(np.int64)
    cnt = np.bincount(segvals, minlength=9)
    frac = cnt / nvox
    primary = int(cnt[1:9].argmax() + 1) if cnt[1:9].sum() > 0 else None
    involved = [k for k in range(1, 9) if frac[k] >= config.MIN_SEGMENT_FRACTION]
    if primary and primary not in involved:
        involved.append(primary)
    involved = sorted(involved)

    return {
        "volume_mm3": float(vol_mm3),
        "voxel_count": int(nvox),
        "equiv_diameter_mm": float((6 * vol_mm3 / np.pi) ** (1 / 3)),
        "bbox_x_LR_mm": float(bbox[0]), "bbox_y_AP_mm": float(bbox[1]), "bbox_z_SI_mm": float(bbox[2]),
        "bbox_volume_mm3": float(np.prod(bbox)),
        "max_diameter_3d_mm": float(d3),
        "axial_long_mm": float(long_ax), "axial_short_mm": float(short_ax),
        "axial_slice": z_best,
        "centroid_vox": [float(c) for c in cen],
        "centroid_mm": [float(c) for c in cen * zooms],
        "mean_hu": float(hu.mean()), "median_hu": float(np.median(hu)), "std_hu": float(hu.std()),
        "min_hu": float(hu.min()), "max_hu": float(hu.max()),
        "primary_segment": primary,
        "primary_segment_label": f"Segment {SEGMENT_ROMAN[primary]}" if primary else "Unassigned",
        "lobe": LOBE.get(primary, "Unassigned"),
        "segments_involved": involved,
        "segments_involved_label": ", ".join(SEGMENT_ROMAN[k] for k in involved) if involved else "—",
        "segment_overlap_pct": {SEGMENT_ROMAN[k]: round(float(frac[k] * 100), 1) for k in range(1, 9) if frac[k] > 0},
        "outside_liver_pct": round(float(frac[0] * 100), 1),
    }


def analyze_lesions(ct, lesion_mask, seg_labels, zooms, min_volume_mm3=config.MIN_LESION_MM3):
    zooms = np.asarray(zooms, float)
    vox_mm3 = float(np.prod(zooms))
    cc, _ = ndi.label(lesion_mask, structure=ndi.generate_binary_structure(3, 3))
    kept = []
    for lab, sl in enumerate(ndi.find_objects(cc), 1):
        if sl is None:
            continue
        sub = cc[sl] == lab
        nvox = int(sub.sum())
        if nvox * vox_mm3 >= min_volume_mm3:
            kept.append((nvox, sl, sub))
    kept.sort(key=lambda t: -t[0])

    labels = np.zeros(cc.shape, dtype=np.int16)
    lesions = []
    for lid, (_, sl, sub) in enumerate(kept, 1):
        view = labels[sl]
        view[sub] = lid
        offset = np.array([s.start for s in sl])
        m = _lesion_metrics(sub, offset, ct[sl], seg_labels[sl], zooms, vox_mm3)
        m["id"] = lid
        lesions.append(m)
    return lesions, labels


def liver_statistics(ct, seg_labels, lesion_mask, zooms):
    zooms = np.asarray(zooms, float)
    vox = float(np.prod(zooms))  # mm³ per voxel
    cnt = np.bincount(seg_labels.ravel().astype(np.int64), minlength=9)
    seg_vol = {SEGMENT_ROMAN[k]: round(float(cnt[k] * vox), 1) for k in range(1, 9)}
    liver_vol = float(cnt[1:9].sum() * vox)
    les_cnt = np.bincount(seg_labels[lesion_mask].astype(np.int64), minlength=9) if lesion_mask.any() else np.zeros(9, int)
    lesion_per_seg = {SEGMENT_ROMAN[k]: round(float(les_cnt[k] * vox), 1) for k in range(1, 9)}
    parenchyma = (seg_labels > 0) & ~lesion_mask
    hu = ct[parenchyma].astype(np.float32) if parenchyma.any() else np.array([np.nan], np.float32)
    return {
        "liver_volume_mm3": round(liver_vol, 1),
        "segment_volume_mm3": seg_vol,
        "lesion_volume_per_segment_mm3": lesion_per_seg,
        "parenchyma_mean_hu": float(np.nanmean(hu)),
        "parenchyma_median_hu": float(np.nanmedian(hu)),
        "parenchyma_std_hu": float(np.nanstd(hu)),
        "unassigned_lesion_mm3": round(float(les_cnt[0] * vox), 1),
    }


def study_summary(lesions, liver):
    n = len(lesions)
    total = sum(l["volume_mm3"] for l in lesions)
    lv = liver["liver_volume_mm3"]
    out = {
        "lesion_count": n,
        "lesion_present": n > 0,
        "total_lesion_volume_mm3": round(total, 1),
        "liver_volume_mm3": lv,
        "lesion_burden_pct": round(100 * total / lv, 3) if lv > 0 else None,
        "largest_lesion_id": None, "largest_max_diameter_mm": None,
        "largest_axial_long_mm": None, "mean_lesion_hu_weighted": None,
        "segments_with_lesions": [],
    }
    if n:
        big = max(lesions, key=lambda l: l["max_diameter_3d_mm"])
        out["largest_lesion_id"] = big["id"]
        out["largest_max_diameter_mm"] = round(big["max_diameter_3d_mm"], 1)
        out["largest_axial_long_mm"] = round(max(l["axial_long_mm"] for l in lesions), 1)
        out["mean_lesion_hu_weighted"] = round(sum(l["mean_hu"] * l["volume_mm3"] for l in lesions) / total, 1)
        segs = sorted({k for l in lesions for k in l["segments_involved"]})
        out["segments_with_lesions"] = [SEGMENT_ROMAN[k] for k in segs]
    return out
