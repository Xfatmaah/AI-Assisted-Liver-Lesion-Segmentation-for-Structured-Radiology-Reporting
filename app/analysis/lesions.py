from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from app.analysis.liver import LIVER_SEGMENTS
from app.models.analysis import LesionFinding
from app.models.segmentation import SegmentationResult

logger = logging.getLogger(__name__)

LESION_SEGMENTATION_TASK = "liver_lesions"
LIVER_SEGMENTATION_TASK = "liver_segments"
LESION_MASK_GLOB = "*lesion*.nii.gz"


def _discover_lesion_masks(lesion_output_dir: Path) -> list[Path]:
    """Return every lesion instance mask produced by the lesion task.

    The original notebook only ever inspected `lesion_files[0]`; this
    generalizes to all matching mask files so multiple lesions are each
    represented, rather than only the first one found.
    """
    if not lesion_output_dir.exists():
        return []
    return sorted(lesion_output_dir.glob(LESION_MASK_GLOB))


def _map_mask_to_liver_segment(lesion_mask_path: Path, liver_segments_dir: Path) -> str | None:
    """Return the Couinaud segment with the greatest voxel overlap with the lesion mask.

    Returns None if no segment mask overlaps the lesion at all (e.g. the
    lesion sits outside the segmented liver, or segment masks are missing).
    This mirrors the original get_lesion_segment() overlap logic exactly,
    minus the category gate that used to skip this computation entirely for
    cases not folder-labeled "liver_lesion".
    """
    lesion = nib.load(str(lesion_mask_path)).get_fdata() > 0

    best_segment: str | None = None
    best_overlap = 0

    for segment in LIVER_SEGMENTS:
        segment_file = liver_segments_dir / f"{segment}.nii.gz"
        if not segment_file.exists():
            continue

        segment_mask = nib.load(str(segment_file)).get_fdata() > 0
        overlap = int(np.logical_and(segment_mask, lesion).sum())

        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment

    if best_overlap == 0 or best_segment is None:
        return None

    segment_number = best_segment.rsplit("_", 1)[-1]
    return f"Segment {segment_number}"


def compute_lesion_findings(segmentation: SegmentationResult) -> list[LesionFinding]:
    """Derive structured lesion findings purely from segmentation output.

    This never consults case-category metadata. A case is treated as having
    lesions if and only if the `liver_lesions` task produced one or more
    lesion mask files with nonzero measured volume.

    ASSUMPTION (flagged for confirmation against real TotalSegmentator
    output): each lesion mask filename stem is assumed to match its
    corresponding key in statistics.json. If a lesion mask file has no
    matching statistics key, its volume defaults to 0 mL and a warning is
    logged; see `total_lesion_volume_from_stats` for an independent
    aggregate that does not depend on this assumption.
    """
    lesion_task = segmentation.task(LESION_SEGMENTATION_TASK)
    if lesion_task is None or not lesion_task.succeeded:
        return []

    lesion_output_dir = Path(lesion_task.output_dir)
    liver_segments_task = segmentation.task(LIVER_SEGMENTATION_TASK)
    liver_segments_dir = (
        Path(liver_segments_task.output_dir)
        if liver_segments_task and liver_segments_task.succeeded
        else None
    )

    findings: list[LesionFinding] = []
    stats = lesion_task.statistics

    for mask_path in _discover_lesion_masks(lesion_output_dir):
        lesion_key = mask_path.name.replace(".nii.gz", "").replace(".nii", "")
        lesion_stats = stats.get(lesion_key)

        if lesion_stats is None:
            logger.warning(
                "No statistics entry found for lesion mask '%s' (study %s); "
                "assuming 0 mL for this lesion.",
                lesion_key,
                segmentation.ct_scan.study_id,
            )
            volume_mL = 0.0
        else:
            volume_mL = round(lesion_stats.get("volume", 0) / 1000, 2)

        if volume_mL <= 0:
            continue

        segment = (
            _map_mask_to_liver_segment(mask_path, liver_segments_dir)
            if liver_segments_dir is not None
            else None
        )

        findings.append(LesionFinding(lesion_id=lesion_key, volume_mL=volume_mL, liver_segment=segment))

    return findings


def total_lesion_volume_from_stats(segmentation: SegmentationResult) -> float:
    """Sum every statistics.json key containing "lesion" (mL).

    This exactly reproduces the original notebook's aggregate lesion-volume
    calculation and is kept as an independent cross-check alongside
    `compute_lesion_findings`: if the two disagree for a case, the
    per-mask-file assumption above doesn't hold for that TotalSegmentator
    output and should be investigated (see README limitations).
    """
    lesion_task = segmentation.task(LESION_SEGMENTATION_TASK)
    stats = lesion_task.statistics if lesion_task and lesion_task.succeeded else {}

    total_mm3 = sum(val.get("volume", 0) for key, val in stats.items() if "lesion" in key.lower())
    return round(total_mm3 / 1000, 2)
