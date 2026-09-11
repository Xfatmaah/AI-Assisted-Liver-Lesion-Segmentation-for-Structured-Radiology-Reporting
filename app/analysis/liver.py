from __future__ import annotations

from app.models.analysis import LiverMeasurements
from app.models.segmentation import SegmentationResult

# The 8 Couinaud liver segments - a fixed anatomical constant, not a dataset assumption.
LIVER_SEGMENTS: tuple[str, ...] = tuple(f"liver_segment_{i}" for i in range(1, 9))

LIVER_SEGMENTATION_TASK = "liver_segments"


def compute_liver_measurements(segmentation: SegmentationResult) -> LiverMeasurements:
    """Compute per-segment and total liver volume (mL) from segmentation stats.

    A missing segment class in the statistics is treated as 0 mL rather than
    raising, since its absence can legitimately reflect model output (e.g.
    a segment not visible in the scan) rather than an error.
    """
    task_outcome = segmentation.task(LIVER_SEGMENTATION_TASK)
    stats = task_outcome.statistics if task_outcome and task_outcome.succeeded else {}

    volumes_mL: dict[str, float] = {}
    total_mm3 = 0.0

    for segment in LIVER_SEGMENTS:
        volume_mm3 = stats.get(segment, {}).get("volume", 0)
        total_mm3 += volume_mm3
        volumes_mL[segment] = round(volume_mm3 / 1000, 2)

    return LiverMeasurements(
        segment_volumes_mL=volumes_mL,
        total_liver_vol_mL=round(total_mm3 / 1000, 2),
    )
