from __future__ import annotations

import logging

from app.analysis.lesions import compute_lesion_findings, total_lesion_volume_from_stats
from app.analysis.liver import compute_liver_measurements
from app.models.analysis import CaseAnalysis, CaseStatus
from app.models.segmentation import SegmentationResult

logger = logging.getLogger(__name__)

REQUIRED_TASKS = ("liver_segments", "liver_lesions")


def determine_case_status(segmentation: SegmentationResult, analysis: CaseAnalysis) -> CaseStatus:
    """Classify the case outcome from segmentation success and findings only.

    Case-category metadata (e.g. a dataset folder name) is never consulted
    here. This replaces the original notebook's

        if case_class != "liver_lesion": return None

    gate: whether a case is reported as having a lesion now depends solely
    on whether the segmentation actually found one.
    """
    failed_tasks = [
        name
        for name in REQUIRED_TASKS
        if (task := segmentation.task(name)) is None or not task.succeeded
    ]
    if failed_tasks:
        return CaseStatus.SEGMENTATION_FAILED

    return CaseStatus.VALID_WITH_LESION if analysis.lesion_findings else CaseStatus.VALID_NO_LESION


def analyze_case(segmentation: SegmentationResult) -> tuple[CaseAnalysis, CaseStatus]:
    """Produce structured findings and a status for one segmented case.

    A lesion-free result is a valid finding, not an error - it is
    represented as CaseStatus.VALID_NO_LESION rather than raising or being
    treated as a failed case. Exceptions raised here (e.g. malformed
    statistics, unreadable mask files) are expected to propagate to the
    caller, which maps them to CaseStatus.ANALYSIS_FAILED.
    """
    liver_measurements = compute_liver_measurements(segmentation)
    lesion_findings = compute_lesion_findings(segmentation)

    analysis = CaseAnalysis(liver_measurements=liver_measurements, lesion_findings=lesion_findings)

    stats_total = total_lesion_volume_from_stats(segmentation)
    if abs(stats_total - analysis.total_lesion_volume_mL) > 0.01:
        logger.warning(
            "Lesion volume mismatch for study %s: mask-based sum=%.2f mL, "
            "statistics-key sum=%.2f mL. Statistics keys may not correspond "
            "1:1 with lesion mask filenames for this TotalSegmentator version.",
            segmentation.ct_scan.study_id,
            analysis.total_lesion_volume_mL,
            stats_total,
        )

    status = determine_case_status(segmentation, analysis)
    return analysis, status
