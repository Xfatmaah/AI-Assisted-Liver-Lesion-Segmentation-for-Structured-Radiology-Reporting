from __future__ import annotations

import logging
from pathlib import Path

from app.analysis.case_analysis import analyze_case
from app.models.analysis import CaseStatus
from app.models.ct_scan import CTScan
from app.models.pipeline_result import PipelineResult
from app.segmentation.config import SegmentationConfig
from app.segmentation.total_segmentator import SegmentationError, run_segmentation

logger = logging.getLogger(__name__)


def process_case(
    ct_scan: CTScan,
    output_root: Path,
    config: SegmentationConfig | None = None,
) -> PipelineResult:
    """Run the full pipeline (segmentation -> analysis) for a single case.

    Always returns a PipelineResult rather than raising, so a batch caller
    can process many cases without one failure aborting the run. Failure
    modes are distinguished via `PipelineResult.status`
    (SEGMENTATION_FAILED vs ANALYSIS_FAILED).
    """
    try:
        segmentation = run_segmentation(ct_scan, output_root, config)
    except SegmentationError as exc:
        logger.error("Segmentation failed for %s: %s", ct_scan.study_id, exc)
        return PipelineResult(ct_scan=ct_scan, status=CaseStatus.SEGMENTATION_FAILED, error_message=str(exc))

    try:
        analysis, status = analyze_case(segmentation)
    except Exception as exc:  # noqa: BLE001 - converted into a structured, classified result below
        logger.exception("Analysis failed for %s", ct_scan.study_id)
        return PipelineResult(ct_scan=ct_scan, status=CaseStatus.ANALYSIS_FAILED, error_message=str(exc))

    return PipelineResult(ct_scan=ct_scan, status=status, analysis=analysis)


def discover_cases(input_dir: Path) -> list[CTScan]:
    """Walk a directory of CT scans, treating immediate subdirectories as
    optional category metadata - never as processing control flow.

    Supports two layouts:
      - input_dir/*.nii(.gz)             -> category=None
      - input_dir/<category>/*.nii(.gz)  -> category=<subfolder name>
    """
    input_dir = Path(input_dir)
    cases: list[CTScan] = []

    top_level_files = sorted(input_dir.glob("*.nii.gz")) + sorted(input_dir.glob("*.nii"))
    for f in top_level_files:
        cases.append(CTScan(study_id=f.name.replace(".nii.gz", "").replace(".nii", ""), file_path=str(f)))

    for sub_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        nii_files = sorted(sub_dir.glob("*.nii.gz")) + sorted(sub_dir.glob("*.nii"))
        for f in nii_files:
            cases.append(
                CTScan(
                    study_id=f.name.replace(".nii.gz", "").replace(".nii", ""),
                    file_path=str(f),
                    category=sub_dir.name,
                )
            )

    return cases


def process_directory(
    input_dir: Path,
    output_root: Path,
    config: SegmentationConfig | None = None,
) -> list[PipelineResult]:
    """Discover and process every case found under `input_dir`."""
    cases = discover_cases(input_dir)
    logger.info("Discovered %d case(s) under %s", len(cases), input_dir)
    return [process_case(case, output_root, config) for case in cases]
