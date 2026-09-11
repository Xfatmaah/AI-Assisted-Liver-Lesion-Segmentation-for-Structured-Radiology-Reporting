from __future__ import annotations

from pathlib import Path

import numpy as np

from app.analysis.case_analysis import analyze_case
from app.analysis.liver import compute_liver_measurements
from app.analysis.lesions import compute_lesion_findings
from app.models.analysis import CaseStatus
from app.models.ct_scan import CTScan

from tests.conftest import make_segmentation_result


def _ct_scan(ct_scan_file: Path, study_id: str = "AC0001", category: str | None = None) -> CTScan:
    return CTScan(study_id=study_id, file_path=str(ct_scan_file), category=category)


def test_liver_measurements_sum_segments(tmp_path, ct_scan_file):
    ct_scan = _ct_scan(ct_scan_file)
    segmentation = make_segmentation_result(tmp_path, ct_scan)

    measurements = compute_liver_measurements(segmentation)

    assert measurements.total_liver_vol_mL == sum(measurements.segment_volumes_mL.values())
    assert measurements.segment_volumes_mL["liver_segment_6"] == 8.0  # 2x2x2 voxels * 1000 / 1000
    assert measurements.segment_volumes_mL["liver_segment_1"] == 0.0


def test_no_lesion_case_is_valid_not_failed(tmp_path, ct_scan_file):
    """A case with a real, empty liver_lesions output must be VALID_NO_LESION,
    never treated as a failure."""
    ct_scan = _ct_scan(ct_scan_file, category="normal")
    segmentation = make_segmentation_result(tmp_path, ct_scan, lesion_masks={})

    analysis, status = analyze_case(segmentation)

    assert status == CaseStatus.VALID_NO_LESION
    assert analysis.lesion_findings == []
    assert analysis.summary == "No lesion detected."


def test_lesion_present_case_maps_to_correct_segment(tmp_path, ct_scan_file):
    """Lesion sitting inside segment 6's voxels should be localized to Segment 6,
    with NO dependency on any category label."""
    lesion_mask = np.zeros((4, 4, 4))
    lesion_mask[0:2, 0:2, 0:2] = 1  # fully inside liver_segment_6

    ct_scan = _ct_scan(ct_scan_file, category="abnormal_no_liver")  # deliberately NOT "liver_lesion"
    segmentation = make_segmentation_result(tmp_path, ct_scan, lesion_masks={"lesion_01": lesion_mask})

    analysis, status = analyze_case(segmentation)

    assert status == CaseStatus.VALID_WITH_LESION
    assert len(analysis.lesion_findings) == 1
    assert analysis.lesion_findings[0].liver_segment == "Segment 6"
    assert analysis.lesion_findings[0].volume_mL == 8.0


def test_multiple_lesions_are_all_captured(tmp_path, ct_scan_file):
    """Regression test for the original bug where only lesion_files[0] was ever used."""
    lesion_1 = np.zeros((4, 4, 4))
    lesion_1[0:2, 0:2, 0:2] = 1  # inside segment 6

    lesion_2 = np.zeros((4, 4, 4))
    lesion_2[2:4, 2:4, 2:4] = 1  # inside segment 7

    ct_scan = _ct_scan(ct_scan_file)
    segmentation = make_segmentation_result(
        tmp_path, ct_scan, lesion_masks={"lesion_01": lesion_1, "lesion_02": lesion_2}
    )

    findings = compute_lesion_findings(segmentation)

    assert len(findings) == 2
    segments_found = {f.liver_segment for f in findings}
    assert segments_found == {"Segment 6", "Segment 7"}


def test_segmentation_failure_is_distinguished_from_no_lesion(tmp_path, ct_scan_file):
    ct_scan = _ct_scan(ct_scan_file)
    segmentation = make_segmentation_result(tmp_path, ct_scan, lesions_succeeded=False, lesion_masks={})

    analysis, status = analyze_case(segmentation)

    assert status == CaseStatus.SEGMENTATION_FAILED
    # No lesion findings can be computed from a failed task.
    assert analysis.lesion_findings == []
