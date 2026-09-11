from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.models.analysis import CaseStatus
from app.models.ct_scan import CTScan
from app.pipeline import case_processor

from tests.conftest import make_segmentation_result


def test_missing_input_file_is_segmentation_failed(tmp_path):
    ct_scan = CTScan(study_id="missing_case", file_path=str(tmp_path / "does_not_exist.nii.gz"))

    result = case_processor.process_case(ct_scan, output_root=tmp_path / "out")

    assert result.status == CaseStatus.SEGMENTATION_FAILED
    assert result.analysis is None
    assert "does not exist" in result.error_message


def test_valid_no_lesion_case_end_to_end(tmp_path, ct_scan_file, monkeypatch):
    ct_scan = CTScan(study_id="AC0001", file_path=str(ct_scan_file), category="normal")

    fake_result = make_segmentation_result(tmp_path, ct_scan, lesion_masks={})
    monkeypatch.setattr(case_processor, "run_segmentation", lambda *a, **k: fake_result)

    result = case_processor.process_case(ct_scan, output_root=tmp_path / "out")

    assert result.status == CaseStatus.VALID_NO_LESION
    assert result.analysis is not None
    assert result.analysis.lesion_findings == []


def test_valid_with_lesion_case_end_to_end_ignores_category(tmp_path, ct_scan_file, monkeypatch):
    """Category is 'normal' (not 'liver_lesion'), but a real lesion mask exists -
    the pipeline must still report it, unlike the original notebook's gate."""
    lesion_mask = np.zeros((4, 4, 4))
    lesion_mask[0:2, 0:2, 0:2] = 1

    ct_scan = CTScan(study_id="AC0002", file_path=str(ct_scan_file), category="normal")
    fake_result = make_segmentation_result(tmp_path, ct_scan, lesion_masks={"lesion_01": lesion_mask})
    monkeypatch.setattr(case_processor, "run_segmentation", lambda *a, **k: fake_result)

    result = case_processor.process_case(ct_scan, output_root=tmp_path / "out")

    assert result.status == CaseStatus.VALID_WITH_LESION
    assert len(result.analysis.lesion_findings) == 1


def test_segmentation_task_failure_is_reported_not_swallowed(tmp_path, ct_scan_file, monkeypatch):
    ct_scan = CTScan(study_id="AC0003", file_path=str(ct_scan_file))
    fake_result = make_segmentation_result(tmp_path, ct_scan, segments_succeeded=False, lesion_masks={})
    monkeypatch.setattr(case_processor, "run_segmentation", lambda *a, **k: fake_result)

    result = case_processor.process_case(ct_scan, output_root=tmp_path / "out")

    assert result.status == CaseStatus.SEGMENTATION_FAILED


def test_analysis_failure_is_distinguished(tmp_path, ct_scan_file, monkeypatch):
    ct_scan = CTScan(study_id="AC0004", file_path=str(ct_scan_file))
    fake_result = make_segmentation_result(tmp_path, ct_scan, lesion_masks={})
    monkeypatch.setattr(case_processor, "run_segmentation", lambda *a, **k: fake_result)

    def _boom(_segmentation):
        raise ValueError("malformed statistics")

    monkeypatch.setattr(case_processor, "analyze_case", _boom)

    result = case_processor.process_case(ct_scan, output_root=tmp_path / "out")

    assert result.status == CaseStatus.ANALYSIS_FAILED
    assert "malformed statistics" in result.error_message


def test_discover_cases_treats_subfolder_as_optional_category(tmp_path):
    (tmp_path / "liver_lesion").mkdir()
    (tmp_path / "normal").mkdir()
    (tmp_path / "liver_lesion" / "case_a.nii.gz").write_bytes(b"")
    (tmp_path / "normal" / "case_b.nii.gz").write_bytes(b"")

    cases = case_processor.discover_cases(tmp_path)

    by_id = {c.study_id: c for c in cases}
    assert by_id["case_a"].category == "liver_lesion"
    assert by_id["case_b"].category == "normal"
    assert len(cases) == 2
