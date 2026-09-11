from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from app.models.ct_scan import CTScan
from app.models.segmentation import SegmentationResult, SegmentationTaskOutcome
from app.models.tool_call import ToolCall


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4)), str(path))


@pytest.fixture
def ct_scan_file(tmp_path: Path) -> Path:
    """A stand-in CT scan file (content doesn't matter for these tests)."""
    scan_path = tmp_path / "AC0001.nii.gz"
    _save_mask(scan_path, np.zeros((4, 4, 4)))
    return scan_path


def make_segmentation_result(
    tmp_path: Path,
    ct_scan: CTScan,
    *,
    segments_succeeded: bool = True,
    lesions_succeeded: bool = True,
    lesion_masks: dict[str, np.ndarray] | None = None,
    lesion_stats: dict | None = None,
    segment_stats: dict | None = None,
) -> SegmentationResult:
    """Build a SegmentationResult backed by real (tiny) NIfTI files on disk,
    so lesion<->segment overlap logic is exercised for real rather than mocked.
    """
    case_dir = tmp_path / "case_out" / ct_scan.study_id
    segments_dir = case_dir / "liver_segments"
    lesions_dir = case_dir / "liver_lesions"

    shape = (4, 4, 4)

    # Segment 6 occupies one quadrant, segment 7 another, others empty.
    segment_masks = {f"liver_segment_{i}": np.zeros(shape) for i in range(1, 9)}
    segment_masks["liver_segment_6"][0:2, 0:2, 0:2] = 1
    segment_masks["liver_segment_7"][2:4, 2:4, 2:4] = 1

    for name, mask in segment_masks.items():
        _save_mask(segments_dir / f"{name}.nii.gz", mask)

    default_segment_stats = {
        name: {"volume": int(mask.sum()) * 1000} for name, mask in segment_masks.items()
    }
    segment_stats = segment_stats if segment_stats is not None else default_segment_stats

    lesion_masks = lesion_masks or {}
    for name, mask in lesion_masks.items():
        _save_mask(lesions_dir / f"{name}.nii.gz", mask)

    default_lesion_stats = {
        name: {"volume": int(mask.sum()) * 1000} for name, mask in lesion_masks.items()
    }
    lesion_stats = lesion_stats if lesion_stats is not None else default_lesion_stats

    tasks = {
        "liver_segments": SegmentationTaskOutcome(
            tool_call=ToolCall(tool_name="liver_segments", input_path=ct_scan.file_path),
            output_dir=str(segments_dir),
            statistics=segment_stats,
            succeeded=segments_succeeded,
        ),
        "liver_lesions": SegmentationTaskOutcome(
            tool_call=ToolCall(tool_name="liver_lesions", input_path=ct_scan.file_path),
            output_dir=str(lesions_dir),
            statistics=lesion_stats,
            succeeded=lesions_succeeded,
        ),
    }

    return SegmentationResult(ct_scan=ct_scan, output_dir=str(case_dir), tasks=tasks)
