from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class CaseStatus(str, Enum):
    """High-level outcome of running the pipeline on one case.

    Deliberately has no generic "SUCCESS" value: a successful run always
    ends in one of the two VALID_* states, since "no lesion found" is a
    valid, reportable outcome rather than a fallback.
    """

    VALID_NO_LESION = "VALID_NO_LESION"
    VALID_WITH_LESION = "VALID_WITH_LESION"
    SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"


class LiverMeasurements(BaseModel):
    """Per-Couinaud-segment and total liver volume, in milliliters."""

    segment_volumes_mL: dict[str, float]
    total_liver_vol_mL: float


class LesionFinding(BaseModel):
    """A single detected liver lesion, derived from segmentation output."""

    lesion_id: str
    volume_mL: float
    liver_segment: str | None = None


class CaseAnalysis(BaseModel):
    """Structured findings derived from one case's segmentation output.

    An empty `lesion_findings` list is a valid finding ("no lesion
    detected"), not an error state.
    """

    liver_measurements: LiverMeasurements
    lesion_findings: list[LesionFinding]

    @property
    def total_lesion_volume_mL(self) -> float:
        return round(sum(f.volume_mL for f in self.lesion_findings), 2)

    @property
    def lesion_count(self) -> int:
        return len(self.lesion_findings)

    @property
    def summary(self) -> str:
        """A short, deterministic (non-LLM) human-readable finding summary."""
        if not self.lesion_findings:
            return "No lesion detected."
        segments = ", ".join(f.liver_segment or "unknown segment" for f in self.lesion_findings)
        return (
            f"{self.lesion_count} lesion(s) detected "
            f"({self.total_lesion_volume_mL} mL total) in: {segments}."
        )
