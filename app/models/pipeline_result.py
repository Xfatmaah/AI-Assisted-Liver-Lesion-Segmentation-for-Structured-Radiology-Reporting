from __future__ import annotations

from pydantic import BaseModel

from app.models.analysis import CaseAnalysis, CaseStatus
from app.models.ct_scan import CTScan


class PipelineResult(BaseModel):
    """Final structured output of running the pipeline on a single case.

    `analysis` is present whenever segmentation succeeded, regardless of
    whether a lesion was found — only SEGMENTATION_FAILED and
    ANALYSIS_FAILED leave it as None.
    """

    ct_scan: CTScan
    status: CaseStatus
    analysis: CaseAnalysis | None = None
    error_message: str | None = None
