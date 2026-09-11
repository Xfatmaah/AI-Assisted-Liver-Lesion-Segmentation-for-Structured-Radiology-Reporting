from __future__ import annotations

from pydantic import BaseModel

from app.models.ct_scan import CTScan
from app.models.tool_call import ToolCall


class SegmentationTaskOutcome(BaseModel):
    """Result of running a single TotalSegmentator task for one case.

    `succeeded=False` represents a genuine segmentation/tool failure (e.g.
    TotalSegmentator raised an exception, or produced no statistics file).
    It is distinct from a successful run that simply found nothing — that
    case has `succeeded=True` with empty/zero statistics.
    """

    tool_call: ToolCall
    output_dir: str
    statistics: dict = {}
    succeeded: bool
    error_message: str | None = None


class SegmentationResult(BaseModel):
    """Aggregated segmentation output for one CT case, across all tasks run."""

    ct_scan: CTScan
    output_dir: str
    tasks: dict[str, SegmentationTaskOutcome]

    def task(self, name: str) -> SegmentationTaskOutcome | None:
        """Look up a task's outcome by name (e.g. "liver_segments")."""
        return self.tasks.get(name)
