from __future__ import annotations

import json
import logging
from pathlib import Path

from app.models.ct_scan import CTScan
from app.models.segmentation import SegmentationResult, SegmentationTaskOutcome
from app.models.tool_call import ToolCall
from app.segmentation.config import SegmentationConfig

logger = logging.getLogger(__name__)


class SegmentationError(RuntimeError):
    """Raised for case-level segmentation problems (e.g. missing input file)."""


def _run_totalsegmentator_task(
    input_path: Path,
    task_output_dir: Path,
    task: str,
    config: SegmentationConfig,
) -> dict:
    """Run one TotalSegmentator task, or reuse cached statistics if present.

    Preserves the original notebook's caching behavior: if the task's
    statistics file already exists, cached output is reused instead of
    re-running segmentation. Callers are responsible for catching exceptions
    raised here and converting them into a SegmentationTaskOutcome.
    """
    from totalsegmentator.python_api import totalsegmentator  # imported lazily: heavy, optional at import time

    task_output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = task_output_dir / config.statistics_filename

    if stats_file.exists():
        logger.info("Task '%s' already exists for %s. Skipping.", task, input_path.name)
        with open(stats_file) as f:
            return json.load(f)

    totalsegmentator(
        input=str(input_path),
        output=str(task_output_dir),
        task=task,
        statistics=True,
        quiet=config.quiet,
        nr_thr_resamp=config.nr_thr_resamp,
        nr_thr_saving=config.nr_thr_saving,
    )

    if not stats_file.exists():
        raise RuntimeError(
            f"TotalSegmentator task '{task}' completed but produced no "
            f"statistics file at {stats_file}."
        )

    with open(stats_file) as f:
        return json.load(f)


def run_segmentation(
    ct_scan: CTScan,
    output_root: Path,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Run all configured TotalSegmentator tasks for a single CT scan.

    Each task's failure is captured independently (mirroring the original
    notebook's per-task try/except), so a failure in one task doesn't
    prevent inspecting the results of another. Only a missing/invalid input
    file raises `SegmentationError` directly, since no task can meaningfully
    run at all in that case.
    """
    config = config or SegmentationConfig()
    input_path = Path(ct_scan.file_path)

    if not input_path.exists():
        raise SegmentationError(f"CT scan file does not exist: {input_path}")

    case_output_dir = Path(output_root) / ct_scan.study_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    task_outcomes: dict[str, SegmentationTaskOutcome] = {}

    for task in config.tasks:
        tool_call = ToolCall(tool_name=task, input_path=str(input_path))
        task_output_dir = case_output_dir / task

        try:
            stats = _run_totalsegmentator_task(input_path, task_output_dir, task, config)
            task_outcomes[task] = SegmentationTaskOutcome(
                tool_call=tool_call,
                output_dir=str(task_output_dir),
                statistics=stats,
                succeeded=True,
            )
        except Exception as exc:  # noqa: BLE001 - intentionally broad; classified into a structured outcome below
            logger.exception("Segmentation task '%s' failed for %s", task, ct_scan.study_id)
            task_outcomes[task] = SegmentationTaskOutcome(
                tool_call=tool_call,
                output_dir=str(task_output_dir),
                statistics={},
                succeeded=False,
                error_message=str(exc),
            )

    return SegmentationResult(
        ct_scan=ct_scan,
        output_dir=str(case_output_dir),
        tasks=task_outcomes,
    )
