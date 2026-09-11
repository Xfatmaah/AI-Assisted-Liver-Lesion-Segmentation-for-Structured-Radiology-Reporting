from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for running TotalSegmentator.

    Defaults match the original notebook exactly, so behavior is preserved
    unless explicitly overridden (e.g. from environment variables in
    app/main.py).
    """

    tasks: tuple[str, ...] = ("liver_segments", "liver_lesions")
    statistics_filename: str = "statistics.json"
    nr_thr_resamp: int = 1
    nr_thr_saving: int = 1
    quiet: bool = True
