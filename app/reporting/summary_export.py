from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.analysis.liver import LIVER_SEGMENTS
from app.models.pipeline_result import PipelineResult


def _result_to_row(result: PipelineResult) -> dict:
    row: dict = {
        "study_id": result.ct_scan.study_id,
        "category": result.ct_scan.category,
        "status": result.status.value,
    }

    if result.analysis is None:
        for segment in LIVER_SEGMENTS:
            row[f"{segment}_vol_mL"] = None
        row["total_liver_vol_mL"] = None
        row["lesion_total_vol_mL"] = None
        row["lesion_segments"] = None
        row["error_message"] = result.error_message
        return row

    measurements = result.analysis.liver_measurements
    for segment in LIVER_SEGMENTS:
        row[f"{segment}_vol_mL"] = measurements.segment_volumes_mL.get(segment)
    row["total_liver_vol_mL"] = measurements.total_liver_vol_mL
    row["lesion_total_vol_mL"] = result.analysis.total_lesion_volume_mL
    row["lesion_segments"] = (
        ", ".join(f.liver_segment or "unknown" for f in result.analysis.lesion_findings) or None
    )
    row["error_message"] = None
    return row


def build_summary_dataframe(results: list[PipelineResult]) -> pd.DataFrame:
    """Build the case-level summary table (equivalent to the notebook's summary_df)."""
    return pd.DataFrame(_result_to_row(r) for r in results)


def export_summary(
    results: list[PipelineResult],
    output_dir: Path,
    filename: str = "summary_all_cases.csv",
) -> Path:
    """Write the summary table to CSV and return its path."""
    df = build_summary_dataframe(results)
    output_path = Path(output_dir) / filename
    df.to_csv(output_path, index=False)
    return output_path


def split_by_zero_volume(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Split cases into 'any zero volume' vs 'no zero volume' CSVs.

    Reproduces the notebook's zero-volume triage, generalized to whatever
    liver-segment volume columns are actually present in `df` rather than
    assuming a fixed set of 8.
    """
    segment_cols = [c for c in df.columns if c.startswith("liver_segment_") and c.endswith("_vol_mL")]

    zero_segment = (df[segment_cols] == 0).any(axis=1) if segment_cols else pd.Series(False, index=df.index)
    zero_liver_or_lesion = (df["total_liver_vol_mL"] == 0) | (df["lesion_total_vol_mL"] == 0)

    zero_mask = zero_liver_or_lesion | zero_segment
    zero_df = df[zero_mask].copy()
    non_zero_df = df[~zero_mask].copy()

    output_dir = Path(output_dir)
    zero_path = output_dir / "cases_with_zero_volume.csv"
    non_zero_path = output_dir / "cases_without_zero_volume.csv"
    zero_df.to_csv(zero_path, index=False)
    non_zero_df.to_csv(non_zero_path, index=False)

    return zero_path, non_zero_path
