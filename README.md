# AI-Assisted Liver Lesion Segmentation for Structured Radiology Reporting

A research-prototype pipeline that segments the liver and liver lesions from
CT scans, computes volumetric measurements, and localizes lesions to
Couinaud liver segments — as a foundation for AI-assisted, radiologist-
reviewed structured reporting.

> **Research prototype, not a diagnostic tool.** This pipeline is
> AI-assisted decision-support software for research and evaluation. It has
> not been clinically validated, does not replace radiologist review, and
> all findings require confirmation by a qualified radiologist before any
> clinical use.

## Problem Statement

Manually reviewing liver CT scans for lesions, measuring liver segment
volumes, and localizing findings to the correct Couinaud segment is
time-consuming and repetitive. This project explores how much of that
workflow can be automated with existing open-source segmentation models,
producing structured, reviewable output rather than free-text notes.

## Pipeline

```
CT Scan
  ↓
Liver / Liver Lesion Segmentation      (TotalSegmentator)
  ↓
Quantitative Measurements              (per-segment + total liver volume, lesion volume)
  ↓
Lesion Localization                    (voxel-overlap mapping to Couinaud segments)
  ↓
Structured Findings                    (CaseAnalysis, CaseStatus)
```

**Implemented today:** segmentation, volumetric measurement, and lesion
localization, as described above, exposed via a Python API and a minimal
FastAPI endpoint.

**Not yet implemented (planned / future work):** LLM-based structured
report generation from `CaseAnalysis` output. The `AgentResponse` model in
`app/models/` is a placeholder for that future layer; no LLM-generated
report is produced by the current pipeline.

## Architecture

```
app/
├── main.py                    # FastAPI app: /health, /echo, /cases/process
├── models/
│   ├── ct_scan.py             # CTScan — pipeline input; `category` is optional metadata only
│   ├── segmentation.py        # SegmentationTaskOutcome, SegmentationResult
│   ├── analysis.py            # LiverMeasurements, LesionFinding, CaseAnalysis, CaseStatus
│   ├── pipeline_result.py     # PipelineResult — final structured output
│   ├── agent_response.py      # (existing) reserved for future LLM report layer
│   └── tool_call.py           # (existing) reused to represent one segmentation task invocation
│
├── segmentation/
│   ├── config.py               # SegmentationConfig (tasks, thread counts, stats filename)
│   └── total_segmentator.py    # Runs TotalSegmentator per task, with statistics.json caching
│
├── analysis/
│   ├── liver.py                # Per-segment + total liver volume
│   ├── lesions.py               # Lesion discovery (all masks), volume, Couinaud segment mapping
│   └── case_analysis.py         # Combines the above; derives CaseStatus — category-agnostic
│
├── pipeline/
│   └── case_processor.py        # discover_cases / process_case / process_directory
│
├── reporting/
│   └── summary_export.py        # Summary DataFrame, CSV export, zero-volume triage
│
└── utils/
    └── logging_config.py        # Replaces ad hoc print() calls

notebooks/
└── 01_liver_segmentation_pipeline.ipynb   # Demo layer; imports app/, does not duplicate logic

tests/
└── test_analysis.py, test_pipeline.py     # Unit tests, incl. synthetic NIfTI masks
```

### Case status

Every processed case ends in exactly one of:

| Status | Meaning |
|---|---|
| `VALID_NO_LESION` | Segmentation succeeded; no lesion was found. This is a valid, reportable finding — not a failure. |
| `VALID_WITH_LESION` | Segmentation succeeded; one or more lesions were found and localized. |
| `SEGMENTATION_FAILED` | TotalSegmentator failed to run, or the input CT scan file could not be found. |
| `ANALYSIS_FAILED` | Segmentation succeeded, but computing measurements/findings from its output raised an error (e.g. malformed statistics, unreadable mask). |

Crucially, **no dataset category or folder name influences this status or
whether lesions are reported.** A case's `category` field (if provided) is
carried through as metadata for dataset organization and evaluation only.

## Technologies

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for CT segmentation (`liver_segments`, `liver_lesions` tasks)
- Pydantic for structured data models
- FastAPI for the API layer
- pandas / nibabel / numpy for measurement and mask analysis
- pytest for testing

## Installation

```bash
uv sync
```

## Usage

### As a Python library / in a notebook

```python
from pathlib import Path
from app.pipeline.case_processor import process_directory
from app.reporting.summary_export import export_summary

results = process_directory(Path("CT_scans"), Path("liver_seg_outputs"))
export_summary(results, Path("liver_seg_outputs"))
```

See `notebooks/01_liver_segmentation_pipeline.ipynb` for a full walkthrough.

### As an API

```bash
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/cases/process \
  -H "Content-Type: application/json" \
  -d '{"study_id": "case_001", "file_path": "/data/CT_scans/case_001.nii.gz"}'
```

### Running tests

```bash
uv run pytest
```

## Project Structure

See [Architecture](#architecture) above.

## Example Workflow

1. Place CT scans under a directory, optionally grouped into subfolders for dataset bookkeeping.
2. Run `process_directory(...)` to segment and analyze every case.
3. Inspect `PipelineResult.status` and `PipelineResult.analysis` per case.
4. Export a summary table with `export_summary(...)` for batch review.

## Current Limitations

- **No clinical validation.** Segmentation and lesion-detection accuracy have not been evaluated against radiologist ground truth in this repository; no performance metrics are claimed.
- **Lesion-to-statistics key mapping assumption.** `analysis/lesions.py` assumes each lesion mask filename corresponds to a matching key in TotalSegmentator's `statistics.json`. This is cross-checked at runtime against an independent aggregate (`total_lesion_volume_from_stats`), and a mismatch is logged as a warning — but the assumption itself has not been confirmed against a real `liver_lesions` statistics.json sample. **REQUIRES CONFIRMATION.**
- **No LLM-based report generation.** `AgentResponse` exists as a placeholder model for a future structured-reporting layer; the current pipeline returns structured data (`PipelineResult`), not narrative text.
- **Single-scan, single-process execution.** No batching/parallelism across cases or GPU device selection is currently configurable beyond TotalSegmentator's own defaults.

## Future Work

- LLM-assisted structured report drafting from `CaseAnalysis`, with mandatory radiologist review before finalization.
- Configurable device (CPU/GPU) and parallelism for batch processing.
- Evaluation against radiologist-annotated ground truth, if/when such a dataset is available.
