<div align="center">

# AI-Assisted Liver Lesion Segmentation for Structured Radiology Reporting

**A modular pipeline for liver and liver-lesion segmentation, volumetric analysis, and Couinaud segment localization from CT imaging.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/API-FastAPI-009688)
![Segmentation](https://img.shields.io/badge/segmentation-TotalSegmentator-orange)
![Tests](https://img.shields.io/badge/tests-pytest-informational)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)

</div>

---

> **Intended use:** Research and educational purposes only. This software is not a medical device, has not been clinically validated, and must not be used for diagnosis or treatment decisions. All outputs require review by a qualified radiologist.

## Overview

Reviewing liver CT scans for lesions, measuring segment volumes, and assigning findings to the correct Couinaud segment is time-consuming and repetitive. This project investigates how much of that workflow can be automated with open-source segmentation models, and how results can be packaged as **structured, auditable data** rather than free-text notes.

Given a CT volume, the pipeline:

1. Segments the liver (by Couinaud segment) and liver lesions using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).
2. Computes total liver, per-segment, and per-lesion volumes.
3. Localizes each lesion to Couinaud segments by voxel overlap.
4. Returns a typed result with an explicit case status, suitable for review, export, and downstream report generation.

## Key Features

| Capability | Description |
| --- | --- |
| Segmentation | Liver segments and liver lesions via TotalSegmentator (`liver_segments`, `liver_lesions`) |
| Quantification | Total liver volume, per-segment volume, individual and total lesion volume |
| Localization | Voxel-overlap mapping of lesions to Couinaud segments |
| Explicit case status | Every case resolves to exactly one well-defined state; "no lesion" is a valid finding, not a failure |
| Batch processing | Directory-level processing with `statistics.json` caching to avoid redundant inference |
| Export | Summary DataFrame and CSV, including zero-volume triage |
| Interfaces | Python API and a FastAPI service |
| Quality | Pydantic-validated models; unit tests built on synthetic NIfTI masks |

## Pipeline

```
CT Scan (NIfTI)
      |
      v
Segmentation ............ TotalSegmentator (liver_segments, liver_lesions)
      |
      v
Quantitative Analysis ... total / per-segment liver volume, lesion volume
      |
      v
Lesion Localization ..... voxel-overlap mapping to Couinaud segments
      |
      v
Structured Output ....... CaseAnalysis + CaseStatus -> PipelineResult
      |
      v
Summary Export .......... DataFrame / CSV for batch review
```

### Implementation status

| Component | Status |
| --- | --- |
| Segmentation, volumetrics, lesion localization | Implemented |
| Python API and FastAPI endpoint | Implemented |
| Summary / CSV export | Implemented |
| LLM-assisted structured report drafting | Planned |

The current pipeline emits structured data, not narrative text. The `AgentResponse` model in `app/models/` is reserved for the future reporting layer.

## Repository Structure

```
.
├── app/
│   ├── main.py                    # FastAPI app: /health, /echo, /cases/process
│   ├── models/                    # Pydantic data models
│   │   ├── ct_scan.py             #   CTScan (pipeline input)
│   │   ├── segmentation.py        #   SegmentationTaskOutcome, SegmentationResult
│   │   ├── analysis.py            #   LiverMeasurements, LesionFinding, CaseAnalysis, CaseStatus
│   │   ├── pipeline_result.py     #   PipelineResult (final output)
│   │   ├── agent_response.py      #   Reserved for future report layer
│   │   └── tool_call.py           #   One segmentation task invocation
│   ├── segmentation/
│   │   ├── config.py              #   SegmentationConfig
│   │   └── total_segmentator.py   #   Per-task execution with statistics caching
│   ├── analysis/
│   │   ├── liver.py               #   Liver and segment volumes
│   │   ├── lesions.py             #   Lesion discovery, volume, Couinaud mapping
│   │   └── case_analysis.py       #   Aggregation and status derivation
│   ├── pipeline/
│   │   └── case_processor.py      #   discover_cases / process_case / process_directory
│   ├── reporting/
│   │   └── summary_export.py      #   Summary table, CSV export, triage
│   └── utils/
│       └── logging_config.py
├── notebooks/                     # Demonstration notebooks (import from app/)
├── tests/                         # Unit tests (test_analysis.py, test_pipeline.py)
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Getting Started

### Prerequisites

- Python version as pinned in `.python-version`
- [uv](https://docs.astral.sh/uv/) for dependency management
- Recommended: CUDA-capable GPU (CPU inference is supported but slower)
- Internet access on first run, as TotalSegmentator downloads model weights

> Some TotalSegmentator tasks may require a license. Review the [TotalSegmentator documentation](https://github.com/wasserth/TotalSegmentator) for current terms before use.

### Installation

```bash
git clone https://github.com/Xfatmaah/AI-Assisted-Liver-Lesion-Segmentation-for-Structured-Radiology-Reporting.git
cd AI-Assisted-Liver-Lesion-Segmentation-for-Structured-Radiology-Reporting
uv sync
```

## Usage

### Python API

```python
from pathlib import Path
from app.pipeline.case_processor import process_directory
from app.reporting.summary_export import export_summary

results = process_directory(Path("CT_scans"), Path("liver_seg_outputs"))
export_summary(results, Path("liver_seg_outputs"))
```

A complete walkthrough is provided in `notebooks/01_liver_segmentation_pipeline.ipynb`.

### REST API

Start the service:

```bash
uv run uvicorn app.main:app --reload
```

Submit a case:

```bash
curl -X POST http://localhost:8000/cases/process \
  -H "Content-Type: application/json" \
  -d '{"study_id": "case_001", "file_path": "/data/CT_scans/case_001.nii.gz"}'
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service health check |
| `/echo` | POST | Echo test endpoint |
| `/cases/process` | POST | Run the full pipeline on a single CT scan |

Interactive documentation is available at `http://localhost:8000/docs`.

### Typical workflow

1. Place NIfTI CT scans in a directory (subfolders are optional and used only for dataset bookkeeping).
2. Run `process_directory(...)` to segment and analyze all cases.
3. Inspect `PipelineResult.status` and `PipelineResult.analysis` per case.
4. Export the summary table with `export_summary(...)` for batch review.

## Output Schema

Each case is returned as a `PipelineResult`:

```
PipelineResult
├── study_id
├── status                   VALID_WITH_LESION | VALID_NO_LESION |
│                            SEGMENTATION_FAILED | ANALYSIS_FAILED
└── analysis (CaseAnalysis)
    ├── liver                LiverMeasurements: total and per-Couinaud-segment volume
    └── lesions[]            LesionFinding: volume and overlapping Couinaud segment(s)
```

### Case status definitions

| Status | Meaning |
| --- | --- |
| `VALID_NO_LESION` | Segmentation succeeded and no lesion was found. A valid, reportable finding. |
| `VALID_WITH_LESION` | Segmentation succeeded; one or more lesions were found and localized. |
| `SEGMENTATION_FAILED` | TotalSegmentator failed to run, or the input CT file was not found. |
| `ANALYSIS_FAILED` | Segmentation succeeded, but measurement or finding computation raised an error (for example, malformed statistics or an unreadable mask). |

**Design principle:** dataset category or folder names never influence case status or lesion reporting. The optional `category` field is carried through as metadata only.

## Testing

```bash
uv run pytest
```

Tests exercise the analysis and pipeline logic using synthetic NIfTI masks, so they run without patient data or GPU inference.

## Limitations

- **No clinical validation.** Segmentation and detection accuracy have not been evaluated against radiologist ground truth in this repository. No performance metrics are claimed.
- **Unverified statistics key mapping.** `analysis/lesions.py` assumes each lesion mask filename matches a key in TotalSegmentator's `statistics.json`. The assumption is cross-checked at runtime against an independent aggregate (`total_lesion_volume_from_stats`), and mismatches are logged as warnings, but it has not yet been confirmed against a real `liver_lesions` output.
- **No narrative report generation.** The pipeline returns structured data only.
- **Limited execution control.** Cases are processed sequentially; cross-case parallelism and device selection rely on TotalSegmentator defaults.

## Roadmap

- [ ] LLM-assisted structured report drafting from `CaseAnalysis`, with mandatory radiologist review
- [ ] Configurable device (CPU/GPU) and parallel batch processing
- [ ] Validation of the lesion-key mapping against real `liver_lesions` output
- [ ] Evaluation against radiologist-annotated ground truth

## Contributing

Contributions, issues, and suggestions are welcome. To contribute:

1. Fork the repository and create a feature branch.
2. Make your changes with accompanying tests.
3. Run `uv run pytest` and confirm all tests pass.
4. Open a pull request describing the change and its motivation.

## License

No license is currently specified. Add a `LICENSE` file (for example MIT or Apache-2.0) to define terms of use and contribution.

## Acknowledgements

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for the underlying segmentation models. Please cite its authors as described in their repository when using this work.
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), [nibabel](https://nipy.org/nibabel/), [pandas](https://pandas.pydata.org/), and [NumPy](https://numpy.org/).


