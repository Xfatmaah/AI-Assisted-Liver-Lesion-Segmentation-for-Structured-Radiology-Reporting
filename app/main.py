import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models.ct_scan import CTScan
from app.models.pipeline_result import PipelineResult
from app.pipeline.case_processor import process_case
from app.segmentation.config import SegmentationConfig
from app.utils.logging_config import configure_logging

configure_logging()

app = FastAPI()

# Configurable via environment variable; defaults preserve the notebook's original relative path.
OUTPUT_ROOT = Path(os.environ.get("LIVER_SEG_OUTPUT_DIR", "liver_seg_outputs"))


class Message(BaseModel):
    text: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/echo")
async def echo():
    return {"message": "Echo endpoint"}


@app.post("/cases/process", response_model=PipelineResult)
async def process_case_endpoint(ct_scan: CTScan) -> PipelineResult:
    """Run the segmentation + analysis pipeline on a single CT scan.

    Research-prototype endpoint: returns structured, unvalidated
    measurements and findings intended for radiologist review, not a
    clinical diagnosis. `ct_scan.category`, if provided, is stored for
    reference only and never affects what is processed or reported.
    """
    if not Path(ct_scan.file_path).exists():
        raise HTTPException(status_code=400, detail=f"CT scan file not found: {ct_scan.file_path}")

    return process_case(ct_scan, OUTPUT_ROOT, SegmentationConfig())
