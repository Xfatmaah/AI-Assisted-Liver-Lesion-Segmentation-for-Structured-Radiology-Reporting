# AI-Assisted Liver Lesion Reporting (prototype)

CT NIfTI -> TotalSegmentator (liver_segments + liver_lesions) -> per-lesion measurements -> RADAR -> overlay viewer + structured report.

## Run
    pip install -r requirements.txt
    # RADAR: your existing damo-radar folder (cloned + checkpoints downloaded, CPU patch applied as in the notebook)
    set RADAR_REPO_DIR=C:\Users\HP EliteBook\Desktop\Liver Segmentation Project\damo-radar      (Windows)
    export RADAR_REPO_DIR=/path/to/damo-radar                                                  (Linux/Mac)
    python app.py            # open http://127.0.0.1:7860
    python run_cli.py CT.nii.gz [--prior PRIOR.nii.gz] [--no-radar] [--llm]   # headless

## Optional environment variables
| Variable | Purpose |
|---|---|
| RADAR_REPO_DIR | location of damo-radar (needs RADAR_inference/ and ckpt/) |
| PRECOMPUTED_SEG_ROOT | reuse notebook outputs, e.g. .../liver_seg_outputs (cases found by name) |
| PRECOMPUTED_RADAR_CSVS | reuse RADAR_infer_results_*.csv (separate several with `;` on Windows / `:` on Linux) |
| ANTHROPIC_API_KEY, LLM_MODEL | only for the optional "Rephrase impression" switch |
| LIVER_AGENT_WORKDIR | cache / output folder (default ./workspace) |
| RADAR_POSITIVE_THRESHOLD, MIN_LESION_ML | report thresholds |

All results are cached per study in `workspace/<case>_<hash>/` (re-running the same file is instant).
