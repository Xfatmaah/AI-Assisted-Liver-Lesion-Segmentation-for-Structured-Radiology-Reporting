"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Where uploaded studies, TotalSegmentator outputs, RADAR results and reports are cached
WORK_DIR = Path(os.getenv("LIVER_AGENT_WORKDIR", BASE_DIR / "workspace"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Cloned RADAR repo (contains RADAR_inference/ and ckpt/). Same layout as in RADAR_Pipeline notebook.
_DEFAULT_RADAR = Path.home() / "Desktop" / "Liver Segmentation Project" / "damo-radar"
RADAR_REPO_DIR = Path(os.getenv("RADAR_REPO_DIR", _DEFAULT_RADAR))

# Optional: reuse outputs you already computed in the notebooks (saves hours on CPU)
#   PRECOMPUTED_SEG_ROOT  -> e.g. .../liver_seg_outputs   (contains <class>/<case>/liver_segments|liver_lesions)
#   PRECOMPUTED_RADAR_CSVS -> RADAR_infer_results_*.csv files, separated by ';' (Windows) or ':' (Linux)
_seg_root = os.getenv("PRECOMPUTED_SEG_ROOT", "")
PRECOMPUTED_SEG_ROOT = Path(_seg_root) if _seg_root else None
PRECOMPUTED_RADAR_CSVS = [p for p in os.getenv("PRECOMPUTED_RADAR_CSVS", "").split(os.pathsep) if p]

RADAR_POSITIVE_THRESHOLD = float(os.getenv("RADAR_POSITIVE_THRESHOLD", "0.5"))
MIN_LESION_MM3 = float(os.getenv("MIN_LESION_MM3", "50"))     # ignore specks smaller than this
MIN_SEGMENT_FRACTION = 0.10                                    # lesion "spans" a segment if >=10% of its voxels fall in it

# Optional LLM used ONLY to rephrase the impression (needs ANTHROPIC_API_KEY)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

# Reference performance quoted in the project brief (CT liver lesion detection sensitivity)
REFERENCE_SENSITIVITY_CT = 75.8

DISCLAIMER = "Draft for radiologist review — not a diagnosis."
