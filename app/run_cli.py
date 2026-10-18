"""Headless run:  python run_cli.py CT.nii.gz [--prior PRIOR.nii.gz] [--no-radar] [--llm]"""
import argparse

from agent import LiverReportingAgent

ap = argparse.ArgumentParser()
ap.add_argument("ct")
ap.add_argument("--prior")
ap.add_argument("--no-radar", action="store_true")
ap.add_argument("--llm", action="store_true")
ap.add_argument("--min-mm3", type=float, default=50)
a = ap.parse_args()
for kind, payload in LiverReportingAgent().run(a.ct, a.prior, not a.no_radar, a.llm, a.min_mm3):
    if kind == "log":
        print(payload, flush=True)
    elif kind == "result":
        print("\nFiles:\n  " + "\n  ".join(payload["files"]))
