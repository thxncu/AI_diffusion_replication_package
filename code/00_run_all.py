#!/usr/bin/env python3
"""Run the full Q1 2026 replication workflow.

The script regenerates harmonized Q1 2026 data, main tables, robustness tables,
and figures from the included public/harmonized datasets. It can be executed from
any working directory.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

def main() -> None:
    # Step 1: merge Microsoft Q1 2026 update, reconstruct clusters, regressions,
    # figures, and main diagnostic tables.
    run([
        PYTHON,
        str(ROOT / "code" / "reproduce_q1_2026_analysis.py"),
        "--input", str(ROOT / "data" / "final_analysis_data_with_bti_and_selection.csv"),
        "--q1", str(ROOT / "data" / "AI_Diffusion_Q12026_Update.csv"),
        "--outdir", str(ROOT),
    ])
    # Step 2: run extended robustness checks used in the Supplementary Material.
    run([
        PYTHON,
        str(ROOT / "code" / "reproduce_extended_robustness_analysis.py"),
    ])
    print("\nReplication run completed. Outputs are in data/, tables/, figures/, and reports/.")

if __name__ == "__main__":
    main()
