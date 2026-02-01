#!/usr/bin/env python3
"""
run_pipeline.py
================
Transit Desert Identification Pipeline runner with:

1) Clean-slate runs (empties previous raw/processed/outputs but keeps folders)
2) Step 2 Compute Supply (CPTA) before Step 3
3) Run logging (captures everything printed during the run into logs/*.log)

Usage examples:
  python run_pipeline.py
  python run_pipeline.py --steps 1,2,3,4,5
  python run_pipeline.py --no-clean
  python run_pipeline.py --quiet
  python run_pipeline.py --include-jobs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List


# ---------------------------
# Project paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------
# Step registry
# ---------------------------
@dataclass(frozen=True)
class StepDef:
    id: str
    name: str
    script: str
    description: str


STEPS = {
    "1": StepDef(
        id="1",
        name="Download Data",
        script="src/01_download_data.py",
        description="Download GTFS, Census tracts, and ACS demographic data",
    ),
    "2": StepDef(
        id="2",
        name="Compute Supply (CPTA)",
        script="src/02_compute_supply.py",
        description="Calculate transit supply metrics and CPTA score",
    ),
    "2b": StepDef(
        id="2b",
        name="Compute Job Accessibility (optional)",
        script="src/02b_compute_jobs_accessibility.py",
        description="Calculate jobs accessible by transit (requires r5py + Java)",
    ),
    "3": StepDef(
        id="3",
        name="Compute Demand (TDI)",
        script="src/03_compute_demand.py",
        description="Calculate transit demand metrics and TDI score",
    ),
    "4": StepDef(
        id="4",
        name="Identify Transit Deserts",
        script="src/04_identify_deserts.py",
        description="Apply LISA clustering and classify transit deserts",
    ),
    "5": StepDef(
        id="5",
        name="Generate Visualizations",
        script="src/05_visualize.py",
        description="Create maps, figures, and summary tables",
    ),
}


# ---------------------------
# Printing helpers
# ---------------------------
def print_banner() -> None:
    print("=" * 70)
    print("  TRANSIT DESERT IDENTIFICATION PIPELINE")
    print("  A Geospatial Framework for Equity-Focused Service Gap Analysis")
    print("=" * 70)
    print()
    print("Pipeline Steps:")
    print("-" * 50)
    for k in ["1", "2", "2b", "3", "4", "5"]:
        if k in STEPS:
            s = STEPS[k]
            print(f"  {s.id}. {s.name}")
            print(f"     {s.description}")
    print("-" * 50)
    print()


# ---------------------------
# Cleaning (keep base folders)
# ---------------------------
def _empty_dir(dir_path: Path) -> None:
    """Delete everything inside dir_path, but keep the directory."""
    dir_path.mkdir(parents=True, exist_ok=True)
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except FileNotFoundError:
                pass


def clean_run_artifacts() -> None:
    """
    Clean previous run artifacts and cached inputs.
    Keeps directories but removes contents.
    """
    targets = [
        # raw inputs
        PROJECT_ROOT / "data" / "raw" / "gtfs",
        PROJECT_ROOT / "data" / "raw" / "acs",
        PROJECT_ROOT / "data" / "raw" / "census",
        # processed outputs
        PROJECT_ROOT / "data" / "processed",
        # viz outputs
        PROJECT_ROOT / "outputs" / "maps",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "outputs" / "tables",
    ]

    print("\n🧹 Cleaning previous run artifacts (keeping folders)...")
    for t in targets:
        _empty_dir(t)
        print(f"  ✓ Emptied: {t}")


# ---------------------------
# Logging
# ---------------------------
def get_run_log_path() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"pipeline_run_{ts}.log"


def log_header(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------
# Step execution (tee output to terminal + file)
# ---------------------------
def run_step(step_id: str, *, verbose: bool, log_path: Optional[Path]) -> bool:
    if step_id not in STEPS:
        raise ValueError(f"Unknown step: {step_id}")

    step = STEPS[step_id]
    script_path = PROJECT_ROOT / step.script

    if not script_path.exists():
        msg = f"✗ Script not found: {script_path}"
        print(msg)
        if log_path:
            log_header(log_path, msg)
        return False

    print("\n" + "=" * 60)
    print(f"STEP {step.id}: {step.name}")
    print("=" * 60)

    start_time = time.time()

    def log_write(line: str) -> None:
        if not log_path:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    cmd = [sys.executable, str(script_path)]
    log_write(f"\n\n=== STEP {step.id}: {step.name} ===\n")
    log_write(f"CMD: {' '.join(cmd)}\n\n")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            if verbose:
                print(line, end="")
            log_write(line)

        rc = proc.wait()
        elapsed = time.time() - start_time

        if rc == 0:
            msg = f"\n✓ Step {step.id} completed in {elapsed:.1f} seconds\n"
            print(msg)
            log_write(msg)
            return True
        else:
            msg = f"\n✗ Step {step.id} failed (exit code {rc})\n"
            print(msg)
            log_write(msg)
            return False

    except Exception as e:
        msg = f"\n✗ Error running step {step.id}: {e}\n"
        print(msg)
        log_write(msg)
        return False


# ---------------------------
# Pipeline orchestration
# ---------------------------
def parse_steps_arg(steps_str: Optional[str], include_jobs: bool) -> List[str]:
    """
    Default pipeline steps:
      1 -> 2 -> (2b optional) -> 3 -> 4 -> 5
    """
    if steps_str:
        steps = [s.strip().lower() for s in steps_str.split(",") if s.strip()]
    else:
        steps = ["1", "2", "3", "4", "5"]
        if include_jobs:
            # insert 2b right after 2
            steps = ["1", "2", "2b", "3", "4", "5"]

    # Basic validation
    for s in steps:
        if s not in STEPS:
            raise ValueError(f"Invalid step '{s}'. Valid: {', '.join(STEPS.keys())}")
    return steps


def run_pipeline(
    steps: List[str],
    *,
    verbose: bool,
    clean: bool,
    log_path: Optional[Path],
) -> bool:
    print_banner()

    if clean:
        clean_run_artifacts()

    if log_path:
        print(f"\n📝 Logging to: {log_path}")
        log_header(log_path, "=" * 70)
        log_header(log_path, "TRANSIT DESERT PIPELINE RUN LOG")
        log_header(log_path, f"Started: {datetime.now().isoformat(timespec='seconds')}")
        log_header(log_path, f"Steps: {', '.join(steps)}")
        log_header(log_path, "=" * 70 + "\n")

    print("\nSteps to run:", ", ".join(steps))

    print("\n" + "=" * 60)
    print("PIPELINE START")
    print("=" * 60)

    pipeline_start = time.time()
    step_status = {}

    for step_id in steps:
        ok = run_step(step_id, verbose=verbose, log_path=log_path)
        step_status[step_id] = ok
        if not ok:
            print("\n⚠ Pipeline stopped at step", step_id)
            break

    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for s in steps:
        status = "✓ Complete" if step_status.get(s) else ("✗ Failed" if s in step_status else "— Skipped")
        print(f"  Step {s}: {status}")
    print(f"\nTotal time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)\n")

    all_ok = all(step_status.get(s, False) for s in steps if s in step_status) and (
        len(step_status) == len(steps)
    )

    if log_path:
        log_header(log_path, "\n" + "=" * 70)
        log_header(log_path, f"Finished: {datetime.now().isoformat(timespec='seconds')}")
        log_header(log_path, f"Total time: {total_elapsed:.1f}s")
        log_header(log_path, f"Status: {'SUCCESS' if all_ok else 'FAILED'}")
        log_header(log_path, "=" * 70 + "\n")

    print("✓ Pipeline completed successfully" if all_ok else "✗ Pipeline completed with errors")
    return all_ok


# ---------------------------
# CLI
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Transit Desert pipeline")

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated steps to run. Example: 1,2,3,4,5 (default runs all)",
    )
    parser.add_argument(
        "--include-jobs",
        action="store_true",
        help="Include Step 2b (job accessibility). Default: off.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print step output to terminal (still logs to file).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete data/outputs from previous run.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a run log file.",
    )

    args = parser.parse_args()

    steps = parse_steps_arg(args.steps, include_jobs=args.include_jobs)
    log_path = None if args.no_log else get_run_log_path()

    # If quiet, we still want logging unless disabled
    verbose = not args.quiet
    clean = not args.no_clean

    ok = run_pipeline(steps, verbose=verbose, clean=clean, log_path=log_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()