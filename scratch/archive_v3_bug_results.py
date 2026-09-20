#!/usr/bin/env python3
import shutil
from pathlib import Path

dest_dir = Path("archive/results_v3_double_softmax_bug_20260920")
dest_dir.mkdir(parents=True, exist_ok=True)

raw_src = Path("v3/results/raw")
derived_src = Path("v3/results/derived")

if raw_src.exists():
    shutil.copytree(raw_src, dest_dir / "raw", dirs_exist_ok=True)
    print(f"Copied {raw_src} -> {dest_dir / 'raw'}")

if derived_src.exists():
    shutil.copytree(derived_src, dest_dir / "derived", dirs_exist_ok=True)
    print(f"Copied {derived_src} -> {dest_dir / 'derived'}")

# Remove old results so cache is invalidated
for target in [
    raw_src / "v3_rq1_results.json",
    derived_src / "v3_gate_decision.json",
]:
    if target.exists():
        target.unlink()
        print(f"Removed old target file: {target}")

print("V3 archival and cache invalidation completed successfully.")
