"""
scripts/cleanup_project_files.py

Clean up Mac garbage files, move reporting scripts to tools/reporting,
move fairshare_gpu to docs/infrastructure, clean scratch, and archive old docs.
"""
import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]

print("=== Starting Project File Cleanup ===")

# 1. Remove .DS_Store and ._* files
deleted_garbage_count = 0
for p in root.rglob(".DS_Store"):
    try:
        p.unlink()
        deleted_garbage_count += 1
    except Exception as e:
        print(f"Error deleting {p}: {e}")

for p in root.rglob("._*"):
    try:
        p.unlink()
        deleted_garbage_count += 1
    except Exception as e:
        print(f"Error deleting {p}: {e}")

print(f"[1] Deleted {deleted_garbage_count} garbage files (.DS_Store, ._*)")

# 2. Clean scratch directory
scratch_dir = root / "scratch"
if scratch_dir.exists():
    shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(exist_ok=True)
    (scratch_dir / ".gitkeep").touch()
    print("[2] Cleaned scratch/ directory")

# 3. Move root reporting scripts to tools/reporting/
tools_reporting = root / "tools" / "reporting"
tools_reporting.mkdir(parents=True, exist_ok=True)

reporting_scripts = ["get_stats.py", "get_md_tables.py", "get_arousal_stats.py"]
for script_name in reporting_scripts:
    script_path = root / script_name
    if script_path.exists():
        dst = tools_reporting / script_name
        shutil.move(str(script_path), str(dst))
        print(f"[3] Moved {script_name} -> tools/reporting/{script_name}")

# 4. Move fairshare_gpu to docs/infrastructure/
fairshare = root / "fairshare_gpu"
if fairshare.exists():
    infra_dir = root / "docs" / "infrastructure"
    infra_dir.mkdir(parents=True, exist_ok=True)
    dst_fairshare = infra_dir / "fairshare_gpu"
    if dst_fairshare.exists():
        shutil.rmtree(dst_fairshare)
    shutil.move(str(fairshare), str(dst_fairshare))
    print("[4] Moved fairshare_gpu/ -> docs/infrastructure/fairshare_gpu/")

# 5. Archive legacy documentation in docs/
docs_dir = root / "docs"
docs_archive = docs_dir / "archive"
docs_archive.mkdir(parents=True, exist_ok=True)

# Keep current critical documentation folders
keep_docs = {
    "archive",
    "infrastructure",
    "repository_consolidation_and_pipeline_unification",
}

archived_docs_count = 0
for item in docs_dir.iterdir():
    if item.name not in keep_docs and not item.name.startswith("."):
        dst = docs_archive / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(item), str(docs_archive))
        archived_docs_count += 1

print(f"[5] Moved {archived_docs_count} legacy docs items into docs/archive/")

print("=== Project Cleanup Finished Successfully ===")
