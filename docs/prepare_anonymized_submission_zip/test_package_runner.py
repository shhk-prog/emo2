import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path("/mnt/nas/home/hiromi/src/emo2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import package_submission_code as psc

print("--- Testing Packaging Logic Directly ---")
files = psc.collect_files_to_package(PROJECT_ROOT)
print(f"Total files collected for packaging: {len(files)}")

# Check if any sensitive files or directories leaked into files list
leaks = []
for f in files:
    s = str(f)
    if "iclr2027" in s:
        leaks.append(f"iclr2027 leak: {f}")
    if "docs" in s and not s.startswith("v"):  # docs/ directly
        leaks.append(f"docs leak: {f}")
    if "scratch" in s:
        leaks.append(f"scratch leak: {f}")
    if "raw" in s and "data/raw" in s:
        leaks.append(f"data/raw leak: {f}")
    if f.name == "package_submission_code.py":
        leaks.append(f"package_submission_code.py leak: {f}")

if leaks:
    print("LEAKS FOUND in collection list:")
    for l in leaks[:10]:
        print(" ", l)
else:
    print("SUCCESS: Zero sensitive files/directories in collection list!")

# Verify 49 upstream artifacts are present in files list
manifest_path = PROJECT_ROOT / "results" / "derived" / "paper_summary" / "paper_summary_manifest.json"
import json
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
upstream_arts = set(
    s["artifact"]
    for e in manifest.get("provenance_map", {}).values()
    for s in e.get("sources", [])
    if s.get("artifact")
)
missing_in_pack = [a for a in upstream_arts if Path(a) not in files]
print(f"Upstream artifacts in manifest: {len(upstream_arts)}")
print(f"Upstream artifacts included in package: {len(upstream_arts) - len(missing_in_pack)} / {len(upstream_arts)}")
if missing_in_pack:
    print("MISSING upstream artifacts in package collection:")
    for m in missing_in_pack[:10]:
        print(" ", m)
else:
    print("SUCCESS: 100% of 49 upstream artifacts are included in package collection!")
