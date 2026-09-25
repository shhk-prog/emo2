import json
from pathlib import Path
import sys

PROJECT_ROOT = Path("/mnt/nas/home/hiromi/src/emo2")
sys.path.insert(0, str(PROJECT_ROOT))

# package_submission_code の関数をインポート
import scripts.package_submission_code as psc

files = psc.collect_files_to_package(PROJECT_ROOT)
print(f"Total collected files: {len(files)}")

# 49件のartifactが含まれているか確認
manifest_path = PROJECT_ROOT / "results" / "derived" / "paper_summary" / "paper_summary_manifest.json"
manifest = json.loads(manifest_path.read_text())
artifacts = sorted(set(
    s["artifact"]
    for entry in manifest.get("provenance_map", {}).values()
    for s in entry.get("sources", [])
    if s.get("artifact")
))

missing_in_collection = []
for a in artifacts:
    if Path(a) not in files:
        missing_in_collection.append(a)

print(f"Total artifacts required: {len(artifacts)}")
print(f"Artifacts missing from collection: {len(missing_in_collection)}")
if missing_in_collection:
    for m in missing_in_collection:
        print(f"  MISSING: {m}")
else:
    print("ALL 49 UPSTREAM ARTIFACTS ARE INCLUDED IN COLLECTION!")
