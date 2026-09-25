import json
from pathlib import Path

PROJECT_ROOT = Path("/mnt/nas/home/hiromi/src/emo2")
manifest_path = PROJECT_ROOT / "results" / "derived" / "paper_summary" / "paper_summary_manifest.json"

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

provenance_map = manifest.get("provenance_map", {})

all_artifacts = set()
for rec_id, entry in provenance_map.items():
    for src in entry.get("sources", []):
        art = src.get("artifact")
        if art:
            all_artifacts.add(art)

print(f"Total unique source artifacts: {len(all_artifacts)}")

total_bytes = 0
for art in sorted(all_artifacts):
    p = PROJECT_ROOT / art
    sz = p.stat().st_size if p.exists() else -1
    if sz >= 0:
        total_bytes += sz
    print(f"{sz:10d} bytes | {art}")

print(f"\nTotal size of all 49 upstream artifacts: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
