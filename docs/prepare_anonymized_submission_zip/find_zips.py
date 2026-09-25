from pathlib import Path

home = Path("/mnt/nas/home/hiromi")
for p in home.glob("*.zip"):
    print(f"Found zip: {p} ({p.stat().st_size:,} bytes)")

for p in (home / "Downloads").glob("*.zip") if (home / "Downloads").exists() else []:
    print(f"Found zip in Downloads: {p} ({p.stat().st_size:,} bytes)")
