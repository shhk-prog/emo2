import os
from pathlib import Path

root = Path("/mnt/nas/home/hiromi/src/emo2")
subdirs = ["src", "configs", "scripts", "tests", "behavioral", "v1", "v2", "v3", "data"]
patterns = ["hiromi", "/mnt/nas", "/mnt/data", "iag-02"]

for s in subdirs:
    p = root / s
    if not p.exists():
        continue
    for fpath in p.rglob("*"):
        if fpath.is_file() and not fpath.name.endswith((".pyc", ".png", ".jpg", ".npy", ".pt", ".bin")):
            try:
                txt = fpath.read_text(encoding="utf-8", errors="ignore")
                for pat in patterns:
                    if pat in txt:
                        print(f"{fpath.relative_to(root)} contains {pat}", flush=True)
            except Exception as e:
                pass
