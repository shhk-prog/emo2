from pathlib import Path

root = Path("/mnt/nas/home/hiromi/src/emo2")

for d in ["v1/data", "v2/data", "v3/data", "behavioral/data"]:
    p = root / d
    if not p.exists():
        print(f"Directory {d} does not exist", flush=True)
        continue
    files = list(p.rglob("*"))
    file_objs = [f for f in files if f.is_file()]
    tot_sz = sum(f.stat().st_size for f in file_objs)
    print(f"Directory {d}: {len(file_objs)} files, {tot_sz:,} bytes ({tot_sz / 1024 / 1024:.2f} MB)", flush=True)
    for f in sorted(file_objs, key=lambda x: x.stat().st_size, reverse=True)[:5]:
        print(f"  {f.stat().st_size:10d} bytes | {f.relative_to(root)}", flush=True)
