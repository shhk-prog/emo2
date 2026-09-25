import os
from pathlib import Path

PROJECT_ROOT = Path("/mnt/nas/home/hiromi/src/emo2")

PATTERNS = ["hiromi", "/mnt/nas", "/mnt/data", "iag-02"]

IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
}

results = []

for root, dirs, files in os.walk(PROJECT_ROOT):
    # filter ignore dirs
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    rel_root = Path(root).relative_to(PROJECT_ROOT)

    for f in files:
        if f.endswith((".pyc", ".png", ".jpg", ".pdf", ".zip", ".tar.gz", ".npy", ".pt", ".bin")):
            continue
        fpath = Path(root) / f
        rel_fpath = fpath.relative_to(PROJECT_ROOT)

        # skip docs/prepare_anonymized_submission_zip itself
        if "docs/prepare_anonymized_submission_zip" in str(rel_fpath):
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pat in PATTERNS:
            if pat in text:
                # count occurrences
                cnt = text.count(pat)
                results.append((str(rel_fpath), pat, cnt))

print(f"Total files with identifying strings: {len(set(r[0] for r in results))}")
print("-" * 80)
files_dict = {}
for rel_path, pat, cnt in results:
    if rel_path not in files_dict:
        files_dict[rel_path] = []
    files_dict[rel_path].append(f"{pat} ({cnt})")

for path, info in sorted(files_dict.items()):
    print(f"{path}: {', '.join(info)}")
