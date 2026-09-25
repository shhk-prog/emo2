#!/usr/bin/env python3
"""
scripts/package_submission_code.py

ICLR 2027 Supplementary Material 用のクリーンで完全匿名化された
コードパッケージ（ZIP）を生成し、自己検証（Double-blind, Provenance, pytest）を行うスクリプト。
"""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 匿名化で絶対に許容しない文字列パターン
PROHIBITED_PATTERNS = [
    "hiromi",
    "/mnt/nas",
    "/mnt/data",
    "iag-02",
]

# ルートから明示的に同梱する単体ファイル
ROOT_FILES_TO_INCLUDE = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".python-version",
    "setup_env.sh",
    "environment_setup_guide.md",
    "pytest.ini",
]

# 同梱するディレクトリルート
DIRECTORIES_TO_INCLUDE = [
    "src",
    "tests",
    "configs",
    "scripts",
    "behavioral",
    "v1",
    "v2",
    "v3",
    "results/derived/paper_summary",
]

# 除外パターン（パスのいずれかの部分に一致した場合にスキップ）
GLOBAL_EXCLUDE_DIR_NAMES = {
    ".git",
    ".agents",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "docs",
    "scratch",
    "iclr2027",
    "fairshare_gpu",
    "archive",
    "tools",
    "logs",
    "logs_old",
    "raw",  # results/raw や data/raw 等の生データ
}

GLOBAL_EXCLUDE_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.out",
    "*.err",
    ".DS_Store",
    "*~",
    "*.tmp",
    "*.zip",
    "*.tar.gz",
    "*.lock",
    "*.pt",
    "*.bin",
    "*.npy",
    "*.npz",
]


def should_include_path(rel_path: Path) -> bool:
    """指定された相対パスがZIP同梱対象かどうかを判定"""
    parts = rel_path.parts

    # 除外ディレクトリ名が含まれているか確認
    for part in parts:
        if part in GLOBAL_EXCLUDE_DIR_NAMES:
            return False

    # ファイル名パターンの除外確認
    filename = rel_path.name
    for pat in GLOBAL_EXCLUDE_FILE_PATTERNS:
        if fnmatch.fnmatch(filename, pat):
            return False

    # 49件の上流artifact (results/derived) は確実に含める
    if "results" in parts and "derived" in parts:
        return True

    # サブプロジェクト (behavioral, v1, v2, v3) の raw データは除外
    if "data" in parts and "raw" in parts:
        return False

    return True


def collect_files_to_package(root_dir: Path) -> list[Path]:
    """パッケージに含めるファイル一覧を決定論的に高速収集（不要ディレクトリ枝刈り付き）"""
    files_to_pack = set()

    # 1. ルート単体ファイル
    for rf in ROOT_FILES_TO_INCLUDE:
        p = root_dir / rf
        if p.exists() and p.is_file():
            files_to_pack.add(p.relative_to(root_dir))

    # 2. 指定ディレクトリ配下
    for d in DIRECTORIES_TO_INCLUDE:
        target_dir = root_dir / d
        if not target_dir.exists():
            continue
        if target_dir.is_file():
            files_to_pack.add(target_dir.relative_to(root_dir))
            continue

        for root, dirs, files in os.walk(target_dir):
            rel_root = Path(root).relative_to(root_dir)
            
            # ディレクトリの枝刈り (GLOBAL_EXCLUDE_DIR_NAMES)
            dirs[:] = [d_name for d_name in dirs if d_name not in GLOBAL_EXCLUDE_DIR_NAMES]

            for fname in files:
                rel_fpath = rel_root / fname
                if should_include_path(rel_fpath):
                    files_to_pack.add(rel_fpath)

    return sorted(list(files_to_pack))


def create_zip_archive(files: list[Path], root_dir: Path, out_zip: Path):
    """ファイル一覧からZIPアーカイブを作成"""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    print(f"Creating ZIP archive at {out_zip} ...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel_p in files:
            full_p = root_dir / rel_p
            # 決定論的タイムスタンプ（再現可能なZIP）
            zinfo = zipfile.ZipInfo.from_file(full_p, arcname=str(rel_p))
            # 2026-09-25 00:00:00 に固定
            zinfo.date_time = (2026, 9, 25, 0, 0, 0)
            with open(full_p, "rb") as f:
                zf.writestr(zinfo, f.read())

    sz_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Archive created: {len(files)} files, {sz_mb:.2f} MB")


def validate_extracted_package(extract_dir: Path):
    """解凍されたコードパッケージを厳格に自己検証"""
    print("\n--- Running Package Validations ---")

    # 1. 二重盲検（Double-blind）検査
    print("[1/4] Checking Double-blind Anonymity...")
    violations = []
    text_extensions = {".py", ".sh", ".sbatch", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".csv"}

    for p in extract_dir.rglob("*"):
        if p.is_file() and p.suffix in text_extensions:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for pat in PROHIBITED_PATTERNS:
                    if pat in content:
                        violations.append((p.relative_to(extract_dir), pat))
            except Exception as e:
                print(f"Warning: could not read {p}: {e}")

    if violations:
        print("ERROR: Double-blind violations detected!")
        for rel_p, pat in violations[:20]:
            print(f"  VIOLATION: {rel_p} contains prohibited '{pat}'")
        raise RuntimeError(f"Package contains {len(violations)} anonymity violations!")
    print("PASS: Double-blind anonymity check passed (0 violations found).")

    # 2. 49件の上流artifact存在検査
    print("[2/4] Checking 49 Upstream Manifest Artifacts...")
    manifest_path = extract_dir / "results" / "derived" / "paper_summary" / "paper_summary_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = sorted(set(
        s["artifact"]
        for entry in manifest.get("provenance_map", {}).values()
        for s in entry.get("sources", [])
        if s.get("artifact")
    ))

    missing_arts = []
    for art in artifacts:
        if not (extract_dir / art).exists():
            missing_arts.append(art)

    if missing_arts:
        print(f"ERROR: {len(missing_arts)} / {len(artifacts)} upstream artifacts missing!")
        for m in missing_arts[:10]:
            print(f"  MISSING: {m}")
        raise RuntimeError("Missing upstream derived artifacts in package!")
    print(f"PASS: All {len(artifacts)} upstream derived artifacts are present.")

    # 3. scripts/build_all_paper_summaries.py --strict 実行検証
    print("[3/4] Testing scripts/build_all_paper_summaries.py --strict in extracted dir...")
    res = subprocess.run(
        [sys.executable, "scripts/build_all_paper_summaries.py", "--strict"],
        cwd=extract_dir,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"ERROR: build_all_paper_summaries.py failed with returncode {res.returncode}")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError("build_all_paper_summaries.py --strict failed!")
    print("PASS: Paper summary builds successfully with --strict.")

    # 4. 全件 pytest テスト
    print("[4/4] Running pytest suite in extracted dir...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(extract_dir / "src")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=extract_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"ERROR: pytest failed with returncode {res.returncode}")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError("pytest suite failed in extracted package!")
    print("PASS: Pytest suite completed with 0 failures!")
    print("Pytest output summary:\n", res.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description="Package and Validate Anonymous Submission Code")
    parser.add_argument(
        "--out",
        type=str,
        default="results/submission/supplementary_code.zip",
        help="Output ZIP file path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip extraction and self-validation",
    )
    args = parser.parse_args()

    out_zip = (PROJECT_ROOT / args.out).resolve()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output target: {out_zip}")

    files = collect_files_to_package(PROJECT_ROOT)
    print(f"Found {len(files)} files matching packaging criteria.")

    create_zip_archive(files, PROJECT_ROOT, out_zip)

    if not args.skip_validation:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            print(f"\nExtracting archive to temporary directory: {tmp_path} ...")
            with zipfile.ZipFile(out_zip, "r") as zf:
                zf.extractall(tmp_path)

            validate_extracted_package(tmp_path)

    print("\n" + "=" * 60)
    print("SUCCESS: Anonymous Submission Code Package is Ready!")
    print(f"Archive: {out_zip}")
    print(f"Size:    {out_zip.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
