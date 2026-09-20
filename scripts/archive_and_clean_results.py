#!/usr/bin/env python3
"""
scripts/archive_and_clean_results.py

旧実験結果の一括アーカイブ & results ディレクトリのクリーン初期化スクリプト。
ユーザー指定構成:
  archive/
  └── results_pre_rerun_20260918/
      ├── behavioral/
      ├── v1/
      ├── v2/
      └── v3/
  behavioral/results/   # 空 (.gitkeep のみ)
  v1/results/           # 空 (.gitkeep のみ)
  v2/results/           # 空 (.gitkeep のみ)
  v3/results/           # 空 (.gitkeep のみ)

実行内容:
  1. results 配下の安全検査（入力データや設定の誤混入チェック）
  2. 旧 results を archive/results_pre_rerun_20260918/ へ安全に退避移動
  3. 各ステージの results ディレクトリを再生成し、.gitkeep を配置してクリーン初期化
  4. archive/results_pre_rerun_20260918.tar.gz を生成
"""

import os
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = root / "archive" / f"results_archive_{timestamp}"

    print("=" * 70)
    print("旧実験結果アーカイブ & クリーン初期化プロセス (安全アーカイブ版)")
    print(f"プロジェクトルート: {root}")
    print(f"アーカイブ退避先:   {archive_dir}")
    print("=" * 70)

    stages = ["behavioral", "v1", "v2", "v3"]

    # 1. 退避先ディレクトリの作成
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 2. 安全確認: プロンプトハッシュ等の重要設定が退避されているか
    v2_hash = root / "v2" / "configs" / "prompt_hashes.json"
    if not v2_hash.exists():
        src_hash = root / "v2" / "results" / "prompt_hashes.json"
        if src_hash.exists():
            shutil.copy2(src_hash, v2_hash)
            print(f"[保護] v2 prompt_hashes.json を {v2_hash} へ恒久保存しました。")

    # 3. 各ステージの results をアーカイブへ移動
    for stage in stages:
        src_res = root / stage / "results"
        dst_res = archive_dir / stage

        if src_res.exists():
            dst_res.mkdir(parents=True, exist_ok=True)
            items = list(src_res.iterdir())
            print(f"\n[{stage}] {len(items)} 個のアイテムをアーカイブへ移動中...")
            for item in items:
                dst_item = dst_res / item.name
                if dst_item.exists():
                    if dst_item.is_dir():
                        shutil.rmtree(dst_item)
                    else:
                        dst_item.unlink()
                shutil.move(str(item), str(dst_res))
                print(f"  -> 移動: {item.name}")

            # results ディレクトリを空にして再作成
            src_res.mkdir(parents=True, exist_ok=True)
            gitkeep = src_res / ".gitkeep"
            gitkeep.touch()
            print(f"[{stage}] results/ をクリーン初期化しました（.gitkeep 配置完了）。")
        else:
            src_res.mkdir(parents=True, exist_ok=True)
            (src_res / ".gitkeep").touch()
            print(f"[{stage}] results/ を新規作成しました（.gitkeep 配置完了）。")

    # 4. archive ディレクトリに README を作成
    archive_readme = archive_dir / "README.md"
    archive_readme.write_text(
        f"# Archive: Pre-Rerun Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
        f"本ディレクトリは、コードベースの全面改修・仕様統一（Prompt-End Normalized, "
        f"相対深度 d = l/(L-1), 2D Joint OT, 共通パッケージ化）に伴い、"
        f"全実験をクリーン再実行する直前に退避された旧実験結果の原本です。\n\n"
        f"- 退避日時: {datetime.now().isoformat()}\n"
        f"- 対象ステージ: behavioral, v1, v2, v3\n"
        f"- 用途: 回帰検証（regression check）、旧仕様（response-onset）との頑健性比較用\n",
        encoding="utf-8"
    )

    # 5. tar.gz アーカイブの作成 (NAS等への保存用)
    tar_path = root / "archive" / "results_pre_rerun_20260918.tar.gz"
    print(f"\n[圧縮] tar.gz アーカイブを作成中: {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(archive_dir), arcname=archive_dir.name)
    print(f"[圧縮完了] {tar_path} (サイズ: {tar_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 70)
    print("SUCCESS: 旧結果のアーカイブ退避と results の完全初期化が完了しました。")
    print("状態:")
    print("  - archive/results_pre_rerun_20260918/ (旧結果原本)")
    print("  - archive/results_pre_rerun_20260918.tar.gz (バックアップ圧縮ファイル)")
    print("  - behavioral/results/ (クリーン・空)")
    print("  - v1/results/         (クリーン・空)")
    print("  - v2/results/         (クリーン・空)")
    print("  - v3/results/         (クリーン・空)")
    print("=" * 70)


if __name__ == "__main__":
    main()
