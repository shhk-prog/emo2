#!/usr/bin/env python3
"""
scripts/reaggregate_v1_phase_a.py
V1 Phase A の幾何学指標再集計スクリプト (Item 4 準拠)
既存の e2_emobank_geometry.csv の r2_cross_r_to_s, r2_cross_s_to_r から
direct_transfer_score_raw, direct_transfer_score_clipped を算出し、
direct_transfer_score に生値を反映して再集計・保存する。
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    repo_root = Path(__file__).resolve().parent.parent
    derived_dir = repo_root / "v1" / "results" / "derived" / "v1_phase_a"

    model_dirs = [d for d in derived_dir.iterdir() if d.is_dir() and d.name != "dry_run"]
    print(f"Found {len(model_dirs)} models for Phase A reaggregation: {[d.name for d in model_dirs]}")

    for m_dir in model_dirs:
        e2_path = m_dir / "e2_emobank_geometry.csv"
        if not e2_path.exists():
            print(f"Skipping {m_dir.name}: e2_emobank_geometry.csv not found")
            continue

        df = pd.read_csv(e2_path)
        print(f"Processing {m_dir.name} ({len(df)} rows)...")

        # 生値の計算
        raw_scores = (df["r2_cross_r_to_s"].to_numpy() + df["r2_cross_s_to_r"].to_numpy()) / 2.0
        clipped_scores = np.clip(raw_scores, 0.0, None)

        df["direct_transfer_score_raw"] = raw_scores
        df["direct_transfer_score_clipped"] = clipped_scores
        # Primary direct_transfer_score は生値を保持 (Item 4)
        df["direct_transfer_score"] = raw_scores

        df.to_csv(e2_path, index=False)
        print(f"Updated {e2_path} with raw direct_transfer_scores (mean raw={np.mean(raw_scores):.4f}, clipped={np.mean(clipped_scores):.4f})")

    print("Phase A reaggregation completed successfully!")

if __name__ == "__main__":
    main()
