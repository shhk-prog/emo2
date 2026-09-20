#!/usr/bin/env python3
"""
v1/primary/phase_c/select_e4_sites.py

Deterministic candidate layer selection for E4 Causal Interchangeability
strictly based on the Discovery split of E3 Causal Map.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def select_e4_candidate_layers(
    e3_csv_path: str, num_layers: int = 28
) -> list[int]:
    df = pd.read_csv(e3_csv_path)

    # Use discovery_mag if available, else overall magnitude
    mag_r_col = (
        "discovery_mag_reader"
        if "discovery_mag_reader" in df.columns
        else "magnitude_reader"
    )
    mag_s_col = (
        "discovery_mag_self"
        if "discovery_mag_self" in df.columns
        else "magnitude_self"
    )

    valid_r = df[mag_r_col].dropna() if mag_r_col in df.columns else pd.Series(dtype=float)
    valid_s = df[mag_s_col].dropna() if mag_s_col in df.columns else pd.Series(dtype=float)

    reader_peak_l = int(df.loc[valid_r.idxmax(), "layer"]) if not valid_r.empty else max(0, int(num_layers * 0.3))
    self_peak_l = int(df.loc[valid_s.idxmax(), "layer"]) if not valid_s.empty else max(0, int(num_layers * 0.7))

    near_l1 = max(0, min(reader_peak_l - 1, num_layers - 1))
    near_l2 = max(0, min(self_peak_l + 1, num_layers - 1))
    late_l = int(num_layers * 0.85)

    candidate_layers = sorted(
        list(set([reader_peak_l, self_peak_l, near_l1, near_l2, late_l]))
    )
    return candidate_layers


def main():
    parser = argparse.ArgumentParser(
        description="Select E4 Candidate Layers from E3 Discovery Split"
    )
    parser.add_argument(
        "--e3-csv",
        type=str,
        required=True,
        help="Path to e3_causal_map.csv",
    )
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    candidates = select_e4_candidate_layers(args.e3_csv, args.num_layers)
    print(f"Selected E4 candidate layers: {candidates}")

    out_json = args.out_json
    if out_json is None:
        out_json = str(Path(args.e3_csv).parent / "e4_selected_candidates.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_e3_csv": args.e3_csv,
                "num_layers": args.num_layers,
                "selected_candidate_layers": candidates,
            },
            f,
            indent=2,
        )
    print(f"Saved E4 candidate configuration to {out_json}")


if __name__ == "__main__":
    main()
