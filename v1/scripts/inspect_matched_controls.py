#!/usr/bin/env python3
import os
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

cache_dir = "v1/data/raw/keidolabs___aipsy-affect/default/0.0.0/1e24598f2041e86f788d1974cb3661b67d236fe4"
splits = ["clinical", "neutral", "moderate", "complex_neutral"]
dfs = {}
for s in splits:
    arrow_file = os.path.join(cache_dir, f"aipsy-affect-{s}.arrow")
    with pa.memory_map(arrow_file, "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    df = table.to_pandas()
    df["split"] = s
    dfs[s] = df

print("--- Clinical Samples (head 3) ---")
print(dfs["clinical"][["id", "emotion", "intensity", "matched_control_id"]].head(3))

print("\n--- Neutral Samples matching Clinical (head 3) ---")
c_matches = dfs["clinical"]["matched_control_id"].head(3).tolist()
print(dfs["neutral"][dfs["neutral"]["id"].isin(c_matches)][["id", "emotion", "intensity", "matched_control_id"]])

print("\n--- Moderate Samples (head 3) ---")
print(dfs["moderate"][["id", "emotion", "intensity", "matched_control_id"]].head(3))
m_matches = dfs["moderate"]["matched_control_id"].head(3).tolist()
print("Moderate matched controls in neutral or complex_neutral:")
print(dfs["neutral"][dfs["neutral"]["id"].isin(m_matches)][["id", "emotion", "matched_control_id"]])
print(dfs["complex_neutral"][dfs["complex_neutral"]["id"].isin(m_matches)][["id", "emotion", "matched_control_id"]])

print("\n--- Complex Neutral Samples (head 3) ---")
print(dfs["complex_neutral"][["id", "emotion", "intensity", "matched_control_id"]].head(3))
cn_matches = dfs["complex_neutral"]["matched_control_id"].head(3).tolist()
print("Complex neutral matched controls:")
print(dfs["neutral"][dfs["neutral"]["id"].isin(cn_matches)][["id", "emotion", "matched_control_id"]])
print(dfs["clinical"][dfs["clinical"]["id"].isin(cn_matches)][["id", "emotion", "matched_control_id"]])
