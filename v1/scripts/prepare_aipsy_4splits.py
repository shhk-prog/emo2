#!/usr/bin/env python3
"""
Prepares the complete 4-split AIPsy-Affect dataset (N=480) with explicit
pairings for:
  - 192 Minimal Pairs: Clinical (Peak) vs. Matched Neutral (None)
  - 48 Matched Triplets: Neutral (None) -> Moderate (Moderate) -> Clinical (Peak)
  - 48 Complexity Controls: Complex Neutral
"""

import os
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

def load_split_arrow(cache_dir, split_name):
    arrow_path = os.path.join(cache_dir, f"aipsy-affect-{split_name}.arrow")
    if not os.path.exists(arrow_path):
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")
    with pa.memory_map(arrow_path, "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    df = table.to_pandas()
    df["split"] = split_name
    return df

def main():
    cache_dir = "v1/data/raw/keidolabs___aipsy-affect/default/0.0.0/1e24598f2041e86f788d1974cb3661b67d236fe4"
    out_dir = "v1/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "aipsy_4split_all.csv")

    print("Loading 4 splits from arrow cache...")
    df_clin = load_split_arrow(cache_dir, "clinical")
    df_neut = load_split_arrow(cache_dir, "neutral")
    df_mod = load_split_arrow(cache_dir, "moderate")
    df_comp = load_split_arrow(cache_dir, "complex_neutral")

    print(f"Loaded counts: Clinical={len(df_clin)}, Neutral={len(df_neut)}, Moderate={len(df_mod)}, ComplexNeutral={len(df_comp)}")

    # 1. Setup Pair IDs for Clinical <-> Neutral (192 pairs)
    # Clinical id: B-rage-d1-v1 -> matched_control: N-rage-d1-c1
    # Pair ID can be derived from clinical id, e.g. "pair_rage_d1_v1"
    df_clin["pair_id"] = df_clin["id"].apply(lambda x: "P_" + str(x)[2:])
    
    # Neutral matched_control_id points back to clinical id
    df_neut["pair_id"] = df_neut["matched_control_id"].apply(lambda x: "P_" + str(x)[2:] if pd.notna(x) else None)
    
    # Also for neutral, infer original emotion category from matched clinical stimulus
    clin_id_to_emotion = dict(zip(df_clin["id"], df_clin["emotion"]))
    df_neut["target_emotion"] = df_neut["matched_control_id"].map(clin_id_to_emotion)
    df_clin["target_emotion"] = df_clin["emotion"]

    # 2. Setup Triplet IDs for Neutral -> Moderate -> Clinical (48 triplets)
    # Moderate id: Bm-rage-d1-v1 -> corresponding clinical: B-rage-d1-v1, corresponding neutral: N-rage-d1-c1
    mod_id_to_triplet = {}
    clin_id_to_triplet = {}
    neut_id_to_triplet = {}
    
    for _, row in df_mod.iterrows():
        mid = str(row["id"]) # e.g. Bm-rage-d1-v1
        base_suffix = mid[3:] # rage-d1-v1
        triplet_id = f"T_{base_suffix}"
        mod_id_to_triplet[mid] = triplet_id
        
        # Clinical counterpart: B-rage-d1-v1
        cid = f"B-{base_suffix}"
        clin_id_to_triplet[cid] = triplet_id
        
        # Neutral counterpart: N-rage-d1-c1 (where version v1 maps to c1)
        # In AIPsy, the format is c{version_num}
        parts = base_suffix.rsplit("-v", 1)
        if len(parts) == 2:
            nid = f"N-{parts[0]}-c{parts[1]}"
            neut_id_to_triplet[nid] = triplet_id

    df_mod["triplet_id"] = df_mod["id"].map(mod_id_to_triplet)
    df_mod["target_emotion"] = df_mod["emotion"]
    df_clin["triplet_id"] = df_clin["id"].map(clin_id_to_triplet)
    df_neut["triplet_id"] = df_neut["id"].map(neut_id_to_triplet)

    # 3. Complex Neutral
    df_comp["pair_id"] = None
    df_comp["triplet_id"] = None
    df_comp["target_emotion"] = "neutral"

    # Combine all
    all_df = pd.concat([df_clin, df_neut, df_mod, df_comp], ignore_index=True)
    all_df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved {len(all_df)} stimuli to {out_csv}")
    print("=" * 60)
    print("Split breakdown:")
    print(all_df["split"].value_counts())
    print("\nMatched Pairs count (Clinical <-> Neutral):", all_df["pair_id"].nunique())
    print("Matched Triplets count (Neutral -> Moderate -> Clinical):", all_df["triplet_id"].nunique())
    print("\nSample Triplet row count check:")
    t_counts = all_df[all_df["triplet_id"].notna()]["triplet_id"].value_counts()
    print(f"Number of triplets with exactly 3 conditions: {(t_counts == 3).sum()} / {len(t_counts)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
