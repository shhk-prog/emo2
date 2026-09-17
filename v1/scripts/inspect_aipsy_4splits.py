#!/usr/bin/env python3
import os
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

def load_aipsy_4splits():
    cache_dir = "v1/data/raw/keidolabs___aipsy-affect/default/0.0.0/1e24598f2041e86f788d1974cb3661b67d236fe4"
    splits = ["clinical", "neutral", "moderate", "complex_neutral"]
    dfs = []
    for s in splits:
        arrow_file = os.path.join(cache_dir, f"aipsy-affect-{s}.arrow")
        with pa.memory_map(arrow_file, "r") as source:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        df = table.to_pandas()
        df["split"] = s
        dfs.append(df)
        print(f"Loaded {s}: {len(df)} rows")
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

if __name__ == "__main__":
    df = load_aipsy_4splits()
    print("Total rows:", len(df))
    print("\nSplit x Intensity:")
    print(pd.crosstab(df["split"], df["intensity"]))
    print("\nSplit x Emotion:")
    print(pd.crosstab(df["split"], df["emotion"]))
    print("\nWord count stats by split:")
    print(df.groupby("split")["word_count"].agg(["count", "mean", "std", "min", "max"]))
