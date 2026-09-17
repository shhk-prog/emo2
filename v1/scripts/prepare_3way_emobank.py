#!/usr/bin/env python3
"""
Prepare 3-way EmoBank dataset (VAD for both Reader and Writer perspectives).
Downloads reader.csv and writer.csv from official JULIELab/EmoBank repository
and merges them with existing 321 stimuli to produce stimuli_vad_3way.csv.
"""

import os
import urllib.request
import pandas as pd

RAW_DIR = "v1/data/raw"
PROCESSED_DIR = "v1/data/processed"
STIMULI_PATH = os.path.join(PROCESSED_DIR, "stimuli.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "stimuli_vad_3way.csv")

READER_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/reader.csv"
WRITER_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/writer.csv"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev_test", "all", "321"],
                        help="Data split to use: 'test' (N~1000, official test benchmark), 'dev_test' (N~2000), 'all' (N~10000), '321' (legacy subset)")
    args = parser.parse_args()
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    reader_raw = os.path.join(RAW_DIR, "emobank_reader.csv")
    writer_raw = os.path.join(RAW_DIR, "emobank_writer.csv")
    emobank_main = os.path.join(RAW_DIR, "emobank.csv")
    
    if not os.path.exists(reader_raw):
        print(f"Downloading {READER_URL} -> {reader_raw}...")
        urllib.request.urlretrieve(READER_URL, reader_raw)
    else:
        print(f"Using existing {reader_raw}")
        
    if not os.path.exists(writer_raw):
        print(f"Downloading {WRITER_URL} -> {writer_raw}...")
        urllib.request.urlretrieve(WRITER_URL, writer_raw)
    else:
        print(f"Using existing {writer_raw}")
        
    print("Loading datasets...")
    df_reader = pd.read_csv(reader_raw)
    df_writer = pd.read_csv(writer_raw)
    df_main = pd.read_csv(emobank_main)
    
    df_reader = df_reader.rename(columns={"V": "reader_V", "A": "reader_A", "D": "reader_D"})
    df_writer = df_writer.rename(columns={"V": "writer_V", "A": "writer_A", "D": "writer_D"})
    
    # Filter by split
    if args.split == "test":
        df_base = df_main[df_main["split"] == "test"].copy()
        split_suffix = "test1k"
    elif args.split == "dev_test":
        df_base = df_main[df_main["split"].isin(["dev", "test"])].copy()
        split_suffix = "dev_test2k"
    elif args.split == "all":
        df_base = df_main.copy()
        split_suffix = "all10k"
    elif args.split == "321":
        df_base = pd.read_csv(STIMULI_PATH)
        split_suffix = "321"
    else:
        df_base = df_main[df_main["split"] == "test"].copy()
        split_suffix = "test1k"
        
    print(f"Base stimuli count for split='{args.split}' ({split_suffix}): {len(df_base)}")
    
    # Merge reader and writer on 'id'
    merged = pd.merge(df_base, df_reader[["id", "reader_V", "reader_A", "reader_D"]], on="id", how="inner")
    merged = pd.merge(merged, df_writer[["id", "writer_V", "writer_A", "writer_D"]], on="id", how="inner")
    
    # Add explicit stimulus_id if not present
    if "stimulus_id" not in merged.columns:
        merged["stimulus_id"] = "emobank_" + merged["id"].astype(str)
        
    out_specific = os.path.join(PROCESSED_DIR, f"stimuli_vad_3way_{split_suffix}.csv")
    merged.to_csv(out_specific, index=False)
    # Default pointer
    merged.to_csv(OUT_PATH, index=False)
    
    print(f"Successfully saved {len(merged)} 3-way VAD stimuli to:\n  - {out_specific}\n  - {OUT_PATH}")
    print("\nSample rows:")
    print(merged[["id", "text", "reader_V", "reader_A", "reader_D", "writer_V", "writer_A", "writer_D"]].head())

if __name__ == "__main__":
    main()
