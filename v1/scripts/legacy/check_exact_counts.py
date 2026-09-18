#!/usr/bin/env python3
import pandas as pd

with open("v1/data/processed/aipsy_4split_all.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

df = pd.read_csv("v1/data/processed/aipsy_4split_all.csv")

with open("v1/scripts/exact_counts_out.txt", "w", encoding="utf-8") as out:
    out.write(f"Total lines in file (including header): {len(lines)}\n")
    out.write(f"Header line: {lines[0].strip()}\n")
    out.write(f"Data rows in pandas DataFrame: {len(df)}\n")
    out.write(f"Index range: 0 to {len(df)-1}\n")
    out.write(f"Splits breakdown:\n{df['split'].value_counts().to_string()}\n")
    out.write(f"Total rows sum of splits: {df['split'].value_counts().sum()}\n")
    out.write(f"Unique IDs: {df['id'].nunique()}\n")
