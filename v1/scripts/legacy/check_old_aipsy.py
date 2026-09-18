#!/usr/bin/env python3
import pandas as pd
import glob

print("Checking v1/data/processed/aipsy/*.csv:")
for f in sorted(glob.glob("v1/data/processed/aipsy/*.csv")):
    df = pd.read_csv(f)
    print(f"  {f}: {len(df)} rows")

with open("v1/scripts/exact_counts_out.txt", "a", encoding="utf-8") as out:
    out.write("\nOld processed aipsy files:\n")
    for f in sorted(glob.glob("v1/data/processed/aipsy/*.csv")):
        df = pd.read_csv(f)
        out.write(f"  {f}: {len(df)} rows\n")
