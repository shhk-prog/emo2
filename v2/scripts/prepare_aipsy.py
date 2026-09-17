#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def main():
    print("Loading AIPsy-Affect stimuli.csv...")
    df = pd.read_csv("v1/data/processed/stimuli.csv")
    print(f"Loaded {len(df)} rows.")
    
    print("Columns:", df.columns.tolist())
    if 'intensity' in df.columns:
        print("Intensity values:", df['intensity'].unique())
    print("Condition values:", df['condition'].unique())
    
    # 各 pair_id が持つ intensity のセットを計算
    intensity_sets = df.groupby('pair_id')['intensity'].apply(set)
    print("Example intensity sets:")
    print(intensity_sets.head())
    
    # strict matched subset の抽出 (neutral, moderate, peak がすべて揃っているペア)
    # ただし 'neutral' ではなく 'low' など別名の可能性があるため、セットの中身を見て調整
    # neutral は intensity が NaN または 'neutral' かもしれない
    
    # 状態の出力
    print("\nValue counts for intensity:")
    print(df['intensity'].value_counts(dropna=False))
    
    print("\nCondition vs Intensity:")
    print(pd.crosstab(df['condition'], df['intensity'].fillna('NaN')))

if __name__ == "__main__":
    main()
