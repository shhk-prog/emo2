#!/usr/bin/env python3
"""
Complete V-A metrics calculation for all 8 models on EmoBank and AIPsy-Affect.
Extracts both Valence and Arousal for Recognition and Self-Report across
Continuous (likelihood expectation) and Discrete (Greedy) modes.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = "v1/results/recognition_baseline"

def analyze_emobank():
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_emobank.csv")))
    records = []
    
    for f in csv_files:
        tag = os.path.basename(f).replace("_emobank.csv", "")
        df = pd.read_csv(f)
        
        # Human columns: human_v, human_a
        # Recognition: rec_ev, rec_ea, rec_gv, rec_ga
        # Self-Report: post_ev, post_ea, post_gv, post_ga
        
        # Continuous Recognition r
        r_rec_v_cont, _ = pearsonr(df["human_v"], df["rec_ev"])
        r_rec_a_cont, _ = pearsonr(df["human_a"], df["rec_ea"])
        
        # Greedy Recognition r
        r_rec_v_greedy, _ = pearsonr(df["human_v"], df["rec_gv"])
        r_rec_a_greedy, _ = pearsonr(df["human_a"], df["rec_ga"])
        
        # Continuous Self-Report r
        r_sr_v_cont, _ = pearsonr(df["human_v"], df["post_ev"])
        r_sr_a_cont, _ = pearsonr(df["human_a"], df["post_ea"])
        
        # Greedy Self-Report r
        r_sr_v_greedy, _ = pearsonr(df["human_v"], df["post_gv"])
        r_sr_a_greedy, _ = pearsonr(df["human_a"], df["post_ga"])
        
        # Mean Self-Report E[V], E[A]
        mean_post_ev = df["post_ev"].mean()
        mean_post_ea = df["post_ea"].mean()
        
        # Exact (5, 5) Neutral Rate
        exact_55 = ((df["post_gv"] == 5) & (df["post_ga"] == 5)).mean() * 100.0
        
        is_instruct = "instruct" in tag.lower()
        records.append({
            "tag": tag,
            "type": "Instruct" if is_instruct else "Base",
            "rec_r_v_cont": r_rec_v_cont,
            "rec_r_a_cont": r_rec_a_cont,
            "rec_r_v_greedy": r_rec_v_greedy,
            "rec_r_a_greedy": r_rec_a_greedy,
            "sr_r_v_cont": r_sr_v_cont,
            "sr_r_a_cont": r_sr_a_cont,
            "sr_r_v_greedy": r_sr_v_greedy,
            "sr_r_a_greedy": r_sr_a_greedy,
            "sr_mean_ev": mean_post_ev,
            "sr_mean_ea": mean_post_ea,
            "exact_55": exact_55
        })
        
    return pd.DataFrame(records)

def analyze_aipsy():
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_aipsy.csv")))
    records = []
    
    for f in csv_files:
        tag = os.path.basename(f).replace("_aipsy.csv", "")
        df = pd.read_csv(f)
        
        aff_df = df[df["condition"] == "affective"]
        neu_df = df[df["condition"] == "neutral"]
        
        # Recognition diff
        rec_diff_v = abs(aff_df["rec_ev"].mean() - neu_df["rec_ev"].mean())
        rec_diff_a = abs(aff_df["rec_ea"].mean() - neu_df["rec_ea"].mean())
        
        # Self-Report diff
        sr_diff_v = abs(aff_df["post_ev"].mean() - neu_df["post_ev"].mean())
        sr_diff_a = abs(aff_df["post_ea"].mean() - neu_df["post_ea"].mean())
        
        # Self-Report values
        sr_ev_aff = aff_df["post_ev"].mean()
        sr_ea_aff = aff_df["post_ea"].mean()
        sr_ev_neu = neu_df["post_ev"].mean()
        sr_ea_neu = neu_df["post_ea"].mean()
        
        # Exact (5, 5)
        exact_55 = ((df["post_gv"] == 5) & (df["post_ga"] == 5)).mean() * 100.0
        
        is_instruct = "instruct" in tag.lower()
        records.append({
            "tag": tag,
            "type": "Instruct" if is_instruct else "Base",
            "rec_diff_v": rec_diff_v,
            "rec_diff_a": rec_diff_a,
            "sr_diff_v": sr_diff_v,
            "sr_diff_a": sr_diff_a,
            "sr_ev_aff": sr_ev_aff,
            "sr_ea_aff": sr_ea_aff,
            "sr_ev_neu": sr_ev_neu,
            "sr_ea_neu": sr_ea_neu,
            "exact_55": exact_55
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    df_emo = analyze_emobank()
    df_aipsy = analyze_aipsy()
    
    print("=== EMOBANK COMPLETE V-A ===")
    print(df_emo.to_string())
    print("\n=== AIPSY COMPLETE V-A ===")
    print(df_aipsy.to_string())
    
    df_emo.to_csv(os.path.join(RESULTS_DIR, "emobank_va_complete.csv"), index=False)
    df_aipsy.to_csv(os.path.join(RESULTS_DIR, "aipsy_va_complete.csv"), index=False)
