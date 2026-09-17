#!/usr/bin/env python3
"""
Summarize Cross-Family EmoBank Recognition and Self-Report Results.
Loads all *_summary.json files from the results directory and produces
a comparative Markdown table and CSV report.
"""

import os
import glob
import json
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="v1/results/recognition_baseline")
    parser.add_argument("--out-md", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()
    
    json_files = glob.glob(os.path.join(args.results_dir, "*_summary.json"))
    if not json_files:
        print(f"No summary JSON files found in {args.results_dir}")
        return
        
    records = []
    seen = set()
    
    # Prioritize specific dataset named files over generic old ones
    # Sort so that *_emobank_summary.json and *_aipsy_summary.json come first
    json_files = sorted(json_files, key=lambda x: (0 if ("_emobank_" in x or "_aipsy_" in x) else 1, x))
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            tag = data.get("tag", "unknown")
            dataset = data.get("dataset", "emobank")
            
            key = (tag, dataset)
            if key in seen:
                continue
            seen.add(key)
            records.append(data)
            
    df = pd.DataFrame(records)
    
    # Sort logically
    if "tag" in df.columns:
        df = df.sort_values(by=["dataset", "tag"]).reset_index(drop=True)
        
    out_csv = args.out_csv or os.path.join(args.results_dir, "cross_family_summary.csv")
    out_md = args.out_md or os.path.join(args.results_dir, "cross_family_summary.md")
    
    df.to_csv(out_csv, index=False)
    
    # Create clean Markdown tables for each dataset
    md_rows = ["# Cross-Family Recognition & Self-Report Evaluation Summary\n"]
    
    # 1. EmoBank Table
    df_emo = df[df["dataset"] == "emobank"]
    if not df_emo.empty:
        md_rows.append("## 1. EmoBank Benchmark (N=321)\n")
        md_rows.append("| Model Tag | Type | Recog $r_V$ (Cont) | Recog $r_A$ (Cont) | Recog $r_V$ (Greedy) | Self-Report $r_V$ (Cont) | Self-Report Exact (5,5)% | Self-Report $E[V]$ |")
        md_rows.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for _, r in df_emo.iterrows():
            m_type = "Instruct" if r.get("is_instruct") else "Base"
            r_v_c = f"{r.get('recognition_continuous_r_v', 0.0):.4f}"
            r_a_c = f"{r.get('recognition_continuous_r_a', 0.0):.4f}"
            r_v_g = f"{r.get('recognition_discrete_r_v', 0.0):.4f}"
            sr_v_c = f"{r.get('self_report_continuous_r_v', 0.0):.4f}"
            sr_55 = f"{r.get('self_report_exact_55_neutral_pct', 0.0):.2f}%"
            sr_ev = f"{r.get('self_report_mean_ev', 0.0):.3f}"
            tag = r.get("tag", "unknown")
            md_rows.append(f"| **{tag}** | {m_type} | **{r_v_c}** | {r_a_c} | {r_v_g} | **{sr_v_c}** | {sr_55} | {sr_ev} |")
        md_rows.append("\n")
        
    # 2. AIPsy-Affect Table
    df_aipsy = df[df["dataset"] == "aipsy"]
    if not df_aipsy.empty:
        md_rows.append("## 2. AIPsy-Affect Cohort (N=144)\n")
        md_rows.append("| Model Tag | Type | Recog $\|\\Delta_V\|$ (Aff-Neu) | Self-Report $\|\\Delta_V\|$ (Aff-Neu) | Self-Report Exact (5,5)% | Mean $p(5,5)$% |")
        md_rows.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for _, r in df_aipsy.iterrows():
            m_type = "Instruct" if r.get("is_instruct") else "Base"
            rec_diff = f"{r.get('recognition_abs_diff_aff_neu', 0.0):.3f}"
            sr_diff = f"{r.get('self_report_abs_diff_aff_neu', 0.0):.3f}"
            sr_55 = f"{r.get('self_report_exact_55_neutral_pct', 0.0):.2f}%"
            sr_p55 = f"{r.get('self_report_mean_p55_pct', 0.0):.2f}%"
            tag = r.get("tag", "unknown")
            md_rows.append(f"| **{tag}** | {m_type} | **{rec_diff}** | **{sr_diff}** | {sr_55} | {sr_p55} |")
        md_rows.append("\n")
        
    md_content = "\n".join(md_rows) + "\n"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("\n" + md_content)
    print(f"Summary table saved to:")
    print(f"  CSV: {out_csv}")
    print(f"  MD : {out_md}")

if __name__ == "__main__":
    main()
