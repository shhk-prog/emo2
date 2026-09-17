import pandas as pd
import numpy as np
import os

def main():
    results_path = "v2/results/derived/phase8_causal_scrubbing/strict_causal_scrubbing_results.csv"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    df = pd.read_csv(results_path)
    # Extract pair_id (e.g., 'B-grief-d1-v1' -> 'grief-d1-v1')
    df['pair_id'] = df['id'].apply(lambda x: x[2:] if x.startswith("B-") or x.startswith("I-") else x)

    components = ['attn_14', 'mlp_10', 'mlp_15']
    n_bootstraps = 10000
    np.random.seed(42)

    def format_ci(arr):
        return f"[{np.percentile(arr, 2.5):.4f}, {np.percentile(arr, 97.5):.4f}]"

    def calc_metrics(resampled_df):
        mean_delta_ev_match = resampled_df['delta_Ev_l27'].mean()
        mean_delta_wd_match = resampled_df['delta_WD_l27'].mean()
        mean_delta_ev_rand = resampled_df['delta_Ev_l27_random'].mean()
        mean_delta_wd_rand = resampled_df['delta_WD_l27_random'].mean()
        diff_ev = mean_delta_ev_match - mean_delta_ev_rand
        diff_wd = mean_delta_wd_match - mean_delta_wd_rand
        return mean_delta_ev_match, mean_delta_wd_match, mean_delta_ev_rand, mean_delta_wd_rand, diff_ev, diff_wd

    markdown_rows = []

    for comp in components:
        comp_df = df[df['component'] == comp].copy()
        if len(comp_df) == 0:
            continue
        
        pair_ids = comp_df['pair_id'].unique()
        
        boot_metrics = []
        for i in range(n_bootstraps):
            resampled_pairs = np.random.choice(pair_ids, size=len(pair_ids), replace=True)
            resampled_indices = []
            for pid in resampled_pairs:
                resampled_indices.extend(comp_df[comp_df['pair_id'] == pid].index)
            resampled_df = comp_df.loc[resampled_indices]
            boot_metrics.append(calc_metrics(resampled_df))
            
        boot_metrics = np.array(boot_metrics)
        
        # Means from the actual dataset (not the bootstrap mean)
        mean_delta_ev_match = comp_df['delta_Ev_l27'].mean()
        mean_delta_wd_match = comp_df['delta_WD_l27'].mean()
        mean_delta_ev_rand = comp_df['delta_Ev_l27_random'].mean()
        mean_delta_wd_rand = comp_df['delta_WD_l27_random'].mean()
        diff_ev = mean_delta_ev_match - mean_delta_ev_rand
        diff_wd = mean_delta_wd_match - mean_delta_wd_rand

        print(f"--- Component: {comp} ---")
        print(f"Matched Delta E[V]: {mean_delta_ev_match:.4f} CI: {format_ci(boot_metrics[:,0])}")
        print(f"Matched Delta WD:   {mean_delta_wd_match:.4f} CI: {format_ci(boot_metrics[:,1])}")
        print(f"Random Delta E[V]:  {mean_delta_ev_rand:.4f} CI: {format_ci(boot_metrics[:,2])}")
        print(f"Random Delta WD:    {mean_delta_wd_rand:.4f} CI: {format_ci(boot_metrics[:,3])}")
        print(f"Diff (M-R) Delta E[V]: {diff_ev:.4f} CI: {format_ci(boot_metrics[:,4])}")
        print(f"Diff (M-R) Delta WD:   {diff_wd:.4f} CI: {format_ci(boot_metrics[:,5])}")
        print()

        # Format markdown for Table 4
        markdown_rows.append(f"| `{comp}` matched Base | {mean_delta_ev_match:.4f} | {format_ci(boot_metrics[:,0])} | {mean_delta_wd_match:.4f} | {format_ci(boot_metrics[:,1])} | {diff_ev:.4f} {format_ci(boot_metrics[:,4])} |")
        markdown_rows.append(f"| `{comp}` random source | {mean_delta_ev_rand:.4f} | {format_ci(boot_metrics[:,2])} | {mean_delta_wd_rand:.4f} | {format_ci(boot_metrics[:,3])} | — |")

    print("\n--- Markdown rows for Table 4 ---")
    for row in markdown_rows:
        print(row)

if __name__ == "__main__":
    main()
