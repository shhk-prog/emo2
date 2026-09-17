import pandas as pd
df = pd.read_csv("v2/results/derived/phase6_path_patching/path_patching_results.csv")
agg_df = df.groupby("component")[["delta_Ev_patch", "delta_Ev_rand", "delta_WD_patch", "delta_WD_rand", "orig_prob_5", "patched_prob_5", "rand_prob_5"]].mean()
agg_df.to_csv("v2/results/derived/phase6_path_patching/path_patching_aggregated.csv")
print(agg_df.to_markdown())
