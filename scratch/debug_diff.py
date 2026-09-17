import pandas as pd
import numpy as np

# Check pair-level CSV we just saved
df_pair = pd.read_csv("v3/results/focused_causal_sweep_39pairs_pair_level.csv")
print("Saved pair-level mean:")
print("  L15 MLP Mean:", df_pair["l15_mlp_recovery"].mean())
print("  L15 MLP Med :", df_pair["l15_mlp_recovery"].median())
print("  L24 Resid Mean:", df_pair["l24_resid_recovery"].mean())
print("  L24 Resid Med :", df_pair["l24_resid_recovery"].median())

# Check focused_causal_sweep_39pairs.csv
df_focused = pd.read_csv("v3/results/focused_causal_sweep_39pairs.csv")
l15 = df_focused[(df_focused["layer"] == 15) & (df_focused["component"] == "mlp")]
l24 = df_focused[(df_focused["layer"] == 24) & (df_focused["component"] == "resid")]
print("\nFocused CSV:")
print("  L15 MLP Gen Mean:", l15["gen_mean_ot_recovery"].values[0])
print("  L15 MLP Gen Med :", l15["gen_median_ot_recovery"].values[0])
print("  L24 Resid Gen Mean:", l24["gen_mean_ot_recovery"].values[0])
print("  L24 Resid Gen Med :", l24["gen_median_ot_recovery"].values[0])
