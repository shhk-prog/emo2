import pandas as pd
import json
from scipy.stats import wasserstein_distance

df_base_st = pd.DataFrame([json.loads(l) for l in open("v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B_steering_valence_results.jsonl")])
df_inst_st = pd.DataFrame([json.loads(l) for l in open("v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B-Instruct_steering_valence_results.jsonl")])

import numpy as np
print("=== Steering Slope ===")
base_slopes = []
for l, g in df_base_st.groupby("intervention_layer"):
    slope = np.polyfit(g["intervention_alpha"], g["E_v"], 1)[0]
    print(f"Base Layer {l}: {slope:.3f}")
    
for l, g in df_inst_st.groupby("intervention_layer"):
    slope = np.polyfit(g["intervention_alpha"], g["E_v"], 1)[0]
    print(f"Instruct Layer {l}: {slope:.3f}")

