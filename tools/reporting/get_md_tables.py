import pandas as pd
import json
import numpy as np
from scipy.stats import wasserstein_distance

print("### Cross-decoding and Coupling Table")
df_conf_inst = pd.read_csv("v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B-Instruct_confirmatory_results.csv")
df_conf_base = pd.read_csv("v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B_confirmatory_results.csv")
# Print selected layers
layers = [12, 16, 20, 24, 27]
print("| Layer | Base HIC (Slope) | Base HBC (Slope) | Inst HIC (Slope) | Inst HBC (Slope) |")
print("|---|---|---|---|---|")
for l in layers:
    b = df_conf_base[(df_conf_base["layer"]==l) & (df_conf_base["dimension"]=="valence")].iloc[0] if len(df_conf_base[(df_conf_base["layer"]==l) & (df_conf_base["dimension"]=="valence")]) > 0 else None
    i = df_conf_inst[(df_conf_inst["layer"]==l) & (df_conf_inst["dimension"]=="valence")].iloc[0] if len(df_conf_inst[(df_conf_inst["layer"]==l) & (df_conf_inst["dimension"]=="valence")]) > 0 else None
    if b is not None and i is not None:
        print(f"| {l} | {b['HIC_slope']:.3f} | {b['HBC_slope']:.3f} | {i['HIC_slope']:.3f} | {i['HBC_slope']:.3f} |")

print("\n### Steering Table")
df_base_st = pd.DataFrame([json.loads(line) for line in open("v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B_steering_valence_results.jsonl")])
df_inst_st = pd.DataFrame([json.loads(line) for line in open("v2/results/derived/phase2_steering/Qwen_Qwen2.5-1.5B-Instruct_steering_valence_results.jsonl")])
print("| Layer | Base Slope (dEv/dAlpha) | Instruct Slope (dEv/dAlpha) |")
print("|---|---|---|")
for l in [20, 24, 27]:
    gb = df_base_st[df_base_st["intervention_layer"]==l]
    gi = df_inst_st[df_inst_st["intervention_layer"]==l]
    if len(gb)>0 and len(gi)>0:
        sb = np.polyfit(gb["intervention_alpha"], gb["E_v"], 1)[0]
        si = np.polyfit(gi["intervention_alpha"], gi["E_v"], 1)[0]
        print(f"| {l} | {sb:.3f} | {si:.3f} |")

print("\n### Patching Table")
df_patch = pd.DataFrame([json.loads(l) for l in open("v2/results/derived/phase4_circuit/patching_layer20_mlp.jsonl")])
df_inst = pd.DataFrame([json.loads(l) for l in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")])
df_base = pd.DataFrame([json.loads(l) for l in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_standard_results.jsonl")])
wd_inst = wasserstein_distance(df_inst["E_v"], df_base["E_v"])
wd_patch = wasserstein_distance(df_patch["E_v"], df_base["E_v"])
print("| Condition | E_v Mean | E_v Std | WD to Base |")
print("|---|---|---|---|")
print(f"| Base | {df_base['E_v'].mean():.2f} | {df_base['E_v'].std():.2f} | 0.000 |")
print(f"| Instruct | {df_inst['E_v'].mean():.2f} | {df_inst['E_v'].std():.2f} | {wd_inst:.3f} |")
print(f"| Instruct + Base L20 MLP | {df_patch['E_v'].mean():.2f} | {df_patch['E_v'].std():.2f} | {wd_patch:.3f} |")

