import pandas as pd

df_base = pd.read_csv("v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B_confirmatory_results.csv")
df_inst = pd.read_csv("v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B-Instruct_confirmatory_results.csv")

layers = [12, 16, 20, 24, 27]

print("| Layer | Base HIC (A) | Base HBC (A) | Inst HIC (A) | Inst HBC (A) |")
print("|---|---|---|---|---|")
for l in layers:
    b = df_base[(df_base["layer"]==l) & (df_base["dimension"]=="arousal")].iloc[0] if len(df_base[(df_base["layer"]==l) & (df_base["dimension"]=="arousal")]) > 0 else None
    i = df_inst[(df_inst["layer"]==l) & (df_inst["dimension"]=="arousal")].iloc[0] if len(df_inst[(df_inst["layer"]==l) & (df_inst["dimension"]=="arousal")]) > 0 else None
    if b is not None and i is not None:
        print(f"| {l} | {b['HIC_slope']:.3f} | {b['HBC_slope']:.3f} | {i['HIC_slope']:.3f} | {i['HBC_slope']:.3f} |")

