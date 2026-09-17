import os
import glob
import re
import pandas as pd

log_dir = "logs/scaling"
out_csv = "results/derived/scaling_summary.csv"

summary_data = []

def parse_params(tag):
    if "0.5b" in tag: return 0.5
    if "1.5b" in tag: return 1.5
    if "1b" in tag: return 1.0
    if "3b" in tag: return 3.0
    if "7b" in tag: return 7.0
    return None

def parse_family(tag):
    if "qwen" in tag: return "Qwen2.5"
    if "llama" in tag: return "Llama-3.2"
    return "Other"

for log_file in glob.glob(os.path.join(log_dir, "*.log")):
    tag = os.path.basename(log_file).replace(".log", "")
    with open(log_file, "r") as f:
        content = f.read()
    
    base_shift = re.search(r'Base Clean Shift \(Delta E\[V\]\):\s*([\d\.\-]+)', content)
    instruct_shift = re.search(r'Instruct Clean Shift \(Delta E\[V\]\):\s*([\d\.\-]+)', content)
    suppression = re.search(r'Behavioral Suppression Ratio:\s*([\d\.\-]+)%', content)
    
    base_val = float(base_shift.group(1)) if base_shift else None
    instruct_val = float(instruct_shift.group(1)) if instruct_shift else None
    supp_val = float(suppression.group(1)) if suppression else None
    
    params = parse_params(tag)
    family = parse_family(tag)
    
    summary_data.append({
        "tag": tag,
        "family": family,
        "parameters_b": params,
        "base_clean_shift_v": base_val,
        "instruct_clean_shift_v": instruct_val,
        "behavioral_suppression_ratio": supp_val
    })

df = pd.DataFrame(summary_data)
df = df.sort_values(by=["family", "parameters_b"])

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"Summary written to {out_csv}")
print(df)
