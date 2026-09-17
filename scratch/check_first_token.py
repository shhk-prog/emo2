import sys
import json
from pathlib import Path
from transformers import AutoTokenizer

print("Starting token check...", flush=True)
model_path = "/mnt/nas/home/hiromi/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
print(f"Loading tokenizer from {model_path}...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print("Tokenizer loaded successfully!", flush=True)

candidates = [
    f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}'
    for v in range(1, 10)
    for a in range(1, 10)
    for d in range(1, 10)
]
print(f"Generated {len(candidates)} candidates.", flush=True)

first_ids = [
    tok.encode(c, add_special_tokens=False)[0]
    for c in candidates
]

unique_ids = sorted(set(first_ids))
print(f"unique first-token IDs: {unique_ids}", flush=True)
print(f"number of unique first tokens: {len(unique_ids)}", flush=True)
for x in unique_ids:
    print(f"Token ID {x}: {repr(tok.decode([x]))}", flush=True)

result_data = {
    "unique_first_token_ids": unique_ids,
    "num_unique_first_tokens": len(unique_ids),
    "decoded": {str(x): repr(tok.decode([x])) for x in unique_ids}
}
with open("/mnt/nas/home/hiromi/src/emo/scratch/first_token_result.json", "w") as f:
    json.dump(result_data, f, indent=2)
print("Saved result to scratch/first_token_result.json", flush=True)
