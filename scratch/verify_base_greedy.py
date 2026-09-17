#!/usr/bin/env python3
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools

# Build 729 candidates
candidates = []
vad_triplets = []
for v, a, d in itertools.product(range(1, 10), range(1, 10), range(1, 10)):
    candidates.append(f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}')
    vad_triplets.append((v, a, d))

print("Total candidates:", len(candidates))

# Test on Qwen 2.5 1.5B Base
model_name = "Qwen/Qwen2.5-1.5B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Test with 2 contrasting stimuli
positive_text = "I won the championship and received the gold medal with immense joy!"
negative_text = "My closest friend passed away in a tragic car accident yesterday."

for label, text in [("Positive", positive_text), ("Negative", negative_text)]:
    prompt = f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\nRead the following text and report your affective state.\n\nText: {text}\n\nOutput:\n"
    
    # 1. Check free generation output
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=30, do_sample=False)
    gen_text = tokenizer.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 2. Check 729 candidates log-likelihoods
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cand_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    
    # Compute in sub-batches
    likelihoods = []
    for c_ids in cand_ids_list:
        seq = torch.tensor([prompt_ids + c_ids], device="cuda")
        with torch.no_grad():
            logits = model(seq).logits[:, :-1, :]
        p_len = len(prompt_ids)
        c_len = len(c_ids)
        sub_logits = logits[0, p_len - 1 : p_len - 1 + c_len, :]
        log_probs = torch.nn.functional.log_softmax(sub_logits, dim=-1)
        target = torch.tensor(c_ids, device="cuda").unsqueeze(1)
        ll = log_probs.gather(1, target).squeeze(1).sum().item()
        likelihoods.append(ll)
        
    probs = np.exp(np.array(likelihoods) - np.max(likelihoods))
    probs = probs / np.sum(probs)
    
    # Expected values
    ev = sum(p * t[0] for p, t in zip(probs, vad_triplets))
    ea = sum(p * t[1] for p, t in zip(probs, vad_triplets))
    ed = sum(p * t[2] for p, t in zip(probs, vad_triplets))
    
    # Argmax over 729 candidates
    best_idx = int(np.argmax(probs))
    best_vad = vad_triplets[best_idx]
    best_cand = candidates[best_idx]
    
    # Top 3 candidates
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_info = [(vad_triplets[i], probs[i]) for i in top3_idx]
    
    print(f"\n=== [{label}] ===")
    print(f"Generated text: {repr(gen_text)}")
    print(f"Expected VAD: E[V]={ev:.3f}, E[A]={ea:.3f}, E[D]={ed:.3f}")
    print(f"Argmax VAD among 729 candidates: {best_vad} (Prob: {probs[best_idx]:.4f})")
    print(f"Top 3 candidates: {top3_info}")
