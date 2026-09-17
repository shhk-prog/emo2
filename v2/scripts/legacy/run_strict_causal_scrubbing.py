#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def get_capture_hook(cache, name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            cache[name] = output[0][:, -1, :].detach().clone()
        else:
            cache[name] = output[:, -1, :].detach().clone()
    return hook

def get_patch_hook(cache, name):
    def hook(module, inputs):
        # inputs is a tuple, usually (hidden_states,)
        # We modify the last token's hidden state before it enters the module
        hidden_states = inputs[0].clone()
        # The cache contains the target patched hidden state for the last token
        hidden_states[:, -1, :] = cache[name]
        return (hidden_states,) + inputs[1:]
import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
import random

# Fix random seed for reproducibility of random source patching
random.seed(42)
np.random.seed(42)

def get_capture_hook(cache, name):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            cache[name] = outputs[0][:, -1, :].detach().clone()
        else:
            cache[name] = outputs[:, -1, :].detach().clone()
    return hook

def get_patch_hook(cache, name):
    def hook(module, inputs):
        hidden_states = inputs[0].clone()
        # The cache contains the target patched hidden state for the last token
        hidden_states[:, -1, :] = cache[name]
        return (hidden_states,) + inputs[1:]
    return hook

def apply_chat_template_val_forcing(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += '{\n  "valence": '
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase8_causal_scrubbing")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
        
    print(f"Loading Instruct model: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    inst_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")

    print(f"Loading Base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
    
    numeric_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    
    components_to_test = [("mlp", 10), ("mlp", 15), ("attn", 14)]
    final_layer_idx = len(inst_model.model.layers) - 1 # 27
    
    summary_results = []
    
    for comp_type, layer in components_to_test:
        print(f"--- True Causal Scrubbing Layer {layer} {comp_type} ---")
        
        base_cache = {}
        inst_cache = {}
        patch_cache = {}
        
        if comp_type == "attn":
            base_module = base_model.model.layers[layer].self_attn
            inst_module = inst_model.model.layers[layer].self_attn
        elif comp_type == "mlp":
            base_module = base_model.model.layers[layer].mlp
            inst_module = inst_model.model.layers[layer].mlp
            
        base_handle = base_module.register_forward_hook(get_capture_hook(base_cache, "comp"))
        inst_handle = inst_module.register_forward_hook(get_capture_hook(inst_cache, "comp"))
        
        # Collect base components
        all_base_comps = []
        for idx, row in test_df.iterrows():
            prompt = apply_chat_template_val_forcing(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                base_model(prompt_ids)
            all_base_comps.append(base_cache["comp"])
            
        all_base_comps = torch.stack(all_base_comps)
        
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            prompt = apply_chat_template_val_forcing(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(inst_model.device)
            
            # 1. Clean Run on Instruct
            clean_h26_cache = {}
            clean_h26_handle = inst_model.model.layers[final_layer_idx].register_forward_pre_hook(
                lambda m, inputs: clean_h26_cache.update({"h26": inputs[0][:, -1, :].detach().clone()}) or inputs
            )
            
            with torch.no_grad():
                inst_outputs = inst_model(prompt_ids, output_hidden_states=True)
                
            clean_h26_handle.remove()
            
            orig_logits = inst_outputs.logits[0, -1, numeric_tokens]
            orig_probs = F.softmax(orig_logits, dim=-1)
            orig_Ev = torch.sum(orig_probs * torch.arange(1, 10, device=orig_probs.device)).item()
            
            inst_c = inst_cache["comp"]
            base_c = all_base_comps[i].to(inst_model.device)
            h26_clean = clean_h26_cache["h26"]
            
            # 2. Path Patching to Unembedding (Direct linear replacement)
            # Final residual stream is hidden_states[-1]
            patched_h27 = inst_outputs.hidden_states[-1][0, -1, :].unsqueeze(0) - inst_c + base_c
            patched_norm = inst_model.model.norm(patched_h27)
            patched_logits_full = inst_model.lm_head(patched_norm)
            patched_logits_unembed = patched_logits_full[0, numeric_tokens]
            probs_unembed = F.softmax(patched_logits_unembed, dim=-1)
            Ev_unembed = torch.sum(probs_unembed * torch.arange(1, 10, device=probs_unembed.device)).item()
            
            # 3. Path Patching to Layer 27 (True Causal Scrubbing - Matched)
            patched_h26 = h26_clean - inst_c + base_c
            patch_cache["h26"] = patched_h26
            
            inst_h26_handle = inst_model.model.layers[final_layer_idx].register_forward_pre_hook(get_patch_hook(patch_cache, "h26"))
            with torch.no_grad():
                patched_outputs = inst_model(prompt_ids)
            inst_h26_handle.remove()
            
            patched_logits_l27 = patched_outputs.logits[0, -1, numeric_tokens]
            probs_l27 = F.softmax(patched_logits_l27, dim=-1)
            Ev_l27 = torch.sum(probs_l27 * torch.arange(1, 10, device=probs_l27.device)).item()

            # 3.5 Path Patching to Layer 27 (Random Source)
            random_i = random.randint(0, len(test_df) - 1)
            while random_i == i and len(test_df) > 1:
                random_i = random.randint(0, len(test_df) - 1)
            base_c_random = all_base_comps[random_i].to(inst_model.device)
            
            patched_h26_random = h26_clean - inst_c + base_c_random
            patch_cache["h26"] = patched_h26_random
            
            inst_h26_handle_random = inst_model.model.layers[final_layer_idx].register_forward_pre_hook(get_patch_hook(patch_cache, "h26"))
            with torch.no_grad():
                patched_outputs_random = inst_model(prompt_ids)
            inst_h26_handle_random.remove()
            
            patched_logits_l27_random = patched_outputs_random.logits[0, -1, numeric_tokens]
            probs_l27_random = F.softmax(patched_logits_l27_random, dim=-1)
            Ev_l27_random = torch.sum(probs_l27_random * torch.arange(1, 10, device=probs_l27_random.device)).item()
            
            # 4. Base Baseline
            with torch.no_grad():
                base_outputs = base_model(prompt_ids)
            base_logits = base_outputs.logits[0, -1, numeric_tokens]
            base_probs = F.softmax(base_logits, dim=-1)
            
            # WD Metrics
            wd_inst = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=orig_probs.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            wd_unembed = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=probs_unembed.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            wd_l27 = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=probs_l27.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            wd_l27_random = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=probs_l27_random.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            
            summary_results.append({
                "id": row['id'],
                "component": f"{comp_type}_{layer}",
                "orig_Ev": orig_Ev,
                "Ev_unembed": Ev_unembed,
                "Ev_l27": Ev_l27,
                "Ev_l27_random": Ev_l27_random,
                "wd_inst": wd_inst,
                "wd_unembed": wd_unembed,
                "wd_l27": wd_l27,
                "wd_l27_random": wd_l27_random,
                "delta_Ev_unembed": Ev_unembed - orig_Ev,
                "delta_Ev_l27": Ev_l27 - orig_Ev,
                "delta_Ev_l27_random": Ev_l27_random - orig_Ev,
                "delta_WD_unembed": wd_unembed - wd_inst,
                "delta_WD_l27": wd_l27 - wd_inst,
                "delta_WD_l27_random": wd_l27_random - wd_inst
            })
            
        base_handle.remove()
        inst_handle.remove()
        
    summary_df = pd.DataFrame(summary_results)
    out_csv = os.path.join(args.out_dir, "strict_causal_scrubbing_results.csv")
    summary_df.to_csv(out_csv, index=False)
    
    agg_df = summary_df.groupby("component").mean(numeric_only=True)[["delta_Ev_unembed", "delta_Ev_l27", "delta_WD_unembed", "delta_WD_l27"]]
    print("\n=== True Causal Scrubbing Summary ===")
    print(agg_df.to_markdown())
    agg_csv = os.path.join(args.out_dir, "strict_causal_scrubbing_aggregated.csv")
    agg_df.to_csv(agg_csv)

if __name__ == "__main__":
    main()
