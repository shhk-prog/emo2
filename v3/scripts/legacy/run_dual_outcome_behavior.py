import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

def get_behavioral_log_likelihood_ratio(
    model, 
    tokenizer, 
    context: str, 
    y_support: str, 
    y_neutral: str
) -> float:
    """
    Calculate B(x) = log p(y_support | x) - log p(y_neutral | x)
    """
    def _get_seq_logprob(target_seq):
        inputs = tokenizer(context + target_seq, return_tensors="pt").to(model.device)
        context_len = tokenizer(context, return_tensors="pt").input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, context_len-1:-1, :]
            target_ids = inputs.input_ids[0, context_len:]
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            seq_logprob = log_probs[range(len(target_ids)), target_ids].sum().item()
        return seq_logprob
    
    logp_support = _get_seq_logprob(y_support)
    logp_neutral = _get_seq_logprob(y_neutral)
    
    return logp_support - logp_neutral

def get_patch_hook(base_val_tensor, target_pos):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[0, target_pos, :] = base_val_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                output[0, target_pos, :] = base_val_tensor.to(output.dtype)
            return output
    return hook

def run_dual_outcome_experiment(args):
    """
    Main execution loop for dual outcome measurement:
    1. Base state E[V] and B(x)
    2. Intervened state E[V]_I and B(x)_I
    """
    print(f"Loading {args.model_name_or_path} for Dual Outcome Experiment...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    
    df = pd.read_csv(args.dataset_path)
    candidates, va_pairs = generate_81_candidates()
    
    y_support = " I am sorry to hear that. How can I support you?"
    y_neutral = " Acknowledged. Let me know if you need administrative help."
    
    results = []
    
    # Needs a baseline state dict loaded for patching
    base_states = {}
    if os.path.exists(args.base_states):
        base_states = dict(np.load(args.base_states, allow_pickle=True))
        for k, v in base_states.items():
            if v.shape == (): base_states[k] = v.item()

    out_file = args.output_path
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(out_file, 'w') as f:
        for idx, row in df.iterrows():
            uid = row['id']
            context = row['neutral_text'] if 'neutral_text' in row else row['text']
            
            prompt = f"Read the following text and report your affective state.\n\nText: {context}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            target_pos = prompt_ids.shape[1] - 1
            
            # 1. Baseline
            l_base, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
            e_v_base, e_a_base, _, _, _ = compute_expected_va(l_base, va_pairs)
            b_x_base = get_behavioral_log_likelihood_ratio(model, tokenizer, context, y_support, y_neutral)
            
            # 2. Patching (Intervention)
            e_v_int, b_x_int = None, None
            key = f"{uid}_layer{args.patch_layer}_{args.patch_comp}"
            if key in base_states:
                base_val = torch.tensor(base_states[key]).to(model.device)
                if args.patch_comp == "mlp":
                    target_module = model.model.layers[args.patch_layer].mlp
                else:
                    target_module = model.model.layers[args.patch_layer].self_attn
                
                h_hook = target_module.register_forward_hook(get_patch_hook(base_val, target_pos))
                
                l_int, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
                e_v_int, e_a_int, _, _, _ = compute_expected_va(l_int, va_pairs)
                b_x_int = get_behavioral_log_likelihood_ratio(model, tokenizer, context, y_support, y_neutral)
                
                h_hook.remove()
                
            res = {
                "id": uid,
                "e_v_base": e_v_base,
                "b_x_base": b_x_base,
                "e_v_int": e_v_int,
                "b_x_int": b_x_int
            }
            f.write(json.dumps(res) + "\n")
            results.append(res)
            
    print(f"Finished extracting Dual Outcomes. Saved to {out_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="v2/data/processed/aipsy_annotated/test_strict.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/dual_outcome_results.csv")
    parser.add_argument("--base_states", type=str, default="v2/results/derived/phase4_circuit/Qwen_Qwen2.5-1.5B_module_states.npz")
    parser.add_argument("--patch_layer", type=int, default=15)
    parser.add_argument("--patch_comp", type=str, default="mlp")
    args = parser.parse_args()
    
    run_dual_outcome_experiment(args)
