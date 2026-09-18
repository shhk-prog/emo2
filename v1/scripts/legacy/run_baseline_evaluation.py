#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from affective_empathy_eval.evaluation import InterventionEvaluator
from affective_empathy_eval.metrics import calculate_euclidean_recovery

def get_prompt_text(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Logit Evaluation (Phase 1).")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-1.5B", help="HuggingFace ID for Base model.")
    parser.add_argument("--model-instruct", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace ID for Instruct model.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stimuli to process.")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Exclusion threshold for Recovery denominator.")
    parser.add_argument("--out-dir", type=str, default="results/derived/phase1", help="Output directory.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Data
    stimuli_path = "v1/data/processed/stimuli.csv"
    if not os.path.exists(stimuli_path):
        print(f"Error: {stimuli_path} not found. Run prepare_stimuli.py first.")
        return
        
    df_stim = pd.read_csv(stimuli_path)
    if args.limit:
        df_stim = df_stim.head(args.limit)
        
    # 2. Prompts
    import yaml
    with open("v1/configs/prompts.yaml") as f:
        prompts_config = yaml.safe_load(f)
        
    base_prompt_tmpl = get_prompt_text(prompts_config["prompts"]["empty_baseline"]["file"])
    post_prompt_tmpl = get_prompt_text(prompts_config["prompts"]["post_reported_va"]["file"])
    
    # 3. Model Loading & Evaluation loop
    results = []
    
    for model_id, model_type in [(args.model_base, "Base"), (args.model_instruct, "Instruct")]:
        print(f"\n--- Loading {model_id} ({model_type}) ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            evaluator = InterventionEvaluator(model, tokenizer)
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue
            
        print(f"Running evaluation for {len(df_stim)} stimuli...")
        for _, row in tqdm(df_stim.iterrows(), total=len(df_stim)):
            stim_id = row.get("stimulus_id", row.get("id"))
            stim_text = row["text"]
            human_v = row["V_scaled"]
            human_a = row["A_scaled"]
            
            # Format inputs. For Qwen Instruct, apply chat template. 
            # For Base, just append text.
            if model_type == "Instruct":
                # Baseline
                base_messages = [
                    {"role": "system", "content": "You are a measurement instrument."},
                    {"role": "user", "content": base_prompt_tmpl}
                ]
                base_text = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
                
                # Post
                post_messages = [
                    {"role": "system", "content": "You are a measurement instrument."},
                    {"role": "user", "content": post_prompt_tmpl.replace("{stimulus}", str(stim_text))}
                ]
                post_text = tokenizer.apply_chat_template(post_messages, tokenize=False, add_generation_prompt=True)
            else:
                base_text = base_prompt_tmpl
                post_text = post_prompt_tmpl.replace("{stimulus}", str(stim_text))
            
            # Evaluate Baseline
            base_inputs = tokenizer(base_text, return_tensors="pt").to(model.device)
            base_res = evaluator.evaluate_sequence_logits(base_inputs.input_ids, base_inputs.attention_mask)
            
            # Evaluate Post
            post_inputs = tokenizer(post_text, return_tensors="pt").to(model.device)
            post_res = evaluator.evaluate_sequence_logits(post_inputs.input_ids, post_inputs.attention_mask)
            
            # Calculate Recovery (comparing post to human anchor here as proxy, or skip patch calculation for now)
            # In Phase 1 we just want to calculate baseline reaction and exclude unreactive pairs.
            # Denominator for recovery is ||E_target - E_source||_2 (i.e. ||E_post - E_base||_2)
            dist_post_base = float(np.sqrt((post_res["E_V"] - base_res["E_V"])**2 + (post_res["E_A"] - base_res["E_A"])**2))
            excluded = dist_post_base < args.threshold
            
            results.append({
                "model_id": model_id,
                "model_type": model_type,
                "stimulus_id": stim_id,
                "human_V_scaled": human_v,
                "human_A_scaled": human_a,
                "base_E_V": base_res["E_V"],
                "base_E_A": base_res["E_A"],
                "post_E_V": post_res["E_V"],
                "post_E_A": post_res["E_A"],
                "dist_post_base": dist_post_base,
                "excluded": excluded
            })
            
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    df_res = pd.DataFrame(results)
    out_path = os.path.join(args.out_dir, "baseline_evaluation.csv")
    df_res.to_csv(out_path, index=False)
    
    print(f"\nEvaluation complete. Results saved to {out_path}")
    
    # Print exclusion summary
    if not df_res.empty:
        summary = df_res.groupby("model_type")["excluded"].agg(["count", "sum", "mean"])
        summary.columns = ["total_pairs", "excluded_pairs", "exclusion_rate"]
        print("\nExclusion Summary (Distance < threshold):")
        print(summary)

if __name__ == "__main__":
    main()
