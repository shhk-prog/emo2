import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from affective_empathy_eval.extraction import ManifestManager
from affective_empathy_eval.evaluation import InterventionEvaluator
from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.metrics import calculate_euclidean_recovery

def load_data(split: str) -> pd.DataFrame:
    path = f"v1/data/processed/aipsy/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_aipsy.py first.")
    return pd.read_csv(path)

def load_tensors(manifest_df: pd.DataFrame, position: str = "stimulus_mean_pool"):
    layer_tensors = {}
    for _, row in manifest_df.iterrows():
        if row["extraction_position"] != position:
            continue
        layer_idx = row["layer"]
        if layer_idx not in layer_tensors:
            layer_tensors[layer_idx] = {}
        # We need the full absolute path or relative to project root
        tensor_path = row["tensor_path"]
        layer_tensors[layer_idx][f"{row['stimulus_id']}_{row['condition']}"] = np.load(tensor_path)
    return layer_tensors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--position", type=str, default="stimulus_mean_pool")
    parser.add_argument("--tensors-dir", type=str, default="results/derived/phase2/train_tensors")
    parser.add_argument("--out-dir", type=str, default="results/derived/phase3")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=384)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading {args.split} dataset...")
    df = load_data(args.split)
    
    print(f"Loading tensors from {args.tensors_dir}...")
    manifest_manager = ManifestManager(os.path.join(args.tensors_dir, "manifest.jsonl"))
    manifest_df = manifest_manager.load_all_as_dataframe()
    layer_tensors = load_tensors(manifest_df, args.position)
    
    # We expect 28 layers for Qwen2.5-1.5B
    num_layers = max(layer_tensors.keys()) + 1
    
    # Prepare the stimuli
    prompt_path = "prompts/affective_reception_v1.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        prompt_template = "{stimulus}"

    # Separate into Neutral and Affective
    neutral_df = df[df["condition"] == "neutral"].sort_values("id").reset_index(drop=True)
    affective_df = df[df["condition"] == "affective"].sort_values("id").reset_index(drop=True)
    
    # Align pairs using id and matched_control_id
    merged = pd.merge(affective_df, neutral_df, left_on="matched_control_id", right_on="id", suffixes=('_affective', '_neutral'))
    n_pairs = len(merged)
    print(f"Loaded {n_pairs} paired stimuli.")
    
    # Convert prompts
    neutral_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_neutral"]]
    affective_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_affective"]]
    
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    evaluator = InterventionEvaluator(model, tokenizer)
    
    def run_eval(texts):
        all_results = []
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i+args.batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
            all_results.extend(res)
        return all_results

    # 1. Clean Runs (Baselines)
    print("Running Clean (No Intervention) evaluations...")
    neutral_results = run_eval(neutral_texts)
    affective_results = run_eval(affective_texts)
    
    # 2. Collect Tensors for interventions
    print("Preparing intervention tensors...")
    affective_tensors_by_layer = []
    neutral_tensors_by_layer = []
    for l in range(num_layers):
        aff_t = []
        neu_t = []
        for idx, row in merged.iterrows():
            id_aff = f"{row['id_affective']}_affective"
            id_neu = f"{row['id_neutral']}_neutral"
            aff_t.append(layer_tensors[l][id_aff])
            neu_t.append(layer_tensors[l][id_neu])
        affective_tensors_by_layer.append(np.stack(aff_t)) # (n_pairs, hidden)
        neutral_tensors_by_layer.append(np.stack(neu_t))   # (n_pairs, hidden)
        
    patching_results = []
    
    print(f"Starting Activation Patching across {num_layers} layers...")
    for l in tqdm(range(num_layers), desc="Layers"):
        src_tensors = affective_tensors_by_layer[l]
        
        patched_res = []
        for i in range(0, n_pairs, args.batch_size):
            batch_texts = neutral_texts[i:i+args.batch_size]
            batch_src = src_tensors[i:i+args.batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            prompt_last_idx = inputs.input_ids.shape[1] - 1
            
            with PyTorchActivationPatcher(model, l, batch_src, patch_weight=1.0, position=prompt_last_idx, intervention_type="patch"):
                res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                patched_res.extend(res)
                
        recoveries_V = []
        recoveries_A = []
        for idx in range(n_pairs):
            cln_v = affective_results[idx]["E_V"]
            crp_v = neutral_results[idx]["E_V"]
            ptc_v = patched_res[idx]["E_V"]
            # calculate_recovery: (patched - corrupted) / (clean - corrupted)
            # Actually, I imported calculate_euclidean_recovery from metrics. Let me use a simple scalar recovery since these are scalar E_V and E_A values.
            # (patched - corrupted) / (clean - corrupted)
            if abs(cln_v - crp_v) > 1e-5:
                rec_v = (ptc_v - crp_v) / (cln_v - crp_v)
            else:
                rec_v = np.nan
            recoveries_V.append(rec_v)
            
            cln_a = affective_results[idx]["E_A"]
            crp_a = neutral_results[idx]["E_A"]
            ptc_a = patched_res[idx]["E_A"]
            if abs(cln_a - crp_a) > 1e-5:
                rec_a = (ptc_a - crp_a) / (cln_a - crp_a)
            else:
                rec_a = np.nan
            recoveries_A.append(rec_a)
            
        mean_rec_v = np.nanmean(recoveries_V)
        mean_rec_a = np.nanmean(recoveries_A)
        patching_results.append({"layer": l, "recovery_V": mean_rec_v, "recovery_A": mean_rec_a})
        
        torch.cuda.empty_cache()
        
    patching_df = pd.DataFrame(patching_results)
    patching_df.to_csv(os.path.join(args.out_dir, "activation_patching_results.csv"), index=False)
    print("Activation Patching Results:")
    print(patching_df)
    
    print("\nStarting Ablation across layers...")
    ablation_results = []
    for l in tqdm(range(num_layers), desc="Layers"):
        # Source is the mean of all neutral tensors at this layer
        mean_neu_tensor = np.mean(neutral_tensors_by_layer[l], axis=0)
        
        ablated_res = []
        for i in range(0, n_pairs, args.batch_size):
            batch_texts = affective_texts[i:i+args.batch_size]
            # Since source is a single vector, PyTorchActivationPatcher will expand it
            batch_src = mean_neu_tensor
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            prompt_last_idx = inputs.input_ids.shape[1] - 1
            
            with PyTorchActivationPatcher(model, l, batch_src, patch_weight=1.0, position=prompt_last_idx, intervention_type="replace"):
                res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                ablated_res.extend(res)
                
        # Recovery here is inverse: target is affective, we intervene to make it neutral.
        # Clean: affective. Corrupted: neutral. Patched: ablated (should move towards neutral).
        # We want to measure the drop from affective to neutral.
        # Let's compute proportion of effect removed: (clean - ablated) / (clean - corrupted)
        drops_V = []
        for idx in range(n_pairs):
            cln_v = affective_results[idx]["E_V"]
            crp_v = neutral_results[idx]["E_V"]
            abl_v = ablated_res[idx]["E_V"]
            if abs(cln_v - crp_v) > 1e-5:
                drop = (cln_v - abl_v) / (cln_v - crp_v)
                drops_V.append(drop)
            
        ablation_results.append({"layer": l, "effect_removed_V": np.nanmean(drops_V)})
        torch.cuda.empty_cache()
        
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(args.out_dir, "ablation_results.csv"), index=False)
    print("Ablation Results:")
    print(ablation_df)

    print("\nStarting Steering across layers...")
    steering_results = []
    alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    for l in tqdm(range(num_layers), desc="Layers"):
        # Direction vector d = mu_affective - mu_neutral
        mean_aff = np.mean(affective_tensors_by_layer[l], axis=0)
        mean_neu = np.mean(neutral_tensors_by_layer[l], axis=0)
        direction = mean_aff - mean_neu
        
        for alpha in alphas:
            steered_res = []
            for i in range(0, n_pairs, args.batch_size):
                batch_texts = neutral_texts[i:i+args.batch_size]
                # Intervention: target_vector + alpha * direction
                # Wait, PyTorchActivationPatcher with intervention_type="add" adds patch_weight * source_tensor.
                # So patch_weight = alpha, source_tensor = direction.
                
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                prompt_last_idx = inputs.input_ids.shape[1] - 1
                
                with PyTorchActivationPatcher(model, l, direction, patch_weight=alpha, position=prompt_last_idx, intervention_type="add"):
                    res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                    steered_res.extend(res)
                    
            vals_V = []
            for idx in range(n_pairs):
                vals_V.append(steered_res[idx]["E_V"])
                
            steering_results.append({"layer": l, "alpha": alpha, "mean_E_V": np.nanmean(vals_V)})
            torch.cuda.empty_cache()
            
    steering_df = pd.DataFrame(steering_results)
    steering_df.to_csv(os.path.join(args.out_dir, "steering_results.csv"), index=False)
    print("Steering Results:")
    print(steering_df.head(15)) # Show first 3 layers

if __name__ == "__main__":
    main()
