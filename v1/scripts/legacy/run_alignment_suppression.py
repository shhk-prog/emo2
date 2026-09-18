import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from affective_empathy_eval.data import load_emobank
from affective_empathy_eval.extraction import ManifestManager, PyTorchRepresentationExtractor
from affective_empathy_eval.evaluation import InterventionEvaluator
from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.probing import FixedSplitProber

def load_data(split: str) -> pd.DataFrame:
    path = f"v1/data/processed/aipsy/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_aipsy.py first.")
    return pd.read_csv(path)

def extract_and_load_layer_tensors(model, tokenizer, df: pd.DataFrame, model_name: str, out_dir: str, prompt_template: str, position: str = "stimulus_mean_pool"):
    """
    Extracts layer representations for a given model on the dataset,
    and returns a dictionary mapping layer_idx -> {stimulus_id_condition -> numpy array}.
    """
    extractor = PyTorchRepresentationExtractor(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_revision="main"
    )
    manifest_manager = ManifestManager(out_dir)
    
    requests = []
    for idx, row in df.iterrows():
        full_prompt = prompt_template.replace("{stimulus}", str(row["text"]))
        requests.append({
            "request_id": f"{row['pair_id']}_{row['condition']}",
            "stimulus_id": row["id"],
            "condition": row["condition"],
            "stimulus_text": str(row["text"]),
            "full_prompt_text": full_prompt
        })
        
    manifests = extractor.extract_representations_batch(
        requests=requests,
        output_dir=out_dir,
        manifest_manager=manifest_manager
    )
    
    layer_tensors = {}
    for m in manifests:
        if m.extraction_position == position:
            if m.layer not in layer_tensors:
                layer_tensors[m.layer] = {}
            layer_tensors[m.layer][f"{m.stimulus_id}_{m.condition}"] = np.load(m.tensor_path)
            
    return layer_tensors

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Alignment Suppression Analysis (Base vs. Instruct)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--instruct-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-dir", type=str, default="results/derived/phase5")
    parser.add_argument("--batch-size", type=int, default=384)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading AIPsy-Affect dataset split '{args.split}'...")
    df = load_data(args.split)
    
    # Prompt template
    prompt_path = "prompts/affective_reception_v1.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        prompt_template = "{stimulus}"

    # Align pairs
    neutral_df = df[df["condition"] == "neutral"].sort_values("id").reset_index(drop=True)
    affective_df = df[df["condition"] == "affective"].sort_values("id").reset_index(drop=True)
    
    merged = pd.merge(affective_df, neutral_df, left_on="matched_control_id", right_on="id", suffixes=('_affective', '_neutral'))
    n_pairs = len(merged)
    print(f"Loaded {n_pairs} paired stimuli for Phase 5 alignment suppression evaluation.")

    neutral_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_neutral"]]
    affective_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_affective"]]

    models_to_eval = [
        ("Base", args.base_model),
        ("Instruct", args.instruct_model)
    ]
    
    all_summary_results = []

    for model_label, model_path in models_to_eval:
        print(f"\n=======================================================")
        print(f"  Evaluating Model: {model_label} ({model_path})")
        print(f"=======================================================")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        evaluator = InterventionEvaluator(model, tokenizer)
        num_layers = model.config.num_hidden_layers

        # 1. Extract layer tensors for this model
        tensors_dir = os.path.join(args.out_dir, f"{model_label.lower()}_tensors")
        print(f"Extracting layer activations for {model_label} model...")
        layer_tensors = extract_and_load_layer_tensors(
            model, tokenizer, df, model_path, tensors_dir, prompt_template
        )

        # 2. Clean evaluation (No intervention)
        def run_eval(texts):
            all_res = []
            for i in range(0, len(texts), args.batch_size):
                b_texts = texts[i:i+args.batch_size]
                inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                all_res.extend(res)
            return all_res

        print(f"[{model_label}] Running Clean evaluations...")
        neutral_results = run_eval(neutral_texts)
        affective_results = run_eval(affective_texts)

        # Baseline E[V] shift
        cln_shift_V = np.mean([affective_results[i]["E_V"] - neutral_results[i]["E_V"] for i in range(n_pairs)])
        cln_shift_A = np.mean([affective_results[i]["E_A"] - neutral_results[i]["E_A"] for i in range(n_pairs)])
        print(f"[{model_label}] Clean Affective Shift: Delta E[V] = {cln_shift_V:.4f}, Delta E[A] = {cln_shift_A:.4f}")

        # Prepare layer tensors by pair
        affective_tensors_by_layer = []
        neutral_tensors_by_layer = []
        for l in range(num_layers):
            aff_t, neu_t = [], []
            for idx, row in merged.iterrows():
                id_aff = f"{row['id_affective']}_affective"
                id_neu = f"{row['id_neutral']}_neutral"
                aff_t.append(layer_tensors[l][id_aff])
                neu_t.append(layer_tensors[l][id_neu])
            affective_tensors_by_layer.append(np.stack(aff_t))
            neutral_tensors_by_layer.append(np.stack(neu_t))

        # 3. Activation Patching & Ablation across layers
        print(f"[{model_label}] Running Causal Interventions (Patching & Ablation)...")
        for l in tqdm(range(num_layers), desc=f"{model_label} Interventions"):
            # Activation Patching
            src_tensors = affective_tensors_by_layer[l]
            patched_res = []
            for i in range(0, n_pairs, args.batch_size):
                b_texts = neutral_texts[i:i+args.batch_size]
                b_src = src_tensors[i:i+args.batch_size]
                inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                prompt_last_idx = inputs.input_ids.shape[1] - 1
                with PyTorchActivationPatcher(model, l, b_src, patch_weight=1.0, position=prompt_last_idx, intervention_type="patch"):
                    res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                    patched_res.extend(res)

            # Ablation
            mean_neu_tensor = np.mean(neutral_tensors_by_layer[l], axis=0)
            ablated_res = []
            for i in range(0, n_pairs, args.batch_size):
                b_texts = affective_texts[i:i+args.batch_size]
                inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                prompt_last_idx = inputs.input_ids.shape[1] - 1
                with PyTorchActivationPatcher(model, l, mean_neu_tensor, patch_weight=1.0, position=prompt_last_idx, intervention_type="replace"):
                    res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                    ablated_res.extend(res)

            # Calculate metrics
            rec_V, rec_A, effect_rem_V = [], [], []
            for idx in range(n_pairs):
                c_v, n_v = affective_results[idx]["E_V"], neutral_results[idx]["E_V"]
                p_v = patched_res[idx]["E_V"]
                a_v = ablated_res[idx]["E_V"]

                c_a, n_a = affective_results[idx]["E_A"], neutral_results[idx]["E_A"]
                p_a = patched_res[idx]["E_A"]

                if abs(c_v - n_v) > 1e-5:
                    rec_V.append((p_v - n_v) / (c_v - n_v))
                    effect_rem_V.append((c_v - a_v) / (c_v - n_v))
                else:
                    rec_V.append(np.nan)
                    effect_rem_V.append(np.nan)

                if abs(c_a - n_a) > 1e-5:
                    rec_A.append((p_a - n_a) / (c_a - n_a))
                else:
                    rec_A.append(np.nan)

            all_summary_results.append({
                "model_type": model_label,
                "model_name": model_path,
                "layer": l,
                "clean_delta_V": cln_shift_V,
                "clean_delta_A": cln_shift_A,
                "recovery_V": float(np.nanmean(rec_V)),
                "recovery_A": float(np.nanmean(rec_A)),
                "effect_removed_V": float(np.nanmean(effect_rem_V))
            })

            torch.cuda.empty_cache()

        # Free GPU memory before loading next model
        del model
        del evaluator
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summary_results)
    out_csv = os.path.join(args.out_dir, "alignment_suppression_base_vs_instruct.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nPhase 5 Evaluation Complete. Results saved to {out_csv}")
    
    # Compute Transmission Suppression Ratio (Instruct vs Base)
    base_df = summary_df[summary_df["model_type"] == "Base"].set_index("layer")
    inst_df = summary_df[summary_df["model_type"] == "Instruct"].set_index("layer")
    
    diff_records = []
    for l in range(len(base_df)):
        b_rec = base_df.loc[l, "recovery_V"]
        i_rec = inst_df.loc[l, "recovery_V"]
        b_abl = base_df.loc[l, "effect_removed_V"]
        i_abl = inst_df.loc[l, "effect_removed_V"]
        
        diff_records.append({
            "layer": l,
            "base_recovery_V": b_rec,
            "instruct_recovery_V": i_rec,
            "suppression_gap_recovery_V": b_rec - i_rec,
            "base_effect_removed_V": b_abl,
            "instruct_effect_removed_V": i_abl,
            "suppression_gap_ablation_V": b_abl - i_abl
        })
        
    diff_df = pd.DataFrame(diff_records)
    diff_csv = os.path.join(args.out_dir, "suppression_gap_analysis.csv")
    diff_df.to_csv(diff_csv, index=False)
    print(f"Suppression Gap Analysis saved to {diff_csv}")
    print("\nSummary Head:")
    print(diff_df.head(10))

if __name__ == "__main__":
    main()
