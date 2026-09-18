import os
import gc
import sys
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def setup_logger(log_file_path: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def run_single_model_pipeline(model_path: str, model_label: str, df: pd.DataFrame, neutral_texts: list, affective_texts: list, merged: pd.DataFrame, prompt_template: str, out_dir: str, logger: logging.Logger, batch_size: int = 384):
    logger.info(f"--- Starting Pipeline for {model_label} ({model_path}) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    evaluator = InterventionEvaluator(model, tokenizer)
    num_layers = model.config.num_hidden_layers
    logger.info(f"Model loaded. Total layers: {num_layers}")

    # 1. Extract layer activations
    tensors_dir = os.path.join(out_dir, f"{model_label.lower()}_tensors")
    logger.info(f"Extracting layer activations into {tensors_dir}...")
    layer_tensors = extract_and_load_layer_tensors(
        model, tokenizer, df, model_path, tensors_dir, prompt_template
    )

    # 2. Linear Probing per layer
    logger.info("Running Linear Probing (Ridge) across all layers...")
    probing_records = []
    
    unique_pairs = np.unique(df["pair_id"].values)
    n_train = int(len(unique_pairs) * 0.7)
    train_pairs = set(unique_pairs[:n_train])
    train_mask = np.array([pid in train_pairs for pid in df["pair_id"].values])
    test_mask = ~train_mask

    for l in range(num_layers):
        X, y = [], []
        for idx, row in df.iterrows():
            key = f"{row['id']}_{row['condition']}"
            if key in layer_tensors[l]:
                X.append(layer_tensors[l][key])
                y.append(1 if row["condition"] == "affective" else 0)
        X = np.array(X)
        y = np.array(y)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        prober = FixedSplitProber(n_components=min(50, len(X_train)), task_type="classification")
        try:
            metrics = prober.evaluate(X_train, y_train, X_test, y_test)
            probing_records.append({
                "model_label": model_label,
                "layer": l,
                "roc_auc": metrics.get("roc_auc", np.nan),
                "accuracy": metrics.get("accuracy", np.nan)
            })
        except Exception as e:
            logger.warning(f"Probing layer {l} failed: {e}")
            probing_records.append({
                "model_label": model_label,
                "layer": l,
                "roc_auc": np.nan,
                "accuracy": np.nan
            })
            
    probing_df = pd.DataFrame(probing_records)
    probing_df.to_csv(os.path.join(out_dir, f"{model_label.lower()}_probing_results.csv"), index=False)
    logger.info(f"Probing complete. Peak ROC-AUC: {probing_df['roc_auc'].max():.4f}")

    # 3. Clean Evaluations
    def run_eval(texts):
        all_res = []
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i+batch_size]
            inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
            all_res.extend(res)
        return all_res

    logger.info("Running Clean (no intervention) evaluations...")
    neutral_results = run_eval(neutral_texts)
    affective_results = run_eval(affective_texts)
    
    n_pairs = len(merged)
    cln_shift_V = np.mean([affective_results[i]["E_V"] - neutral_results[i]["E_V"] for i in range(n_pairs)])
    cln_shift_A = np.mean([affective_results[i]["E_A"] - neutral_results[i]["E_A"] for i in range(n_pairs)])
    logger.info(f"Clean Affective Shift: Delta E[V] = {cln_shift_V:.4f}, Delta E[A] = {cln_shift_A:.4f}")

    # Prepare layer tensors by pair
    affective_tensors_by_layer, neutral_tensors_by_layer = [], []
    for l in range(num_layers):
        aff_t, neu_t = [], []
        for idx, row in merged.iterrows():
            id_aff = f"{row['id_affective']}_affective"
            id_neu = f"{row['id_neutral']}_neutral"
            aff_t.append(layer_tensors[l][id_aff])
            neu_t.append(layer_tensors[l][id_neu])
        affective_tensors_by_layer.append(np.stack(aff_t))
        neutral_tensors_by_layer.append(np.stack(neu_t))

    # 4. Activation Patching & Ablation & Steering across layers
    logger.info("Running Causal Interventions (Patching, Ablation, Steering)...")
    intervention_records = []
    
    for l in tqdm(range(num_layers), desc=f"{model_label} Layers"):
        # Activation Patching
        src_tensors = affective_tensors_by_layer[l]
        patched_res = []
        for i in range(0, n_pairs, batch_size):
            b_texts = neutral_texts[i:i+batch_size]
            b_src = src_tensors[i:i+batch_size]
            inputs = tokenizer(b_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            prompt_last_idx = inputs.input_ids.shape[1] - 1
            with PyTorchActivationPatcher(model, l, b_src, patch_weight=1.0, position=prompt_last_idx, intervention_type="patch"):
                res = evaluator.evaluate_sequence_logits_batch(inputs.input_ids, inputs.attention_mask)
                patched_res.extend(res)

        # Ablation
        mean_neu_tensor = np.mean(neutral_tensors_by_layer[l], axis=0)
        ablated_res = []
        for i in range(0, n_pairs, batch_size):
            b_texts = affective_texts[i:i+batch_size]
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

        intervention_records.append({
            "model_label": model_label,
            "model_name": model_path,
            "layer": l,
            "clean_delta_V": cln_shift_V,
            "clean_delta_A": cln_shift_A,
            "recovery_V": float(np.nanmean(rec_V)),
            "recovery_A": float(np.nanmean(rec_A)),
            "effect_removed_V": float(np.nanmean(effect_rem_V))
        })

        torch.cuda.empty_cache()

    interventions_df = pd.DataFrame(intervention_records)
    interventions_df.to_csv(os.path.join(out_dir, f"{model_label.lower()}_interventions.csv"), index=False)
    logger.info(f"Interventions complete for {model_label}.")
    
    # Cleanup memory
    del model
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    
    return interventions_df, cln_shift_V

def main():
    parser = argparse.ArgumentParser(description="Scaling & Multi-Family Alignment Suppression Experiments")
    parser.add_argument("--base-model", type=str, required=True, help="HuggingFace model ID for Base model")
    parser.add_argument("--instruct-model", type=str, required=True, help="HuggingFace model ID for Instruct model")
    parser.add_argument("--tag", type=str, default="", help="Custom tag for the model pair (e.g. qwen2.5_0.5b)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--base-out-dir", type=str, default="results/derived/scaling")
    parser.add_argument("--base-log-dir", type=str, default="logs/scaling")
    args = parser.parse_args()

    # Determine unique folder and log file names
    pair_tag = args.tag if args.tag else f"{args.base_model.replace('/', '_')}_vs_{args.instruct_model.replace('/', '_')}"
    out_dir = os.path.join(args.base_out_dir, pair_tag)
    log_file = os.path.join(args.base_log_dir, f"{pair_tag}.log")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.base_log_dir, exist_ok=True)
    
    logger = setup_logger(log_file)
    logger.info(f"=======================================================")
    logger.info(f"  Scaling Experiment: {pair_tag}")
    logger.info(f"  Base Model    : {args.base_model}")
    logger.info(f"  Instruct Model: {args.instruct_model}")
    logger.info(f"  Output Dir    : {out_dir}")
    logger.info(f"  Log File      : {log_file}")
    logger.info(f"=======================================================")

    df = load_data(args.split)
    prompt_path = "prompts/affective_reception_v1.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        prompt_template = "{stimulus}"

    neutral_df = df[df["condition"] == "neutral"].sort_values("id").reset_index(drop=True)
    affective_df = df[df["condition"] == "affective"].sort_values("id").reset_index(drop=True)
    merged = pd.merge(affective_df, neutral_df, left_on="matched_control_id", right_on="id", suffixes=('_affective', '_neutral'))
    
    neutral_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_neutral"]]
    affective_texts = [prompt_template.replace("{stimulus}", str(s)) for s in merged["text_affective"]]

    # Run Base
    base_df, base_delta_v = run_single_model_pipeline(
        args.base_model, "Base", df, neutral_texts, affective_texts, merged, prompt_template, out_dir, logger, args.batch_size
    )
    
    # Run Instruct
    instruct_df, instruct_delta_v = run_single_model_pipeline(
        args.instruct_model, "Instruct", df, neutral_texts, affective_texts, merged, prompt_template, out_dir, logger, args.batch_size
    )

    # Compute Suppression Gap
    logger.info("Calculating Suppression Gap (Base - Instruct)...")
    gap_records = []
    min_layers = min(len(base_df), len(instruct_df))
    for l in range(min_layers):
        b_rec = base_df.loc[l, "recovery_V"]
        i_rec = instruct_df.loc[l, "recovery_V"]
        b_abl = base_df.loc[l, "effect_removed_V"]
        i_abl = instruct_df.loc[l, "effect_removed_V"]
        
        gap_records.append({
            "tag": pair_tag,
            "layer": l,
            "base_recovery_V": b_rec,
            "instruct_recovery_V": i_rec,
            "suppression_gap_recovery_V": b_rec - i_rec,
            "base_effect_removed_V": b_abl,
            "instruct_effect_removed_V": i_abl,
            "suppression_gap_ablation_V": b_abl - i_abl
        })
        
    gap_df = pd.DataFrame(gap_records)
    gap_csv = os.path.join(out_dir, "suppression_gap.csv")
    gap_df.to_csv(gap_csv, index=False)
    
    # Calculate Overall Behavioral Suppression Ratio
    suppression_ratio = (1.0 - (abs(instruct_delta_v) / (abs(base_delta_v) + 1e-8))) * 100.0
    logger.info(f"--- Final Summary for {pair_tag} ---")
    logger.info(f"Base Clean Shift (Delta E[V]): {base_delta_v:.4f}")
    logger.info(f"Instruct Clean Shift (Delta E[V]): {instruct_delta_v:.4f}")
    logger.info(f"Behavioral Suppression Ratio: {suppression_ratio:.2f}%")
    logger.info(f"Results successfully saved to {out_dir}")

if __name__ == "__main__":
    main()
