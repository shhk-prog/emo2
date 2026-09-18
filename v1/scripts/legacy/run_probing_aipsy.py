#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from affective_empathy_eval.manifests import ManifestManager
from affective_empathy_eval.extraction import PyTorchRepresentationExtractor
from affective_empathy_eval.probing import FixedSplitProber, run_length_control_probing, run_shuffled_control_classification

def load_data(split: str) -> pd.DataFrame:
    path = f"v1/data/processed/aipsy/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_aipsy.py first.")
    return pd.read_csv(path)

def extract_features(df: pd.DataFrame, extractor: PyTorchRepresentationExtractor, out_dir: str, prompt_template: str):
    manifest_manager = ManifestManager(os.path.join(out_dir, "manifest.jsonl"))
    
    requests = []
    for idx, row in df.iterrows():
        # Replace {stimulus} placeholder
        full_prompt = prompt_template.replace("{stimulus}", str(row["text"]))
        
        requests.append({
            "request_id": f"{row['pair_id']}_{row['condition']}",
            "stimulus_id": row["id"],
            "condition": row["condition"],
            "stimulus_text": str(row["text"]),
            "full_prompt_text": full_prompt
        })
        
    # Batch extraction
    manifests = extractor.extract_representations_batch(
        requests=requests,
        output_dir=out_dir,
        manifest_manager=manifest_manager
    )
    return manifests

def load_tensors(manifests, position: str = "stimulus_mean_pool"):
    layer_tensors = {} # layer_idx -> {request_id -> tensor}
    for m in manifests:
        if m.extraction_position == position:
            if m.layer not in layer_tensors:
                layer_tensors[m.layer] = {}
            tensor = np.load(m.tensor_path)
            layer_tensors[m.layer][f"{m.stimulus_id}_{m.condition}"] = tensor
    return layer_tensors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--position", type=str, default="stimulus_mean_pool")
    parser.add_argument("--out-dir", type=str, default="results/derived/phase2")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = load_data("train")
    dev_df = load_data("dev")
    test_df = load_data("test")
    
    if args.limit > 0:
        train_df = train_df.head(args.limit)
        dev_df = dev_df.head(args.limit)
        test_df = test_df.head(args.limit)
        
    # The prompt template should be the same as Phase 1 or a neutral instruction to process the text
    # Let's use a very simple template or the one from configs/prompts.yaml if needed
    # Using the standard affective_reception_v1.txt is ideal, but let's just use "{stimulus}" as baseline reading.
    with open("v1/configs/prompts.yaml", "r") as f:
        import yaml
        prompts = yaml.safe_load(f)
        
    # Standard reading prompt (no task implied, just read the situation)
    # Wait, we want to see if the model encodes the affective state of the situation.
    # The default prompt for reception is in prompts/affective_reception_v1.txt
    prompt_path = "prompts/affective_reception_v1.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        prompt_template = "{stimulus}"
        
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    extractor = PyTorchRepresentationExtractor(model, tokenizer, args.model, "main")
    
    print("Extracting features for Train...")
    train_manifests = extract_features(train_df, extractor, os.path.join(args.out_dir, "train_tensors"), prompt_template)
    print("Extracting features for Dev...")
    dev_manifests = extract_features(dev_df, extractor, os.path.join(args.out_dir, "dev_tensors"), prompt_template)
    print("Extracting features for Test...")
    test_manifests = extract_features(test_df, extractor, os.path.join(args.out_dir, "test_tensors"), prompt_template)
    
    train_layer_tensors = load_tensors(train_manifests, args.position)
    dev_layer_tensors = load_tensors(dev_manifests, args.position)
    test_layer_tensors = load_tensors(test_manifests, args.position)
    
    # Prepare labels (Affective = 1, Neutral = 0)
    train_y = np.array([1 if c == "affective" else 0 for c in train_df["condition"]])
    dev_y = np.array([1 if c == "affective" else 0 for c in dev_df["condition"]])
    test_y = np.array([1 if c == "affective" else 0 for c in test_df["condition"]])
    
    # Prepare length (word_count) for control probe
    train_len = train_df["word_count"].values
    dev_len = dev_df["word_count"].values
    test_len = test_df["word_count"].values
    
    # Match the order with DataFrame (since tensors were extracted in batch order, the ids match)
    def align_tensors(df, layer_tensors, layer):
        X = []
        for idx, row in df.iterrows():
            key = f"{row['id']}_{row['condition']}"
            X.append(layer_tensors[layer][key])
        return np.vstack(X)
        
    prober = FixedSplitProber(task_type="classification")
    
    results = []
    
    layers = sorted(list(train_layer_tensors.keys()))
    print(f"Running probing across {len(layers)} layers...")
    for layer in layers:
        X_train = align_tensors(train_df, train_layer_tensors, layer)
        X_dev = align_tensors(dev_df, dev_layer_tensors, layer)
        X_test = align_tensors(test_df, test_layer_tensors, layer)
        
        # 1. Main Probe (Affective vs Neutral)
        res = prober.evaluate(X_train, train_y, X_test, test_y)
        
        # 2. Shuffled Control Probe
        shuf_res = run_shuffled_control_classification(X_train, train_y, X_test, test_y, prober)
        
        # 3. Length Control Probe
        len_res = run_length_control_probing(X_train, train_len, X_test, test_len)
        
        results.append({
            "layer": layer,
            "accuracy": res.get("accuracy", 0),
            "roc_auc": res.get("roc_auc", 0.5),
            "shuffled_accuracy_mean": shuf_res["mean_shuffled_accuracy"],
            "length_r2": len_res.get("r2", 0)
        })
        
    results_df = pd.DataFrame(results)
    print(results_df)
    results_csv = os.path.join(args.out_dir, "probing_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Probing complete. Results saved to {results_csv}")

if __name__ == "__main__":
    main()
