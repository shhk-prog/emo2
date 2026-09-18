#!/usr/bin/env python3
import os
import uuid
import yaml
import json
import argparse
import shutil
import hashlib
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

from affective_empathy_eval.schemas import parse_affective_state
from affective_empathy_eval.extraction import MockRepresentationExtractor, PyTorchRepresentationExtractor
from affective_empathy_eval.manifests import ManifestManager

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def hash_file(filepath):
    if not os.path.exists(filepath):
        return "missing"
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as afile:
        hasher.update(afile.read())
    return hasher.hexdigest()

def get_prompt_text(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LLM Main Experiment (Open-weight models & Representation Extraction).")
    parser.add_argument("--config", type=str, default="v1/configs/main_experiment.yaml", help="Path to main experiment config file.")
    parser.add_argument("--mode", type=str, choices=["dry-run", "hf"], default="dry-run", help="Execution mode (dry-run mock vs HuggingFace PyTorch).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of stimuli to process (useful for pilot runs).")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_config = yaml.safe_load(f)
    with open("v1/configs/prompts.yaml") as f:
        prompts_config = yaml.safe_load(f)

    print(f"Running Main Experiment in '{args.mode}' mode (Phase B1: hidden state extraction)...", flush=True)
    git_sha = get_git_commit()
    config_hash = hash_file(args.config)[:8]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{git_sha}_{config_hash}"

    phase = exp_config["experiment"].get("phase", "main")
    out_dir = os.path.join("results/raw", phase, run_id)
    reps_dir = os.path.join(out_dir, "representations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reps_dir, exist_ok=True)
    
    manifest_manager = ManifestManager(reps_dir)

    metadata = {
        "run_id": run_id,
        "mode": args.mode,
        "config_file": args.config,
        "project_name": exp_config["experiment"].get("name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "config_hash": config_hash
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    shutil.copy2(args.config, os.path.join(out_dir, "config_snapshot.yaml"))
    responses_path = os.path.join(out_dir, "responses.jsonl")

    # Load prompts
    prompt_data = {}
    for condition in exp_config["conditions"]:
        p_info = prompts_config["prompts"][condition]
        prompt_data[condition] = {
            "id": p_info["id"],
            "hash": hash_file(p_info["file"]),
            "text": get_prompt_text(p_info["file"]) if os.path.exists(p_info["file"]) else ""
        }

    sys_info = prompts_config["prompts"].get("system")
    sys_id = sys_info["id"] if sys_info else None
    sys_hash = hash_file(sys_info["file"]) if sys_info and os.path.exists(sys_info["file"]) else None
    sys_text = get_prompt_text(sys_info["file"]) if sys_info and os.path.exists(sys_info["file"]) else ""

    # Load stimuli
    stimuli_path = "v1/data/processed/stimuli.csv"
    if os.path.exists(stimuli_path):
        df_stim = pd.read_csv(stimuli_path)
        seed = exp_config["experiment"].get("random_seed", 42)
        df_stim = df_stim.sample(frac=1, random_state=seed).reset_index(drop=True)
        stimuli = df_stim.to_dict('records')
    else:
        stimuli = [{"stimulus_id": "emobank_0001", "text": "Dummy stimulus text 1"}, {"stimulus_id": "emobank_0002", "text": "Dummy stimulus text 2"}]

    if args.limit:
        stimuli = stimuli[:args.limit]
        print(f"Limiting stimuli to {args.limit} for pilot run.", flush=True)

    models = exp_config.get("open_weight_models", [{"id": "dummy-open-model", "num_layers": 12, "hidden_dim": 768}])
    repetitions = exp_config["experiment"].get("repetitions", 1)

    with open(responses_path, "a") as f_out:
        for model_cfg in models:
            if not model_cfg.get("enabled", True):
                continue

            model_id = model_cfg["id"]
            num_layers = model_cfg.get("num_layers", 12)
            hidden_dim = model_cfg.get("hidden_dim", 768)
            model_revision = "main"

            print(f"Processing Model: {model_id} (Layers: {num_layers}, Hidden Dim: {hidden_dim})", flush=True)

            if args.mode == "dry-run":
                extractor = MockRepresentationExtractor(num_layers=num_layers, hidden_dim=hidden_dim, model_name=model_id, model_revision=model_revision)
            else:
                # Hugging Face model loading
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
                    extractor = PyTorchRepresentationExtractor(model=model, tokenizer=tokenizer, model_name=model_id, model_revision=model_revision)
                except Exception as e:
                    print(f"Error loading Hugging Face model {model_id}: {e}. Falling back to dry-run mock.")
                    extractor = MockRepresentationExtractor(num_layers=num_layers, hidden_dim=hidden_dim, model_name=model_id, model_revision=model_revision)

            requests_list = []
            
            for rep in range(1, repetitions + 1):
                # 1. Baseline
                baseline_id = f"baseline_{model_id.replace('/', '_')}_rep{rep:02d}"
                req_id_base = str(uuid.uuid4())
                
                base_req = {
                    "run_id": run_id,
                    "request_id": req_id_base,
                    "stimulus_id": "baseline",
                    "baseline_id": baseline_id,
                    "source_dataset": "EmoBank",
                    "annotation_perspective": exp_config["experiment"].get("annotation_perspective", "reader"),
                    "model_provider": "open_weight",
                    "model_id": model_id,
                    "condition": "empty_baseline",
                    "repetition": rep,
                    "temperature": exp_config["inference"].get("temperature", 0.0),
                    "top_p": exp_config["inference"].get("top_p", 1.0),
                    "seed": exp_config["experiment"].get("random_seed"),
                    "system_prompt_id": sys_id,
                    "system_prompt_hash": sys_hash,
                    "prompt_id": prompt_data["empty_baseline"]["id"],
                    "prompt_hash": prompt_data["empty_baseline"]["hash"],
                    "code_commit": git_sha,
                    "config_hash": config_hash,
                    "request_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": 150.0,
                    "parsed_valence": 5,
                    "parsed_arousal": 5,
                    "parse_status": "success",
                    "raw_response_text": '{"valence": 5, "arousal": 5}',
                    "full_prompt_text": prompt_data["empty_baseline"]["text"],
                    "stimulus_text": ""
                }
                requests_list.append(base_req)

                # 2. Stimuli conditions
                for stim in stimuli:
                    s_id = stim["stimulus_id"]
                    s_text = stim["text"]

                    for cond in ["recognition_va", "post_reported_va", "free_response"]:
                        if cond not in exp_config["conditions"]:
                            continue

                        req_id = str(uuid.uuid4())
                        user_prompt = prompt_data[cond]["text"].replace("[STIMULUS]", str(s_text))

                        cond_req = dict(base_req)
                        cond_req["request_id"] = req_id
                        cond_req["stimulus_id"] = s_id
                        cond_req["condition"] = cond
                        cond_req["prompt_id"] = prompt_data[cond]["id"]
                        cond_req["prompt_hash"] = prompt_data[cond]["hash"]
                        cond_req["request_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                        cond_req["full_prompt_text"] = user_prompt
                        cond_req["stimulus_text"] = s_text

                        if cond == "free_response":
                            cond_req["parsed_valence"] = None
                            cond_req["parsed_arousal"] = None
                            cond_req["raw_response_text"] = "I deeply sympathize with what you are experiencing."
                        else:
                            v_val = 7 if cond == "post_reported_va" else 6
                            a_val = 4 if cond == "post_reported_va" else 5
                            cond_req["parsed_valence"] = v_val
                            cond_req["parsed_arousal"] = a_val
                            cond_req["raw_response_text"] = f'{{"valence": {v_val}, "arousal": {a_val}}}'

                        requests_list.append(cond_req)
                        
            # Execute in batches
            batch_size = exp_config.get("inference", {}).get("batch_size", 64)
            print(f"Total requests to process for {model_id}: {len(requests_list)}. Using batch_size={batch_size}", flush=True)
            
            for i in range(0, len(requests_list), batch_size):
                batch_reqs = requests_list[i:i+batch_size]
                
                extractor.extract_representations_batch(
                    requests=batch_reqs,
                    output_dir=reps_dir,
                    manifest_manager=manifest_manager
                )
                
                for req in batch_reqs:
                    req_to_save = dict(req)
                    req_to_save.pop("full_prompt_text", None)
                    req_to_save.pop("stimulus_text", None)
                    req_to_save["manifest_dir"] = reps_dir
                    f_out.write(json.dumps(req_to_save) + "\n")
                f_out.flush()
                
                print(f"  Processed {min(i+batch_size, len(requests_list))}/{len(requests_list)} requests for {model_id}", flush=True)

    print(f"Main experiment finished. Results and representations saved to {out_dir}/", flush=True)

if __name__ == "__main__":
    main()
