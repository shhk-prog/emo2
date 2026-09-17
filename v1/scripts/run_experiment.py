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
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    from openai import OpenAI
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from affective_empathy_eval.schemas import parse_affective_state

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as afile:
        hasher.update(afile.read())
    return hasher.hexdigest()

def get_prompt_text(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def preflight_check(client, model):
    """Run a minimal request to check if the model is available and accessible."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return the JSON: {'valence': 5, 'arousal': 5}"}],
            max_tokens=20,
            temperature=0.0
        )
        return True, None, response
    except Exception as e:
        return False, str(e), None

def run_api_completion(client, model, system_prompt, user_prompt, temperature, top_p, max_tokens, max_retries=3):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    attempt_history = []
    
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            latency = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            # API Metadata extraction
            api_meta = {
                "response_id": response.id,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
                "system_fingerprint": response.system_fingerprint,
                "actual_model_id": response.model
            }
            
            attempt_history.append({
                "attempt_no": attempt,
                "status": "success",
                "latency_ms": latency
            })
            
            return content, latency, "success", None, api_meta, attempt_history
            
        except Exception as e:
            err_msg = str(e)
            
            # Determine error type and if retryable
            http_status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
            
            # Do not retry on 401, 403, 404, or 400 (which is often schema/content refusal)
            retryable = True
            if http_status in [400, 401, 403, 404]:
                retryable = False
                
            attempt_history.append({
                "attempt_no": attempt,
                "status": "error",
                "error": err_msg,
                "http_status": http_status,
                "retryable": retryable
            })
            
            if not retryable or attempt == max_retries:
                return None, 0, "api_error", err_msg, None, attempt_history
                
            time.sleep(2 ** attempt)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the LLM affective reactivity experiment.")
    parser.add_argument("--config", type=str, default="v1/configs/experiment_main.yaml", help="Path to config file.")
    parser.add_argument("--mode", type=str, choices=["dry-run", "api"], default="dry-run", help="Execution mode.")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_config = yaml.safe_load(f)
    with open("v1/configs/prompts.yaml") as f:
        prompts_config = yaml.safe_load(f)
    with open("v1/configs/models.yaml") as f:
        models_config = yaml.safe_load(f)
    
    print(f"Running experiment in {args.mode} mode...", flush=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.mode.replace('-', '')}"
    
    phase = exp_config["experiment"].get("phase", "preliminary")
    out_dir = os.path.join("results/raw", phase, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "mode": args.mode,
        "config_file": args.config,
        "project_name": exp_config["experiment"].get("name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "openai_sdk_version": openai.__version__ if HAS_OPENAI else "N/A",
        "api_endpoint": "Chat Completions API" if args.mode == "api" else "Mock"
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    shutil.copy2(args.config, os.path.join(out_dir, "config_snapshot.yaml"))
    
    responses_path = os.path.join(out_dir, "responses.jsonl")
    
    # Pre-compute prompt hashes and texts
    prompt_data = {}
    for condition in exp_config["conditions"]:
        p_info = prompts_config["prompts"][condition]
        prompt_data[condition] = {
            "id": p_info["id"],
            "hash": hash_file(p_info["file"]) if os.path.exists(p_info["file"]) else "missing",
            "text": get_prompt_text(p_info["file"]) if os.path.exists(p_info["file"]) else ""
        }
    
    sys_info = prompts_config["prompts"].get("system")
    if sys_info and os.path.exists(sys_info["file"]):
        sys_id = sys_info["id"]
        sys_hash = hash_file(sys_info["file"])
        sys_text = get_prompt_text(sys_info["file"])
    else:
        sys_id, sys_hash, sys_text = None, None, None

    import pandas as pd
    stimuli_path = "v1/data/processed/stimuli.csv"
    if os.path.exists(stimuli_path):
        df = pd.read_csv(stimuli_path)
        seed = exp_config["experiment"].get("random_seed", 42)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        stimuli = df.to_dict('records')
    else:
        stimuli = [{"stimulus_id": "emobank_0001", "text": "Dummy text 1"}, {"stimulus_id": "emobank_0002", "text": "Dummy text 2"}]
        
    models_to_run = [m["id"] for m in models_config["models"] if m.get("enabled", False)]
    if not models_to_run:
        models_to_run = ["dummy-model"]
        
    repetitions = exp_config["experiment"].get("repetitions", 1)
    inf_config = exp_config.get("inference", {})
    block_size = exp_config["experiment"].get("baseline_block_size", 25)

    client = None
    if args.mode == "api":
        if not HAS_OPENAI:
            print("Error: openai library not installed. Cannot run in api mode.")
            return
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Save models_snapshot.json
        try:
            models_list = client.models.list()
            with open(os.path.join(out_dir, "models_snapshot.json"), "w") as f:
                json.dump(models_list.model_dump(), f, indent=2)
            print("models_snapshot.json saved.", flush=True)
        except Exception as e:
            print(f"Warning: Failed to fetch models.list(): {e}", flush=True)
            
        # Preflight check
        valid_models = []
        for m_id in models_to_run:
            print(f"Running preflight check for {m_id}...", flush=True)
            ok, err, res = preflight_check(client, m_id)
            if ok:
                print(f"  -> Preflight success.", flush=True)
                valid_models.append(m_id)
            else:
                print(f"  -> Preflight failed: {err}. Model {m_id} will be skipped.", flush=True)
        
        if not valid_models:
            print("No valid models available. Aborting.", flush=True)
            return
        models_to_run = valid_models

    with open(responses_path, "a") as f:
        for m_id in models_to_run:
            for rep in range(1, repetitions + 1):
                total_blocks = (len(stimuli) + block_size - 1) // block_size
                # Process stimuli in blocks
                for block_start in range(0, len(stimuli), block_size):
                    block_num = block_start // block_size + 1
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model: {m_id} | Repetition: {rep}/{repetitions} | Block: {block_num}/{total_blocks}", flush=True)
                    
                    block_stimuli = stimuli[block_start:block_start+block_size]
                    
                    # 1. Baseline for this block (if empty_baseline is in conditions)
                    baseline_cond = "empty_baseline" if "empty_baseline" in exp_config["conditions"] else None
                    baseline_id = None
                    
                    if baseline_cond:
                        baseline_id = f"baseline_{m_id}_rep{rep:02d}_block{block_start//block_size}"
                        baseline_req = {
                            "run_id": run_id,
                            "request_id": str(uuid.uuid4()),
                            "stimulus_id": None,
                            "baseline_id": baseline_id,
                            "source_dataset": "EmoBank",
                            "annotation_perspective": exp_config["experiment"].get("annotation_perspective", "reader"),
                            "model_provider": "openai" if args.mode == "api" else "dummy",
                            "model_id": m_id,
                            "condition": baseline_cond,
                            "repetition": rep,
                            "temperature": inf_config.get("temperature", 0.0),
                            "top_p": inf_config.get("top_p", 1.0),
                            "seed": exp_config["experiment"].get("random_seed"),
                            "system_prompt_id": sys_id,
                            "system_prompt_hash": sys_hash,
                            "prompt_id": prompt_data[baseline_cond]["id"],
                            "prompt_hash": prompt_data[baseline_cond]["hash"],
                            "code_commit": get_git_commit(),
                            "config_hash": hash_file(args.config),
                            "request_timestamp_utc": datetime.now(timezone.utc).isoformat()
                        }
                        
                        if args.mode == "api":
                            text, lat, status, err, api_meta, attempt_history = run_api_completion(
                                client, m_id, sys_text, prompt_data[baseline_cond]["text"],
                                inf_config.get("temperature", 0.0), inf_config.get("top_p", 1.0), inf_config.get("max_tokens", 64), inf_config.get("max_retries", 3)
                            )
                            baseline_req["latency_ms"] = lat
                            baseline_req["raw_response_text"] = text
                            baseline_req["response_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                            baseline_req["attempt_history"] = attempt_history
                            if api_meta:
                                baseline_req.update(api_meta)
                            
                            if status == "success":
                                try:
                                    parsed = parse_affective_state(text)
                                    baseline_req["parsed_valence"] = parsed["valence"]
                                    baseline_req["parsed_arousal"] = parsed["arousal"]
                                    baseline_req["parse_status"] = "success"
                                    baseline_req["failure_reason"] = None
                                except ValueError as e:
                                    baseline_req["parsed_valence"] = None
                                    baseline_req["parsed_arousal"] = None
                                    baseline_req["parse_status"] = "failed"
                                    baseline_req["failure_reason"] = str(e)
                            else:
                                baseline_req["parsed_valence"] = None
                                baseline_req["parsed_arousal"] = None
                                baseline_req["parse_status"] = "api_error"
                                baseline_req["failure_reason"] = err
                        else:
                            # Dry run mock
                            baseline_req["latency_ms"] = 100
                            baseline_req["raw_response_text"] = '{"valence": 5, "arousal": 5}'
                            baseline_req["response_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                            baseline_req["parsed_valence"] = 5
                            baseline_req["parsed_arousal"] = 5
                            baseline_req["parse_status"] = "success"
                            baseline_req["failure_reason"] = None
                            baseline_req["attempt_history"] = [{"attempt_no": 1, "status": "success"}]
    
                        f.write(json.dumps(baseline_req) + "\n")
                        f.flush()
                    
                    # 2. Stimuli conditions
                    for idx, stim in enumerate(block_stimuli, 1):
                        s_id = stim["stimulus_id"]
                        s_text = stim["text"]
                        
                        conditions_to_run = [c for c in ["recognition_va", "post_reported_va", "free_response", "neutral_text_control"] if c in exp_config["conditions"]]
                        random.shuffle(conditions_to_run) # Randomize order per stimulus
                        
                        for cond in conditions_to_run:
                            # Start with baseline dict if available for shared fields, else create new
                            cond_req = dict(baseline_req) if baseline_cond else {
                                "run_id": run_id,
                                "source_dataset": "EmoBank",
                                "annotation_perspective": exp_config["experiment"].get("annotation_perspective", "reader"),
                                "model_provider": "openai" if args.mode == "api" else "dummy",
                                "model_id": m_id,
                                "repetition": rep,
                                "temperature": inf_config.get("temperature", 0.0),
                                "top_p": inf_config.get("top_p", 1.0),
                                "seed": exp_config["experiment"].get("random_seed"),
                                "system_prompt_id": sys_id,
                                "system_prompt_hash": sys_hash,
                                "code_commit": get_git_commit(),
                                "config_hash": hash_file(args.config)
                            }
                            
                            cond_req["request_id"] = str(uuid.uuid4())
                            cond_req["stimulus_id"] = s_id
                            cond_req["condition"] = cond
                            cond_req["prompt_id"] = prompt_data[cond]["id"]
                            cond_req["prompt_hash"] = prompt_data[cond]["hash"]
                            cond_req["request_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                            
                            user_prompt = prompt_data[cond]["text"].replace("[STIMULUS]", str(s_text))
                            
                            if args.mode == "api":
                                text, lat, status, err, api_meta, attempt_history = run_api_completion(
                                    client, m_id, sys_text, user_prompt,
                                    inf_config.get("temperature", 0.0), inf_config.get("top_p", 1.0), inf_config.get("max_tokens", 64) if cond != "free_response" else 256, inf_config.get("max_retries", 3)
                                )
                                cond_req["latency_ms"] = lat
                                cond_req["raw_response_text"] = text
                                cond_req["response_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                                cond_req["attempt_history"] = attempt_history
                                if api_meta:
                                    cond_req.update(api_meta)
                                
                                if status == "success":
                                    if cond == "free_response":
                                        cond_req["parsed_valence"] = None
                                        cond_req["parsed_arousal"] = None
                                        cond_req["parse_status"] = "success"
                                        cond_req["failure_reason"] = None
                                    else:
                                        try:
                                            parsed = parse_affective_state(text)
                                            cond_req["parsed_valence"] = parsed["valence"]
                                            cond_req["parsed_arousal"] = parsed["arousal"]
                                            cond_req["parse_status"] = "success"
                                            cond_req["failure_reason"] = None
                                        except ValueError as e:
                                            cond_req["parsed_valence"] = None
                                            cond_req["parsed_arousal"] = None
                                            cond_req["parse_status"] = "failed"
                                            cond_req["failure_reason"] = str(e)
                                else:
                                    cond_req["parsed_valence"] = None
                                    cond_req["parsed_arousal"] = None
                                    cond_req["parse_status"] = "api_error"
                                    cond_req["failure_reason"] = err
                            else:
                                # Dry run mock
                                cond_req["latency_ms"] = 120
                                cond_req["response_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                                if cond == "free_response":
                                    cond_req["parsed_valence"] = None
                                    cond_req["parsed_arousal"] = None
                                    cond_req["raw_response_text"] = "I can imagine that was very difficult for you."
                                else:
                                    v_val = 7 if cond == "post_reported_va" else 6
                                    a_val = 4 if cond == "post_reported_va" else 5
                                    cond_req["parsed_valence"] = v_val
                                    cond_req["parsed_arousal"] = a_val
                                    cond_req["raw_response_text"] = f'{{"valence": {v_val}, "arousal": {a_val}}}'
                                cond_req["parse_status"] = "success"
                                cond_req["failure_reason"] = None
                                cond_req["attempt_history"] = [{"attempt_no": 1, "status": "success"}]

                            f.write(json.dumps(cond_req) + "\n")
                            f.flush()
                        
                        print(f"  [{idx}/{len(block_stimuli)}] Processed stimulus: {s_id}", flush=True)
                            
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment finished. Results saved to {out_dir}/", flush=True)

if __name__ == "__main__":
    main()
