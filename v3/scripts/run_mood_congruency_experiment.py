#!/usr/bin/env python3
"""
Mood Congruency Causal Experiment:
Evaluating whether steering the internal affective representation (induced mood)
causally biases the model's objective recognition of others' emotions.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from v2.src.likelihood import generate_81_candidates, compute_expected_va


def compute_likelihoods_for_candidates_batched(model, tokenizer, prompt, candidates, device=None):
    """
    Batched computation of candidate log-likelihoods.
    Guarantees exact numerical equivalence to unbatched calculation.
    """
    if device is None:
        device = model.device

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    all_input_ids = []
    cand_token_lens = []
    for cand in candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        all_input_ids.append(prompt_ids + cand_ids)
        cand_token_lens.append(len(cand_ids))
        
    max_len = max(len(ids) for ids in all_input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    
    batch_input_ids = []
    batch_attention_mask = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_token_id] * pad_len)
        batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
        
    input_tensor = torch.tensor(batch_input_ids, device=device)
    mask_tensor = torch.tensor(batch_attention_mask, device=device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
        logits = outputs.logits  # (batch_size, max_len, vocab_size)
        
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    likelihoods = []
    candidate_lengths = []
    prompt_end_idx = prompt_len - 1
    
    for i, cand_len in enumerate(cand_token_lens):
        cand_ids = all_input_ids[i][prompt_len : prompt_len + cand_len]
        logit_indices = [prompt_end_idx + j for j in range(cand_len)]
        
        cand_log_probs = [
            log_probs[i, logit_idx, tok_id].item() 
            for logit_idx, tok_id in zip(logit_indices, cand_ids)
        ]
        likelihoods.append(sum(cand_log_probs))
        candidate_lengths.append(cand_len)
        
    return likelihoods, candidate_lengths


def apply_recognition_prompt(tokenizer, text, is_instruct=True):
    """
    Format prompt for 3rd-person objective recognition of emotion.
    """
    if is_instruct:
        messages = [
            {"role": "system", "content": "You are an expert annotator in affective psychology."},
            {"role": "user", "content": f"Read the following text and estimate the emotional state of the writer/speaker.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Read the following text and estimate the emotional state of the writer/speaker.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9).\n\nOutput:\n"
    return prompt


def compute_contrastive_directions(hidden_states, df_train, layer):
    """
    Compute contrastive directions for Valence and Arousal using annotated train set.
    """
    pos_emotions = ['ecstasy', 'admiration', 'amazement']
    neg_emotions = ['rage', 'grief', 'terror', 'loathing']
    high_a_emotions = ['rage', 'terror', 'ecstasy', 'amazement']
    low_a_emotions = ['grief', 'loathing']

    pos_reps, neg_reps = [], []
    high_a_reps, low_a_reps = [], []

    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states:
            continue
        
        rep = hidden_states[key]
        emo = str(row.get('emotion', '')).lower()
        cond = str(row.get('condition', '')).lower()
        
        if emo in pos_emotions:
            pos_reps.append(rep)
        elif emo in neg_emotions:
            neg_reps.append(rep)
            
        if emo in high_a_emotions:
            high_a_reps.append(rep)
        elif emo in low_a_emotions or cond == 'neutral':
            low_a_reps.append(rep)

    if len(pos_reps) > 0 and len(neg_reps) > 0:
        d_V = np.mean(pos_reps, axis=0) - np.mean(neg_reps, axis=0)
        n = np.linalg.norm(d_V)
        if n > 0:
            d_V = d_V / n
    else:
        d_V = None
        
    if len(high_a_reps) > 0 and len(low_a_reps) > 0:
        d_A = np.mean(high_a_reps, axis=0) - np.mean(low_a_reps, axis=0)
        n = np.linalg.norm(d_A)
        if n > 0:
            d_A = d_A / n
    else:
        d_A = None
        
    # Sign alignment using correlation with ground truth
    z_V_list, z_A_list = [], []
    v_h_list, a_h_list = [], []
    
    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states:
            continue
        rep = hidden_states[key]
        
        if pd.notna(row.get('V_H')) and pd.notna(row.get('A_H')):
            if d_V is not None: 
                z_V_list.append(np.dot(rep, d_V))
                v_h_list.append(row['V_H'])
            if d_A is not None: 
                z_A_list.append(np.dot(rep, d_A))
                a_h_list.append(row['A_H'])
                
    if d_V is not None and len(z_V_list) > 1:
        r, _ = pearsonr(z_V_list, v_h_list)
        if r < 0:
            d_V = -d_V
            z_V_list = [-z for z in z_V_list]
            
    if d_A is not None and len(z_A_list) > 1:
        r, _ = pearsonr(z_A_list, a_h_list)
        if r < 0:
            d_A = -d_A
            z_A_list = [-z for z in z_A_list]
        
    # Standard deviation (sigma) of projection scores
    z_V_list, z_A_list = [], []
    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states:
            continue
        rep = hidden_states[key]
        if d_V is not None:
            z_V_list.append(np.dot(rep, d_V))
        if d_A is not None:
            z_A_list.append(np.dot(rep, d_A))
        
    sigma_V = np.std(z_V_list) if len(z_V_list) > 0 else 1.0
    sigma_A = np.std(z_A_list) if len(z_A_list) > 0 else 1.0
        
    return d_V, d_A, sigma_V, sigma_A


def get_steering_hook(direction, alpha, sigma, prompt_length):
    """
    Hook to steer residual activations at prompt boundary.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
            
        intervention = alpha * sigma * (direction / np.linalg.norm(direction))
        intervention_tensor = torch.tensor(intervention, dtype=h.dtype, device=h.device)
        
        target_pos = prompt_length - 1
        if target_pos < h.shape[1]:
            h[:, target_pos:, :] += intervention_tensor
            
        if isinstance(output, tuple):
            return (h,) + output[1:]
        else:
            return h
    return hook


def main():
    parser = argparse.ArgumentParser(description="Run Mood Congruency Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--is_instruct", action="store_true", default=True)
    parser.add_argument("--stimuli-file", type=str, default="v1/data/processed/stimuli.csv")
    parser.add_argument("--filter-subset", type=str, choices=["all", "ambiguous", "extreme"], default="ambiguous",
                        help="ambiguous: |V_scaled| <= 0.25 (V ~ 2.5-3.5), extreme: |V_scaled| > 0.35")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of stimuli (-1 for all)")
    parser.add_argument("--layers", type=int, nargs='+', default=[14, 16, 20])
    parser.add_argument("--alphas", type=float, nargs='+', default=[-3.0, -1.5, 0.0, 1.5, 3.0])
    parser.add_argument("--directions", type=str, nargs='+', default=["valence", "random"])
    parser.add_argument("--train-data", type=str, default="v2/data/processed/aipsy_annotated/train_strict.csv")
    parser.add_argument("--hidden-states-file", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_hidden_states.npz")
    parser.add_argument("--out-dir", type=str, default="v3/results/raw/mood_congruency")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without loading model weights")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Stimuli
    stimuli_path = os.path.join(PROJECT_ROOT, args.stimuli_file)
    print(f"Loading stimuli from {stimuli_path}...")
    df_stimuli = pd.read_csv(stimuli_path)
    
    if args.filter_subset == "ambiguous":
        # EmoBank scaled: V_scaled is centered around 0 (raw V around 3.0)
        if 'V_scaled' in df_stimuli.columns:
            df_stimuli = df_stimuli[df_stimuli['V_scaled'].abs() <= 0.25]
        elif 'V' in df_stimuli.columns:
            df_stimuli = df_stimuli[(df_stimuli['V'] >= 2.5) & (df_stimuli['V'] <= 3.5)]
    elif args.filter_subset == "extreme":
        if 'V_scaled' in df_stimuli.columns:
            df_stimuli = df_stimuli[df_stimuli['V_scaled'].abs() > 0.35]
            
    if args.limit > 0:
        df_stimuli = df_stimuli.head(args.limit)
    print(f"Selected {len(df_stimuli)} stimuli (subset: {args.filter_subset})")

    # 2. Load Hidden States for Contrastive Direction
    hs_file = os.path.join(PROJECT_ROOT, args.hidden_states_file)
    train_file = os.path.join(PROJECT_ROOT, args.train_data)
    
    print(f"Loading hidden states from {hs_file}...")
    hidden_states = dict(np.load(hs_file, allow_pickle=True))
    for k, v in hidden_states.items():
        if v.shape == ():
            hidden_states[k] = v.item()
    df_train = pd.read_csv(train_file)

    # 3. Dry-run Mode Check
    if args.dry_run:
        print("[DRY RUN MODE] Model inference skipped. Verifying setup...")
        for l in args.layers:
            d_V, d_A, s_V, s_A = compute_contrastive_directions(hidden_states, df_train, l)
            print(f"Layer {l}: d_V norm={np.linalg.norm(d_V) if d_V is not None else 'None'}, sigma_V={s_V:.4f}")
        print("Dry run completed successfully.")
        return

    # 4. Load Model and Tokenizer
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    model.eval()

    candidates, va_pairs = generate_81_candidates("standard")

    model_clean = args.model.replace("/", "_")
    out_file = os.path.join(PROJECT_ROOT, args.out_dir, f"{model_clean}_mood_congruency_{args.filter_subset}.jsonl")
    print(f"Results will be written to: {out_file}")

    with open(out_file, "w") as f_out:
        for l in args.layers:
            print(f"\n================ Processing Layer {l} ================")
            d_V, d_A, sigma_V, sigma_A = compute_contrastive_directions(hidden_states, df_train, l)

            for direction_name in args.directions:
                if direction_name == "valence":
                    target_d = d_V
                    target_sigma = sigma_V
                elif direction_name == "arousal":
                    target_d = d_A
                    target_sigma = sigma_A
                elif direction_name == "random":
                    np.random.seed(42 + l)
                    dim = len(d_V) if d_V is not None else 1536
                    v = np.random.randn(dim)
                    target_d = v / np.linalg.norm(v)
                    target_sigma = sigma_V if d_V is not None else 1.0
                else:
                    raise ValueError(f"Unknown direction: {direction_name}")

                if target_d is None:
                    print(f"Direction {direction_name} could not be computed for layer {l}. Skipping.")
                    continue

                for alpha in args.alphas:
                    total_stim = len(df_stimuli)
                    print(f"Layer {l} | Direction: {direction_name} | Alpha: {alpha} ({total_stim} stimuli)")
                    
                    for idx, (_, row) in enumerate(df_stimuli.iterrows()):
                        stim_id = row.get('id', str(idx))
                        text = row['text']
                        v_human = row.get('V', np.nan)
                        a_human = row.get('A', np.nan)

                        prompt = apply_recognition_prompt(tokenizer, text, is_instruct=args.is_instruct)
                        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                        prompt_length = len(prompt_tokens)

                        hook_fn = get_steering_hook(target_d, alpha, target_sigma, prompt_length)
                        layer_module = model.model.layers[l]
                        handle = layer_module.register_forward_hook(hook_fn)

                        try:
                            # 1. Sequence Likelihood Protocol (81 candidates in single batched forward pass)
                            cand_probs, _ = compute_likelihoods_for_candidates_batched(model, tokenizer, prompt, candidates)
                            e_v, e_a, entropy, p_55, probs = compute_expected_va(cand_probs, va_pairs)
                            max_idx = int(np.argmax(probs))
                            v_greedy, a_greedy = va_pairs[max_idx]
                            p_max = float(probs[max_idx])

                            record = {
                                "model": args.model,
                                "layer": l,
                                "direction": direction_name,
                                "alpha": alpha,
                                "stimulus_id": stim_id,
                                "text": text,
                                "v_human": v_human,
                                "a_human": a_human,
                                "expected_v_rec": float(e_v),
                                "expected_a_rec": float(e_a),
                                "v_greedy": int(v_greedy),
                                "a_greedy": int(a_greedy),
                                "p_max": p_max,
                                "entropy": float(entropy),
                                "p_55": float(p_55),
                                "subset": args.filter_subset
                            }
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()

                        finally:
                            handle.remove()

    print(f"\nAll conditions finished. Results saved to {out_file}")


if __name__ == "__main__":
    main()
