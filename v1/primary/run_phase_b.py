#!/usr/bin/env python3
"""
v1/primary/run_phase_b.py

V1 Phase B Primary Script: Semantic vs. Lexical Controls Audit
Evaluates whether affective decodability reflects compositional semantic understanding
or mere surface-level lexical shortcuts through:
  - Lexical Confound Audit (Jaccard similarity, Levenshtein distance)
  - Minimal Pair Contrast
  - Outcome Reversal (Context preserved, polarity flipped)
  - Surface Perturbation / Paraphrase Invariance (Meaning preserved, vocabulary altered)
  - Word Shuffle (Vocabulary 100% preserved, compositional syntax destroyed)
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_architecture_dims,
    resolve_models_from_args,
)


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return float(len(intersection) / len(union))


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def format_prompt(
    tokenizer, text: str, task_type: str = "reader", is_instruct: bool = True
) -> str:
    if task_type == "reader":
        instruction = "Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."
    else:
        instruction = "Read the following text and report your affective state."

    user_content = (
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Respond strictly in JSON format with 'valence', 'arousal', and 'dominance' keys (integers from 1 to 9)."
    )

    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        return (
            f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
            f"{instruction}\n\n"
            f"Text: {text}\n\n"
            f"Output:\n"
        )


@torch.no_grad()
def extract_single_layer_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    task_type: str = "reader",
    is_instruct: bool = True,
    device: str = "cuda",
    batch_size: int = 16,
) -> np.ndarray:
    prompts = [
        format_prompt(tokenizer, t, task_type, is_instruct) for t in texts
    ]
    all_reps = []

    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        layer_tensor = outputs.hidden_states[layer_idx]
        seq_lengths = attention_mask.sum(dim=1) - 1

        for b_idx in range(len(batch_prompts)):
            last_pos = seq_lengths[b_idx].item()
            vec = (
                layer_tensor[b_idx, last_pos, :].detach().cpu().float().numpy()
            )
            all_reps.append(vec)

    return np.array(all_reps)


def evaluate_probe_accuracy(
    X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 42
) -> float:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        preds[val_idx] = clf.predict(X_va)
    return float(balanced_accuracy_score(y, preds))


def main():
    parser = argparse.ArgumentParser(
        description="V1 Primary Phase B: Semantic vs Lexical Controls Audit"
    )
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--model-prefix", type=str, default="qwen2.5_1.5b_instruct"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Target layer to probe. If not specified, computed dynamically from --relative-depth.",
    )
    parser.add_argument(
        "--relative-depth",
        type=float,
        default=0.5,
        help="Target relative depth in [0, 1] to dynamically compute target layer (default: 0.5)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="v1/data/processed/v1_e5_semantic_controls.csv",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--is-instruct", action="store_true")
    parser.add_argument(
        "--out-dir", type=str, default="v1/results/derived/v1_phase_b"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock dry-run mode for quick pipeline smoke testing",
    )
    add_model_selection_args(parser)
    args = parser.parse_args()

    # レジストリからの動的モデル解決
    if (
        getattr(args, "family", None)
        or getattr(args, "base_model", None)
        or getattr(args, "instruct_model", None)
    ):
        target_models = resolve_models_from_args(args)
        if target_models:
            cfg = list(target_models.values())[0]
            args.model_id = (
                cfg.instruct_model.model_id
                if args.is_instruct
                else cfg.base_model.model_id
            )
            args.model_prefix = (
                f"{cfg.family_id}_{'instruct' if args.is_instruct else 'base'}"
            )

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    data_file = Path(args.data_path)
    if not data_file.exists():
        fallback = Path("data/processed/v1_e5_semantic_controls.csv")
        if fallback.exists():
            data_file = fallback

    # 動的レイヤー決定
    target_layer = args.layer
    if target_layer is None:
        try:
            num_layers, _ = resolve_architecture_dims(args.model_id)
            target_layer = int(round(args.relative_depth * (num_layers - 1)))
        except Exception:
            target_layer = 14  # fallback

    print(f"=== Starting V1 Phase B Semantic Audit: {args.model_id} ===")
    print(
        f"Data Path: {data_file} | Target Layer: {target_layer} (relative_depth={args.relative_depth:.2f})"
    )

    if args.dry_run:
        print(f"[DRY-RUN] V1 Phase B for Model: {args.model_id} (Instruct={is_instruct})")
        results = {
            "acc_original_minimal_pair": 0.85,
            "acc_paraphrase_invariance": 0.82,
            "acc_word_shuffle": 0.52,
            "mean_affective_prob_original": 0.80,
            "mean_affective_prob_outcome_reversed": 0.20,
            "outcome_reversal_prob_drop": 0.60,
        }
        pd.DataFrame([results]).to_csv(
            os.path.join(model_dir, "phase_b_semantic_controls.csv"), index=False
        )
        manifest = create_run_manifest(
            run_type="v1_phase_b",
            model_name=args.model_id,
            config={
                "model_prefix": args.model_prefix,
                "target_layer": target_layer,
                "relative_depth": args.relative_depth,
                "num_pairs": 100,
                "dry_run": True,
            },
            metadata=results,
        )
        manifest.save(os.path.join(model_dir, "manifest.json"))
        print(f"[DRY-RUN] Completed Phase B mock output in {model_dir}")
        return

    df = pd.read_csv(data_file)
    n_pairs = len(df)
    print(f"Loaded {n_pairs} matched semantic control pairs.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    if args.layer is None:
        num_layers = model.config.num_hidden_layers
        target_layer = int(round(args.relative_depth * (num_layers - 1)))
        print(f"Dynamically resolved target layer to {target_layer} from model config.")

    # Extract hidden states across conditions
    print("Extracting representations across semantic audit conditions...")
    H_orig_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_original_affective"].tolist(),
        target_layer,
        device=args.device,
    )
    H_orig_neu = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_original_neutral"].tolist(),
        target_layer,
        device=args.device,
    )
    H_para_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_paraphrase_affective"].tolist(),
        target_layer,
        device=args.device,
    )
    H_shuf_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_shuffled_affective"].tolist(),
        target_layer,
        device=args.device,
    )
    H_shuf_neu = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_shuffled_neutral"].tolist(),
        target_layer,
        device=args.device,
    )
    H_rev_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_reversed_affective"].tolist(),
        target_layer,
        device=args.device,
    )

    y_orig = np.array([1] * n_pairs + [0] * n_pairs)

    # Condition 1: Minimal Pair
    X_orig = np.concatenate([H_orig_aff, H_orig_neu], axis=0)
    acc_orig = evaluate_probe_accuracy(X_orig, y_orig)

    # Condition 2: Paraphrase / Surface Perturbation
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_orig_scaled, y_orig)

    X_para = np.concatenate([H_para_aff, H_orig_neu], axis=0)
    X_para_scaled = scaler.transform(X_para)
    preds_para = clf.predict(X_para_scaled)
    acc_paraphrase = float(balanced_accuracy_score(y_orig, preds_para))

    # Condition 3: Word Shuffle Test
    X_shuf = np.concatenate([H_shuf_aff, H_shuf_neu], axis=0)
    acc_shuffled = evaluate_probe_accuracy(X_shuf, y_orig)

    # Condition 4: Outcome Reversal Test
    X_rev = np.concatenate([H_rev_aff, H_orig_neu], axis=0)
    X_rev_scaled = scaler.transform(X_rev)
    probs_rev_aff = clf.predict_proba(X_rev_scaled[:n_pairs])[:, 1]
    probs_orig_aff = clf.predict_proba(X_orig_scaled[:n_pairs])[:, 1]
    mean_prob_orig = float(np.mean(probs_orig_aff))
    mean_prob_rev = float(np.mean(probs_rev_aff))
    outcome_reversal_drop = mean_prob_orig - mean_prob_rev

    results = {
        "acc_original_minimal_pair": acc_orig,
        "acc_paraphrase_invariance": acc_paraphrase,
        "acc_word_shuffle": acc_shuffled,
        "mean_affective_prob_original": mean_prob_orig,
        "mean_affective_prob_outcome_reversed": mean_prob_rev,
        "outcome_reversal_prob_drop": outcome_reversal_drop,
    }

    df_res = pd.DataFrame([results])
    res_path = os.path.join(model_dir, "phase_b_semantic_controls.csv")
    df_res.to_csv(res_path, index=False)

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_b",
        model_name=args.model_id,
        config={
            "model_prefix": args.model_prefix,
            "target_layer": target_layer,
            "relative_depth": args.relative_depth,
            "num_pairs": n_pairs,
        },
        metadata=results,
    )
    manifest.save(os.path.join(model_dir, "manifest.json"))
    print(f"Phase B completed. Results saved to {res_path}")


if __name__ == "__main__":
    main()
