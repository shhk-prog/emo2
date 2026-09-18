#!/usr/bin/env python3
"""
V1 Phase A: Probing and Geometry Script
Evaluates:
  - E1: Shared Decodability (Layer-wise Ridge/Logistic probing on external and model-behavioral targets)
  - E2: Shared Geometry (Direct Cross-Decoding, RSA/CKA, Procrustes Alignment between Reader and Self)
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedGroupKFold
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(tokenizer, text: str, task_type: str, is_instruct: bool) -> str:
    """Formats prompt identical to behavioral evaluation protocol."""
    if task_type == "reader":
        instruction = "Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."
    elif task_type == "self":
        instruction = "Read the following text and report your affective state."
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    user_content = (
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Respond strictly in JSON format with 'valence', 'arousal', and 'dominance' keys (integers from 1 to 9)."
    )

    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
            f"{instruction}\n\n"
            f"Text: {text}\n\n"
            f"Output:\n"
        )
    return prompt


@torch.no_grad()
def extract_hidden_states_batched(
    model, 
    tokenizer, 
    prompts: List[str], 
    device: str = "cuda", 
    batch_size: int = 16
) -> Dict[int, np.ndarray]:
    """
    Extracts prompt_last_token hidden states across all layers for a list of prompts.
    Returns: dict of layer_idx -> np.ndarray of shape (N, hidden_dim).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_layer_reps: Dict[int, List[np.ndarray]] = {}

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Forward Pass Batches"):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1)
        # Find index of last non-padding token for each item in batch
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)

        for layer_idx, layer_tensor in enumerate(hidden_states):
            if layer_idx not in all_layer_reps:
                all_layer_reps[layer_idx] = []
            
            # Gather last token rep: layer_tensor is (batch_size, seq_len, hidden_dim)
            batch_last_tokens = []
            for b_idx in range(len(batch_prompts)):
                last_pos = seq_lengths[b_idx].item()
                vec = layer_tensor[b_idx, last_pos, :].detach().cpu().float().numpy()
                batch_last_tokens.append(vec)
            all_layer_reps[layer_idx].append(np.array(batch_last_tokens))

    # Concatenate across batches
    final_reps = {
        layer: np.concatenate(batches, axis=0) 
        for layer, batches in all_layer_reps.items()
    }
    return final_reps


def evaluate_regression_probe(
    X: np.ndarray, 
    y: np.ndarray, 
    group_ids: np.ndarray, 
    cv: int = 5, 
    alpha: float = 1.0, 
    n_components: int = 50, 
    seed: int = 42
) -> Dict[str, float]:
    """Runs 5-fold GroupKFold Ridge regression probe."""
    if len(X) < cv or np.isnan(y).any():
        return {"r2": 0.0, "pearson_r": 0.0, "spearman_rho": 0.0, "mse": 0.0}

    # Ensure group_ids are strings to avoid TypeError during sorting/unique
    group_ids = np.array([str(g) for g in group_ids])
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < cv:
        gkf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        splits = gkf.split(X, np.digitize(y, np.histogram_bin_edges(y, bins=cv)))
    else:
        gkf = GroupKFold(n_splits=cv)
        splits = gkf.split(X, y, groups=group_ids)

    y_preds = np.zeros_like(y, dtype=float)

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_comp = min(n_components, len(X_train), X.shape[1])
        if n_comp > 1 and n_comp < X.shape[1]:
            pca = PCA(n_components=n_comp, random_state=seed)
            X_train_trans = pca.fit_transform(X_train_scaled)
            X_val_trans = pca.transform(X_val_scaled)
        else:
            X_train_trans = X_train_scaled
            X_val_trans = X_val_scaled

        clf = Ridge(alpha=alpha, random_state=seed)
        clf.fit(X_train_trans, y_train)
        y_preds[val_idx] = clf.predict(X_val_trans)

    r2 = float(r2_score(y, y_preds))
    mse = float(mean_squared_error(y, y_preds))
    pr, _ = pearsonr(y, y_preds)
    sr, _ = spearmanr(y, y_preds)

    return {
        "r2": r2 if not np.isnan(r2) else 0.0,
        "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
        "spearman_rho": float(sr) if not np.isnan(sr) else 0.0,
        "mse": mse if not np.isnan(mse) else 0.0
    }


def evaluate_classification_probe(
    X: np.ndarray, 
    y: np.ndarray, 
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5, 
    seed: int = 42
) -> Dict[str, float]:
    """
    Runs 5-fold StratifiedGroupKFold Logistic Regression probe for classification.
    Uses groups=group_ids (pair_id) to strictly prevent pair leakage.
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2 or len(X) < cv:
        return {"roc_auc": 0.5, "balanced_acc": 0.5, "f1_macro": 0.0}

    class_counts = [np.sum(y == c) for c in unique_classes]
    actual_cv = min(cv, min(class_counts))
    if actual_cv < 2:
        return {"roc_auc": 0.5, "balanced_acc": 0.5, "f1_macro": 0.0}

    if group_ids is not None:
        group_ids = np.array([str(g) for g in group_ids])
        unique_groups = np.unique(group_ids)
        if len(unique_groups) >= actual_cv:
            sgkf = StratifiedGroupKFold(n_splits=actual_cv, shuffle=True, random_state=seed)
            splits = list(sgkf.split(X, y, groups=group_ids))
        else:
            skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=seed)
            splits = list(skf.split(X, y))
    else:
        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))

    y_preds = np.zeros_like(y, dtype=int)
    is_binary = len(unique_classes) == 2
    
    if is_binary:
        y_probs = np.zeros(len(y), dtype=float)
    else:
        y_probs = np.zeros((len(y), len(unique_classes)), dtype=float)

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_scaled, y_train)
        y_preds[val_idx] = clf.predict(X_val_scaled)
        
        if is_binary:
            y_probs[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            y_probs[val_idx] = clf.predict_proba(X_val_scaled)

    bal_acc = float(balanced_accuracy_score(y, y_preds))
    f1_macro = float(f1_score(y, y_preds, average="macro"))

    try:
        if is_binary:
            auc = float(roc_auc_score(y, y_probs))
        else:
            auc = float(roc_auc_score(y, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = 0.5

    return {
        "roc_auc": auc if not np.isnan(auc) else 0.5,
        "balanced_acc": bal_acc if not np.isnan(bal_acc) else 0.0,
        "f1_macro": f1_macro if not np.isnan(f1_macro) else 0.0
    }


def evaluate_cross_decoding_and_geometry(
    H_R: np.ndarray, 
    H_S: np.ndarray, 
    y: np.ndarray, 
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates E2 (Shared Geometry) between Reader (H_R) and Self (H_S) using HELD-OUT validation:
      1. Direct Cross-Decoding: Fit Scaler & Ridge on train split, evaluate on held-out test split
      2. RSA: Evaluated on held-out test split representations
      3. Orthogonal Procrustes Alignment: SVD rotation fit strictly on train split, evaluated on test split
    Note: Threshold classifications (Shared vs Alignable) are operational classifications.
    """
    n_samples = len(y)
    if group_ids is not None:
        group_ids = np.array([str(g) for g in group_ids])
        unique_groups = np.unique(group_ids)
        if len(unique_groups) >= cv:
            splitter = GroupKFold(n_splits=cv)
            splits = list(splitter.split(H_R, y, groups=group_ids))
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
            splits = list(splitter.split(H_R))
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        splits = list(splitter.split(H_R))

    pred_r_to_s = np.zeros(n_samples, dtype=float)
    pred_s_to_r = np.zeros(n_samples, dtype=float)
    pred_within_r = np.zeros(n_samples, dtype=float)
    pred_within_s = np.zeros(n_samples, dtype=float)
    pred_aligned_s_to_r = np.zeros(n_samples, dtype=float)
    rsa_scores = []

    for train_idx, test_idx in splits:
        HR_tr, HR_te = H_R[train_idx], H_R[test_idx]
        HS_tr, HS_te = H_S[train_idx], H_S[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler_r = StandardScaler()
        scaler_s = StandardScaler()
        HR_tr_s = scaler_r.fit_transform(HR_tr)
        HR_te_s = scaler_r.transform(HR_te)
        HS_tr_s = scaler_s.fit_transform(HS_tr)
        HS_te_s = scaler_s.transform(HS_te)

        # Direct Cross-Decoding
        clf_r = Ridge(alpha=1.0, random_state=seed)
        clf_r.fit(HR_tr_s, y_tr)
        pred_within_r[test_idx] = clf_r.predict(HR_te_s)
        pred_r_to_s[test_idx] = clf_r.predict(HS_te_s)

        clf_s = Ridge(alpha=1.0, random_state=seed)
        clf_s.fit(HS_tr_s, y_tr)
        pred_within_s[test_idx] = clf_s.predict(HS_te_s)
        pred_s_to_r[test_idx] = clf_s.predict(HR_te_s)

        # Held-out Procrustes Alignment: SVD fit on train split only
        try:
            M = np.dot(HS_tr_s.T, HR_tr_s)
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = np.dot(U, Vt)
            HS_te_aligned = np.dot(HS_te_s, Q)
            pred_aligned_s_to_r[test_idx] = clf_r.predict(HS_te_aligned)
        except Exception:
            pred_aligned_s_to_r[test_idx] = pred_r_to_s[test_idx]

        # Held-out RSA on test split
        try:
            rdm_r = pdist(HR_te_s, metric="correlation")
            rdm_s = pdist(HS_te_s, metric="correlation")
            rsa_sp, _ = spearmanr(rdm_r, rdm_s)
            if not np.isnan(rsa_sp):
                rsa_scores.append(rsa_sp)
        except Exception:
            pass

    r2_within_r = float(r2_score(y, pred_within_r))
    r2_within_s = float(r2_score(y, pred_within_s))
    r2_r_to_s = float(r2_score(y, pred_r_to_s))
    r2_s_to_r = float(r2_score(y, pred_s_to_r))
    r2_aligned_transfer = float(r2_score(y, pred_aligned_s_to_r))
    rsa_score = float(np.mean(rsa_scores)) if rsa_scores else 0.0

    direct_transfer_score = max(0.0, (r2_r_to_s + r2_s_to_r) / 2.0)

    # Operational Classification
    if direct_transfer_score >= 0.3:
        pattern = "Operational: Shared Geometry"
    elif r2_aligned_transfer >= 0.3 or rsa_score >= 0.6:
        pattern = "Operational: Alignable Geometry"
    else:
        pattern = "Operational: Task-Divergent Geometry"

    return {
        "r2_within_r": r2_within_r,
        "r2_within_s": r2_within_s,
        "r2_cross_r_to_s": r2_r_to_s,
        "r2_cross_s_to_r": r2_s_to_r,
        "direct_transfer_score": direct_transfer_score,
        "rsa_correlation": rsa_score,
        "r2_aligned_transfer": r2_aligned_transfer,
        "geometry_pattern": pattern,
        "is_held_out": True
    }


def main():
    parser = argparse.ArgumentParser(description="Run V1 Phase A: E1 (Decodability) & E2 (Geometry) Probing.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID.")
    parser.add_argument("--model-prefix", type=str, default="qwen2.5_1.5b_instruct", help="Prefix for file naming.")
    parser.add_argument("--dataset", type=str, choices=["emobank", "aipsy", "both"], default="both", help="Dataset to evaluate.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for pilot test (0 = full).")
    parser.add_argument("--batch-size", type=int, default=16, help="Forward pass batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--is-instruct", action="store_true", help="Whether the model is an instruction-tuned model.")
    parser.add_argument("--out-dir", type=str, default="v1/results/derived/v1_phase_a", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = args.is_instruct or "instruct" in args.model_id.lower() or "chat" in args.model_id.lower() or "it" in args.model_id.lower()

    print(f"=== Starting V1 Phase A Probing for Model: {args.model_id} (Instruct={is_instruct}) ===")
    print(f"Device: {args.device} | Limit: {args.limit} | Output: {model_dir}")

    # Load Model and Tokenizer
    print("\n[1/4] Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    num_layers = model.config.num_hidden_layers + 1

    # =========================================================================
    # PART 1: EmoBank Probing (Regression: Human V/A & Behavioral V/A)
    # =========================================================================
    if args.dataset in ["emobank", "both"]:
        print("\n[2/4] Processing EmoBank Dataset (1,000 items)...")
        emobank_path = "v1/data/processed/stimuli_vad_3way_test1k.csv"
        df_emobank = pd.read_csv(emobank_path)

        # Merge with behavioral predictions if available
        beh_path = f"v1/results/emobank_3way_vad_test1k/{args.model_prefix}_3way_vad.csv"
        if os.path.exists(beh_path):
            print(f"Loading behavioral predictions from {beh_path}...")
            df_beh = pd.read_csv(beh_path)
            # Match wide format: r_ev, r_ea, s_ev, s_ea
            cols_to_merge = ["id"]
            for col in ["r_ev", "r_ea", "s_ev", "s_ea"]:
                if col in df_beh.columns:
                    cols_to_merge.append(col)
            
            df_emobank = df_emobank.merge(df_beh[cols_to_merge], on="id", how="left")
            rename_dict = {
                "r_ev": "beh_reader_E_V",
                "r_ea": "beh_reader_E_A",
                "s_ev": "beh_self_E_V",
                "s_ea": "beh_self_E_A"
            }
            df_emobank.rename(columns=rename_dict, inplace=True)
        else:
            print(f"Notice: Behavioral file {beh_path} not found. Secondary targets will be skipped.")

        if args.limit > 0:
            df_emobank = df_emobank.head(args.limit)

        print(f"Extracting hidden states for {len(df_emobank)} EmoBank stimuli...")
        prompts_r = [format_prompt(tokenizer, t, "reader", is_instruct) for t in df_emobank["text"]]
        prompts_s = [format_prompt(tokenizer, t, "self", is_instruct) for t in df_emobank["text"]]

        print("  Extracting Reader representations...")
        reps_r = extract_hidden_states_batched(model, tokenizer, prompts_r, device=args.device, batch_size=args.batch_size)
        print("  Extracting Self representations...")
        reps_s = extract_hidden_states_batched(model, tokenizer, prompts_s, device=args.device, batch_size=args.batch_size)

        num_layers = len(reps_r)
        print(f"Successfully extracted representations across {num_layers} layers.")

        y_human_v = df_emobank["reader_V"].values
        y_human_a = df_emobank["reader_A"].values
        group_ids = df_emobank["id"].values

        e1_emobank_records = []
        e2_emobank_records = []

        print("\nEvaluating E1 (Decodability) and E2 (Geometry) across layers for EmoBank...")
        for l in range(num_layers):
            H_R_l = reps_r[l]
            H_S_l = reps_s[l]

            res_r_v = evaluate_regression_probe(H_R_l, y_human_v, group_ids)
            res_s_v = evaluate_regression_probe(H_S_l, y_human_v, group_ids)
            res_r_a = evaluate_regression_probe(H_R_l, y_human_a, group_ids)
            res_s_a = evaluate_regression_probe(H_S_l, y_human_a, group_ids)

            rec = {
                "layer": l,
                "dataset": "EmoBank",
                "target_V_human_r2_reader": res_r_v["r2"],
                "target_V_human_r2_self": res_s_v["r2"],
                "target_V_human_pr_reader": res_r_v["pearson_r"],
                "target_V_human_pr_self": res_s_v["pearson_r"],
                "target_A_human_r2_reader": res_r_a["r2"],
                "target_A_human_r2_self": res_s_a["r2"],
                "target_A_human_pr_reader": res_r_a["pearson_r"],
                "target_A_human_pr_self": res_s_a["pearson_r"]
            }

            if "beh_reader_E_V" in df_emobank.columns and not df_emobank["beh_reader_E_V"].isna().all():
                y_beh_r_v = df_emobank["beh_reader_E_V"].values
                y_beh_s_v = df_emobank["beh_self_E_V"].values
                rec["target_V_beh_r2_reader"] = evaluate_regression_probe(H_R_l, y_beh_r_v, group_ids)["r2"]
                rec["target_V_beh_r2_self"] = evaluate_regression_probe(H_S_l, y_beh_s_v, group_ids)["r2"]

            e1_emobank_records.append(rec)

            geom_v = evaluate_cross_decoding_and_geometry(H_R_l, H_S_l, y_human_v, group_ids=group_ids)
            geom_v["layer"] = l
            geom_v["target"] = "Valence_human"
            e2_emobank_records.append(geom_v)

        df_e1_emo = pd.DataFrame(e1_emobank_records)
        df_e2_emo = pd.DataFrame(e2_emobank_records)

        df_e1_emo.to_csv(os.path.join(model_dir, "e1_emobank_decodability.csv"), index=False)
        df_e2_emo.to_csv(os.path.join(model_dir, "e2_emobank_geometry.csv"), index=False)
        print(f"Saved EmoBank probing results to {model_dir}/")

    # =========================================================================
    # PART 2: AIPsy-Affect Probing (Classification & Ordinal Intensity)
    # =========================================================================
    if args.dataset in ["aipsy", "both"]:
        print("\n[3/4] Processing AIPsy-Affect Dataset...")
        aipsy_path = "v1/data/processed/aipsy_4split_all.csv"
        df_aipsy = pd.read_csv(aipsy_path)

        if args.limit > 0:
            # Balanced sampling across splits to ensure all conditions/classes are present
            splits = df_aipsy["split"].dropna().unique()
            samples_per_split = max(1, args.limit // len(splits))
            sub_dfs = [df_aipsy[df_aipsy["split"] == s].head(samples_per_split) for s in splits]
            df_aipsy = pd.concat(sub_dfs, ignore_index=True)
            if len(df_aipsy) > args.limit:
                df_aipsy = df_aipsy.iloc[:args.limit].reset_index(drop=True)

        print(f"Extracting hidden states for {len(df_aipsy)} AIPsy stimuli...")
        prompts_r_aipsy = [format_prompt(tokenizer, t, "reader", is_instruct) for t in df_aipsy["text"]]
        prompts_s_aipsy = [format_prompt(tokenizer, t, "self", is_instruct) for t in df_aipsy["text"]]

        reps_r_aipsy = extract_hidden_states_batched(model, tokenizer, prompts_r_aipsy, device=args.device, batch_size=args.batch_size)
        reps_s_aipsy = extract_hidden_states_batched(model, tokenizer, prompts_s_aipsy, device=args.device, batch_size=args.batch_size)

        # Condition: Affective (clinical, moderate) = 1, Neutral (neutral, complex_neutral) = 0
        y_cond = np.array([1 if str(s).lower() in ["clinical", "moderate"] else 0 for s in df_aipsy["split"]])
        
        # Intensity: neutral = 0, moderate = 1, clinical = 2
        intensity_map = {"neutral": 0, "complex_neutral": 0, "moderate": 1, "clinical": 2}
        y_intensity = np.array([intensity_map.get(str(s).lower(), 0) for s in df_aipsy["split"]])

        if "pair_id" in df_aipsy.columns:
            pair_groups = np.array([
                str(p) if pd.notna(p) else f"single_{idx}" 
                for idx, p in enumerate(df_aipsy["pair_id"])
            ])
        else:
            pair_groups = np.arange(len(y_intensity), dtype=str)

        e1_aipsy_records = []
        e2_aipsy_records = []

        print("\nEvaluating E1 (Decodability with StratifiedGroupKFold) and E2 (Held-out Geometry) across layers for AIPsy...")
        for l in range(len(reps_r_aipsy)):
            H_R_l = reps_r_aipsy[l]
            H_S_l = reps_s_aipsy[l]

            res_r_cls = evaluate_classification_probe(H_R_l, y_cond, group_ids=pair_groups)
            res_s_cls = evaluate_classification_probe(H_S_l, y_cond, group_ids=pair_groups)

            res_r_ord = evaluate_regression_probe(H_R_l, y_intensity, pair_groups)
            res_s_ord = evaluate_regression_probe(H_S_l, y_intensity, pair_groups)

            rec = {
                "layer": l,
                "dataset": "AIPsy-Affect",
                "condition_auc_reader": res_r_cls["roc_auc"],
                "condition_auc_self": res_s_cls["roc_auc"],
                "condition_bal_acc_reader": res_r_cls["balanced_acc"],
                "condition_bal_acc_self": res_s_cls["balanced_acc"],
                "intensity_spearman_reader": res_r_ord["spearman_rho"],
                "intensity_spearman_self": res_s_ord["spearman_rho"]
            }
            e1_aipsy_records.append(rec)

            geom_cond = evaluate_cross_decoding_and_geometry(H_R_l, H_S_l, y_cond.astype(float), group_ids=pair_groups)
            geom_cond["layer"] = l
            geom_cond["target"] = "Condition_affective"
            e2_aipsy_records.append(geom_cond)

        df_e1_aipsy = pd.DataFrame(e1_aipsy_records)
        df_e2_aipsy = pd.DataFrame(e2_aipsy_records)

        df_e1_aipsy.to_csv(os.path.join(model_dir, "e1_aipsy_decodability.csv"), index=False)
        df_e2_aipsy.to_csv(os.path.join(model_dir, "e2_aipsy_geometry.csv"), index=False)
        print(f"Saved AIPsy probing results to {model_dir}/")

    # =========================================================================
    # PART 3: Summary Report Generation
    # =========================================================================
    print("\n[4/4] Generating Phase A Summary Report...")
    summary_path = os.path.join(model_dir, "phase_a_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# V1 Phase A Probing Summary Report: {args.model_id}\n\n")
        f.write(f"- **Model**: `{args.model_id}`\n")
        f.write(f"- **Validation Scheme**: Group-based Held-out Cross-Validation (Leakage-free)\n")
        f.write(f"- **Operational Geometry Note**: Shared/Alignable geometry classifications are operational heuristics. ")
        f.write("In models with low baseline decodability (e.g. Gemma), high RSA or Procrustes alignment must not be interpreted as evidence of shared affect-specific geometry.\n\n")
        f.write(f"- **Layers Evaluated**: {num_layers}\n")
        f.write(f"- **Dataset Evaluated**: `{args.dataset}`\n\n")

        if args.dataset in ["emobank", "both"]:
            peak_r_v = int(df_e1_emo["target_V_human_r2_reader"].idxmax())
            peak_s_v = int(df_e1_emo["target_V_human_r2_self"].idxmax())
            f.write("## EmoBank (E1: Shared Decodability & E2: Shared Geometry)\n\n")
            f.write(f"- **Reader V Peak Layer**: Layer {peak_r_v} ($R^2 = {df_e1_emo.loc[peak_r_v, 'target_V_human_r2_reader']:.3f}$)\n")
            f.write(f"- **Self V Peak Layer**: Layer {peak_s_v} ($R^2 = {df_e1_emo.loc[peak_s_v, 'target_V_human_r2_self']:.3f}$)\n")
            f.write(f"- **Peak Layer Displacement**: $|l^*_R - l^*_S| = {abs(peak_r_v - peak_s_v)}$ layers\n\n")
            
            geom_counts = df_e2_emo["geometry_pattern"].value_counts().to_dict()
            f.write("### Geometry Patterns Across Layers:\n")
            for pat, cnt in geom_counts.items():
                f.write(f"- **{pat}**: {cnt} / {num_layers} layers ({cnt/num_layers*100:.1f}%)\n")
            f.write("\n")

        if args.dataset in ["aipsy", "both"]:
            peak_r_auc = int(df_e1_aipsy["condition_auc_reader"].idxmax())
            peak_s_auc = int(df_e1_aipsy["condition_auc_self"].idxmax())
            f.write("## AIPsy-Affect (E1: Shared Decodability & E2: Shared Geometry)\n\n")
            f.write(f"- **Reader Condition Peak AUC**: Layer {peak_r_auc} ($\text{{AUC}} = {df_e1_aipsy.loc[peak_r_auc, 'condition_auc_reader']:.3f}$)\n")
            f.write(f"- **Self Condition Peak AUC**: Layer {peak_s_auc} ($\text{{AUC}} = {df_e1_aipsy.loc[peak_s_auc, 'condition_auc_self']:.3f}$)\n")
            f.write(f"- **Peak Layer Displacement**: $|l^*_R - l^*_S| = {abs(peak_r_auc - peak_s_auc)}$ layers\n\n")

    print(f"Summary report written to {summary_path}")
    print("=== Phase A Probing Completed Successfully! ===")


if __name__ == "__main__":
    main()
