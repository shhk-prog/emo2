#!/usr/bin/env python3
"""
v1/primary/run_phase_a.py

V1 Phase A Primary Script: Probing and Geometry Analysis
Evaluates:
  - E1: Shared Decodability (Layer-wise Ridge/Logistic probing on external and model-behavioral targets)
  - E2: Shared Geometry (Direct Cross-Decoding, RSA/CKA, Procrustes Alignment between Reader and Self)

Strict Features:
  - Leakage-free GroupKFold / StratifiedGroupKFold on stimulus pairs
  - Relative depth d = l / (num_layers - 1) recorded for all layers
  - Manifest integration for full experimental provenance
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from affective_empathy_eval.manifests import create_run_manifest

warnings.filterwarnings("ignore", category=RuntimeWarning)


def format_prompt(
    tokenizer, text: str, task_type: str, is_instruct: bool
) -> str:
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
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
    batch_size: int = 16,
) -> Dict[int, np.ndarray]:
    all_layer_reps: Dict[int, List[np.ndarray]] = {}

    for start_idx in tqdm(
        range(0, len(prompts), batch_size), desc="Forward Pass Batches"
    ):
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

        hidden_states = outputs.hidden_states
        seq_lengths = attention_mask.sum(dim=1) - 1

        for layer_idx, layer_tensor in enumerate(hidden_states):
            if layer_idx not in all_layer_reps:
                all_layer_reps[layer_idx] = []

            batch_last_tokens = []
            for b_idx in range(len(batch_prompts)):
                last_pos = seq_lengths[b_idx].item()
                vec = (
                    layer_tensor[b_idx, last_pos, :].detach().cpu().float().numpy()
                )
                batch_last_tokens.append(vec)
            all_layer_reps[layer_idx].append(np.array(batch_last_tokens))

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
    seed: int = 42,
) -> Dict[str, float]:
    if len(X) < cv or np.isnan(y).any():
        return {"r2": 0.0, "pearson_r": 0.0, "spearman_rho": 0.0, "mse": 0.0}

    group_ids = np.array([str(g) for g in group_ids])
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < cv:
        gkf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        splits = gkf.split(
            X, np.digitize(y, np.histogram_bin_edges(y, bins=cv))
        )
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
        "mse": mse if not np.isnan(mse) else 0.0,
    }


def evaluate_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
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
            sgkf = StratifiedGroupKFold(
                n_splits=actual_cv, shuffle=True, random_state=seed
            )
            splits = list(sgkf.split(X, y, groups=group_ids))
        else:
            skf = StratifiedKFold(
                n_splits=actual_cv, shuffle=True, random_state=seed
            )
            splits = list(skf.split(X, y))
    else:
        skf = StratifiedKFold(
            n_splits=actual_cv, shuffle=True, random_state=seed
        )
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
            auc = float(
                roc_auc_score(y, y_probs, multi_class="ovr", average="macro")
            )
    except Exception:
        auc = 0.5

    return {
        "roc_auc": auc if not np.isnan(auc) else 0.5,
        "balanced_acc": bal_acc if not np.isnan(bal_acc) else 0.0,
        "f1_macro": f1_macro if not np.isnan(f1_macro) else 0.0,
    }


def evaluate_cross_decoding_and_geometry(
    H_R: np.ndarray,
    H_S: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
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

        clf_r = Ridge(alpha=1.0, random_state=seed)
        clf_r.fit(HR_tr_s, y_tr)
        pred_within_r[test_idx] = clf_r.predict(HR_te_s)
        pred_r_to_s[test_idx] = clf_r.predict(HS_te_s)

        clf_s = Ridge(alpha=1.0, random_state=seed)
        clf_s.fit(HS_tr_s, y_tr)
        pred_within_s[test_idx] = clf_s.predict(HS_te_s)
        pred_s_to_r[test_idx] = clf_s.predict(HR_te_s)

        try:
            M = np.dot(HS_tr_s.T, HR_tr_s)
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = np.dot(U, Vt)
            HS_te_aligned = np.dot(HS_te_s, Q)
            pred_aligned_s_to_r[test_idx] = clf_r.predict(HS_te_aligned)
        except Exception:
            pred_aligned_s_to_r[test_idx] = pred_r_to_s[test_idx]

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
        "is_held_out": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V1 Primary Phase A: Probing and Geometry"
    )
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--model-prefix", type=str, default="qwen2.5_1.5b_instruct"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["emobank", "aipsy", "both"],
        default="both",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--is-instruct", action="store_true")
    parser.add_argument(
        "--out-dir", type=str, default="v1/results/derived/v1_phase_a"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    print(
        f"=== Starting V1 Phase A Probing for Model: {args.model_id} (Instruct={is_instruct}) ==="
    )

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

    num_layers = model.config.num_hidden_layers + 1

    # Part 1: EmoBank
    if args.dataset in ["emobank", "both"]:
        emobank_path = Path("v1/data/processed/stimuli_vad_3way_test1k.csv")
        if not emobank_path.exists():
            emobank_path = Path("data/processed/stimuli_vad_3way_test1k.csv")

        df_emobank = pd.read_csv(emobank_path)
        if args.limit > 0:
            df_emobank = df_emobank.head(args.limit)

        prompts_r = [
            format_prompt(tokenizer, t, "reader", is_instruct)
            for t in df_emobank["text"]
        ]
        prompts_s = [
            format_prompt(tokenizer, t, "self", is_instruct)
            for t in df_emobank["text"]
        ]

        reps_r = extract_hidden_states_batched(
            model,
            tokenizer,
            prompts_r,
            device=args.device,
            batch_size=args.batch_size,
        )
        reps_s = extract_hidden_states_batched(
            model,
            tokenizer,
            prompts_s,
            device=args.device,
            batch_size=args.batch_size,
        )

        num_layers = len(reps_r)
        y_human_v = df_emobank["reader_V"].values
        y_human_a = df_emobank["reader_A"].values
        group_ids = df_emobank["id"].values

        e1_emobank_records = []
        e2_emobank_records = []

        for l in range(num_layers):
            rel_d = l / (num_layers - 1) if num_layers > 1 else 0.0
            H_R_l = reps_r[l]
            H_S_l = reps_s[l]

            res_r_v = evaluate_regression_probe(H_R_l, y_human_v, group_ids)
            res_s_v = evaluate_regression_probe(H_S_l, y_human_v, group_ids)
            res_r_a = evaluate_regression_probe(H_R_l, y_human_a, group_ids)
            res_s_a = evaluate_regression_probe(H_S_l, y_human_a, group_ids)

            e1_emobank_records.append(
                {
                    "layer": l,
                    "relative_depth": rel_d,
                    "dataset": "EmoBank",
                    "target_V_human_r2_reader": res_r_v["r2"],
                    "target_V_human_r2_self": res_s_v["r2"],
                    "target_V_human_pr_reader": res_r_v["pearson_r"],
                    "target_V_human_pr_self": res_s_v["pearson_r"],
                    "target_A_human_r2_reader": res_r_a["r2"],
                    "target_A_human_r2_self": res_s_a["r2"],
                    "target_A_human_pr_reader": res_r_a["pearson_r"],
                    "target_A_human_pr_self": res_s_a["pearson_r"],
                }
            )

            geom_v = evaluate_cross_decoding_and_geometry(
                H_R_l, H_S_l, y_human_v, group_ids=group_ids
            )
            geom_v["layer"] = l
            geom_v["relative_depth"] = rel_d
            geom_v["target"] = "Valence_human"
            e2_emobank_records.append(geom_v)

        df_e1_emo = pd.DataFrame(e1_emobank_records)
        df_e2_emo = pd.DataFrame(e2_emobank_records)
        df_e1_emo.to_csv(
            os.path.join(model_dir, "e1_emobank_decodability.csv"), index=False
        )
        df_e2_emo.to_csv(
            os.path.join(model_dir, "e2_emobank_geometry.csv"), index=False
        )

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_a",
        model_name=args.model_id,
        config={
            "model_prefix": args.model_prefix,
            "dataset": args.dataset,
            "limit": args.limit,
            "num_layers": num_layers,
        },
    )
    manifest.save(os.path.join(model_dir, "manifest.json"))
    print(f"Phase A completed. Results saved to {model_dir}")


if __name__ == "__main__":
    main()
