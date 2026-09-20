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
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import warnings
import yaml
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
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.geometry import get_block_hidden_state
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_single_model_from_args,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)


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
            add_special_tokens=False,
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
        num_blocks = len(hidden_states) - 1

        for block_idx in range(num_blocks):
            layer_tensor = get_block_hidden_state(hidden_states, block_idx)
            if block_idx not in all_layer_reps:
                all_layer_reps[block_idx] = []


            batch_last_tokens = []
            for b_idx in range(len(batch_prompts)):
                last_pos = seq_lengths[b_idx].item()
                vec = (
                    layer_tensor[b_idx, last_pos, :].detach().cpu().float().numpy()
                )
                batch_last_tokens.append(vec)
            all_layer_reps[block_idx].append(np.array(batch_last_tokens))

    final_reps = {}
    for layer, batches in all_layer_reps.items():
        arr = np.concatenate(batches, axis=0)
        if not np.all(np.isfinite(arr)):
            bad_count = int(arr.size - np.isfinite(arr).sum())
            raise FloatingPointError(
                f"Non-finite hidden states detected: {bad_count}"
            )
        max_abs = float(np.max(np.abs(arr)))
        logger.info(
            "Hidden-state max abs = %.6f",
            max_abs,
        )
        arr = arr.astype(np.float32)
        final_reps[layer] = arr
    return final_reps



def evaluate_regression_probe(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    cv: int = 5,
    alpha: float = 1.0,
    n_components: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    # Item 4: 非有限 hidden state を暗黙置換せず例外化
    if not np.all(np.isfinite(X)):
        bad_count = int(np.size(X) - np.isfinite(X).sum())
        raise FloatingPointError(f"Non-finite hidden states detected in regression probe: {bad_count}")

    X = X.astype(np.float32)

    # Item 2: 測定不能を 0 ではなく NaN で保持
    if len(X) < cv or np.isnan(y).any():
        return {
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "mse": float("nan"),
            "status": "failed",
            "failure_reason": "Insufficient samples or NaN in y",
            "n_valid_folds": 0,
        }

    group_ids = np.array([str(g) for g in group_ids])
    unique_groups = np.unique(group_ids)
    n_splits = min(cv, len(unique_groups))
    if n_splits < 2:
        return {
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "mse": float("nan"),
            "status": "failed",
            "failure_reason": f"Insufficient unique groups (n={len(unique_groups)})",
            "n_valid_folds": 0,
        }
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=group_ids))

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
    pr, _ = pearsonr(y, y_preds) if np.std(y_preds) > 0 else (float("nan"), float("nan"))
    sr, _ = spearmanr(y, y_preds) if np.std(y_preds) > 0 else (float("nan"), float("nan"))

    return {
        "r2": r2,
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "mse": mse,
        "status": "success",
        "failure_reason": "none",
        "n_valid_folds": len(splits),
    }


def evaluate_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    # Item 4: 非有限 hidden state を暗黙置換せず例外化
    if not np.all(np.isfinite(X)):
        bad_count = int(np.size(X) - np.isfinite(X).sum())
        raise FloatingPointError(f"Non-finite hidden states detected in classification probe: {bad_count}")

    X = X.astype(np.float32)

    if group_ids is not None:
        group_ids = np.array([str(g) for g in group_ids])
        unique_groups = np.unique(group_ids)
        if len(unique_groups) < 2:
            return {
                "roc_auc": float("nan"),
                "balanced_acc": float("nan"),
                "f1_macro": float("nan"),
                "status": "failed",
                "failure_reason": "insufficient_unique_groups",
                "n_valid_folds": 0,
            }

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    if num_classes < 2:
        return {
            "roc_auc": float("nan"),
            "balanced_acc": float("nan"),
            "f1_macro": float("nan"),
            "status": "failed",
            "failure_reason": "single_class_only",
            "n_valid_folds": 0,
        }

    if group_ids is not None and len(unique_groups) >= cv:
        n_splits = cv
        actual_cv = cv
    elif group_ids is not None and len(unique_groups) >= 2:
        n_splits = len(unique_groups)
        actual_cv = n_splits
    else:
        min_class_count = np.min(np.bincount(y_enc))
        actual_cv = min(cv, min_class_count)
        if actual_cv < 2:
            return {
                "roc_auc": float("nan"),
                "balanced_acc": float("nan"),
                "f1_macro": float("nan"),
                "status": "failed",
                "failure_reason": "insufficient_samples_or_classes",
                "n_valid_folds": 0,
            }

    if group_ids is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        try:
            splits = list(sgkf.split(X, y_enc, groups=group_ids))
        except ValueError:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X, y_enc, groups=group_ids))
    else:
        skf = StratifiedKFold(
            n_splits=actual_cv, shuffle=True, random_state=seed
        )
        splits = list(skf.split(X, y_enc))

    y_preds = np.zeros(len(y_enc), dtype=int)
    is_binary = num_classes == 2

    if is_binary:
        y_probs = np.zeros(len(y_enc), dtype=float)
    else:
        y_probs = np.zeros((len(y_enc), num_classes), dtype=float)

    # Item 36: train fold が単一 class の場合を明示処理
    valid_folds = 0
    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        if np.unique(y_train).size < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train_scaled, y_train)
        y_preds[val_idx] = clf.predict(X_val_scaled)

        probs = clf.predict_proba(X_val_scaled)
        if is_binary:
            y_probs[val_idx] = probs[:, 1]
        else:
            for c_idx, c in enumerate(clf.classes_):
                y_probs[val_idx, c] = probs[:, c_idx]
        valid_folds += 1

    if valid_folds < 2:
        return {
            "roc_auc": float("nan"),
            "balanced_acc": float("nan"),
            "f1_macro": float("nan"),
            "status": "failed",
            "failure_reason": f"insufficient_valid_folds_with_multiple_classes (valid={valid_folds})",
            "n_valid_folds": valid_folds,
        }

    bal_acc = float(balanced_accuracy_score(y_enc, y_preds))
    f1_macro = float(f1_score(y_enc, y_preds, average="macro"))

    try:
        if is_binary:
            auc = float(roc_auc_score(y_enc, y_probs))
        else:
            auc = float(
                roc_auc_score(y_enc, y_probs, multi_class="ovr", average="macro", labels=np.arange(num_classes))
            )
        status = "success"
        failure_reason = "none"
    except Exception as e:
        auc = float("nan")
        status = "partial_failure"
        failure_reason = str(e)

    return {
        "roc_auc": auc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_macro,
        "status": status,
        "failure_reason": failure_reason,
        "n_valid_folds": valid_folds,
    }


def evaluate_cross_decoding_and_geometry(
    H_R: np.ndarray,
    H_S: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    for name, H in {
        "Reader": H_R,
        "Self": H_S,
    }.items():
        if not np.all(np.isfinite(H)):
            raise FloatingPointError(
                f"{name} hidden states contain non-finite values."
            )
    H_R = H_R.astype(np.float32)
    H_S = H_S.astype(np.float32)
    n_samples = len(y)
    if group_ids is not None:
        group_ids = np.array([str(g) for g in group_ids])
        unique_groups = np.unique(group_ids)
        n_splits = min(cv, len(unique_groups))
        if n_splits < 2:
            return {
                "r2_within_r": float("nan"),
                "r2_within_s": float("nan"),
                "r2_cross_r_to_s": float("nan"),
                "r2_cross_s_to_r": float("nan"),
                "direct_transfer_score": float("nan"),
                "rsa_correlation": float("nan"),
                "r2_aligned_transfer": float("nan"),
                "geometry_pattern": "insufficient_groups",
                "is_held_out": True,
            }
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(H_R, y, groups=group_ids))
    else:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        splits = list(splitter.split(H_R))

    pred_r_to_s = np.zeros(n_samples, dtype=float)
    pred_s_to_r = np.zeros(n_samples, dtype=float)
    pred_r_to_s_task_scaled = np.zeros(n_samples, dtype=float)
    pred_s_to_r_task_scaled = np.zeros(n_samples, dtype=float)
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
        # Primary: Reader-fit scaler applied across both tasks (Reader and Self)
        pred_r_to_s[test_idx] = clf_r.predict(scaler_r.transform(HS_te))
        # Secondary: task-specific scaler
        pred_r_to_s_task_scaled[test_idx] = clf_r.predict(HS_te_s)

        clf_s = Ridge(alpha=1.0, random_state=seed)
        clf_s.fit(HS_tr_s, y_tr)
        pred_within_s[test_idx] = clf_s.predict(HS_te_s)
        # Primary: Self-fit scaler applied across both tasks (Self and Reader)
        pred_s_to_r[test_idx] = clf_s.predict(scaler_s.transform(HR_te))
        # Secondary: task-specific scaler
        pred_s_to_r_task_scaled[test_idx] = clf_s.predict(HR_te_s)

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
    r2_r_to_s_task_scaled = float(r2_score(y, pred_r_to_s_task_scaled))
    r2_s_to_r_task_scaled = float(r2_score(y, pred_s_to_r_task_scaled))
    r2_aligned_transfer = float(r2_score(y, pred_aligned_s_to_r))
    rsa_score = float(np.mean(rsa_scores)) if rsa_scores else 0.0

    direct_transfer_score_raw = (r2_r_to_s + r2_s_to_r) / 2.0
    direct_transfer_score_clipped = max(0.0, direct_transfer_score_raw)

    if direct_transfer_score_raw >= 0.3:
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
        "r2_cross_r_to_s_task_scaled": r2_r_to_s_task_scaled,
        "r2_cross_s_to_r_task_scaled": r2_s_to_r_task_scaled,
        "direct_transfer_score": direct_transfer_score_raw,
        "direct_transfer_score_raw": direct_transfer_score_raw,
        "direct_transfer_score_clipped": direct_transfer_score_clipped,
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
        "--config",
        type=str,
        default="configs/v1_experiments.yaml",
        help="Path to V1 experiments config file",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model ID. Required unless --family is set.",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help="Output prefix. Derived from --family or --model-id if omitted.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock dry-run mode for quick pipeline smoke testing",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Specific HuggingFace model git commit SHA or branch",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional unique run_id for results organization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output files already exist",
    )
    add_model_selection_args(parser)
    args = parser.parse_args()
    args.model_id, args.model_prefix = resolve_single_model_from_args(args)

    # Production safety valve: ensure fixed model revision is resolved
    if args.model_revision is None and not getattr(args, "dry_run", False):
        from affective_empathy_eval.models.registry import get_registry
        registry = get_registry()
        fam_cfg = registry.get_family_by_model_id(args.model_id)
        if fam_cfg:
            for spec in (fam_cfg.base_model, fam_cfg.instruct_model):
                if spec.model_id == args.model_id and spec.revision:
                    args.model_revision = spec.revision
                    break
        if not args.model_revision:
            raise ValueError(
                f"Production run requires explicit fixed model revision for '{args.model_id}', but none was provided or resolved from registry."
            )

    # Item 29: configs/v1_experiments.yaml の Phase A 設定読み込み
    cfg_path = Path(args.config)
    phase_a_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            v1_full_cfg = yaml.safe_load(f) or {}
            phase_a_cfg = v1_full_cfg.get("phase_a", {})

    phase_a_seed = int(phase_a_cfg.get("seed", 42))
    phase_a_cv = int(phase_a_cfg.get("cv_folds", 5))
    phase_a_alpha = float(phase_a_cfg.get("ridge_alpha", 1.0))

    if args.dry_run:
        args.out_dir = os.path.join(args.out_dir, "dry_run")

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    manifest_path = os.path.join(model_dir, "manifest.json")
    required_outputs = []
    dataset_paths = []
    if args.dataset in {"emobank", "both"}:
        required_outputs.extend([
            os.path.join(model_dir, "e1_emobank_decodability.csv"),
            os.path.join(model_dir, "e2_emobank_geometry.csv"),
        ])
        p_emo = Path("v1/data/processed/stimuli_vad_3way_test1k.csv")
        if not p_emo.exists():
            p_emo = Path("data/processed/stimuli_vad_3way_test1k.csv")
        if p_emo.exists():
            dataset_paths.append(str(p_emo))
    if args.dataset in {"aipsy", "both"}:
        required_outputs.extend([
            os.path.join(model_dir, "e1_aipsy_classification.csv"),
            os.path.join(model_dir, "e1_aipsy_intensity.csv"),
            os.path.join(model_dir, "e1_aipsy_emotion_secondary.csv"),
        ])
        p_aip = Path("v1/data/processed/aipsy_4split_all.csv")
        if not p_aip.exists():
            p_aip = Path("data/processed/aipsy_4split_all.csv")
        if p_aip.exists():
            dataset_paths.append(str(p_aip))

    from affective_empathy_eval.manifests import (
        is_manifest_matching,
        compute_string_or_dict_hash,
        compute_file_hash,
    )
    from affective_empathy_eval.io import is_experiment_completed

    dataset_hash = (
        compute_string_or_dict_hash([compute_file_hash(Path(dp)) for dp in dataset_paths])
        if dataset_paths
        else "unknown"
    )

    manifest_config = {
        "model_prefix": args.model_prefix,
        "model_id": args.model_id,
        "model_revision": args.model_revision or "main",
        "dataset": args.dataset,
        "dataset_hash": dataset_hash,
        "limit": args.limit,
        "seed": phase_a_seed,
        "cv_folds": phase_a_cv,
        "ridge_alpha": phase_a_alpha,
    }
    expected_cfg_hash = compute_string_or_dict_hash(manifest_config)

    e1_json_path = os.path.join(model_dir, f"v1_e1_decodability_{args.model_prefix}.json")
    if not args.force and not args.dry_run and os.path.exists(manifest_path) and all(os.path.exists(p) for p in required_outputs):
        if is_manifest_matching(
            manifest_path=manifest_path,
            expected_model_name=args.model_id,
            expected_config_hash=expected_cfg_hash,
            expected_dataset_hash=dataset_hash,
            expected_model_revision=args.model_revision,
            expected_dry_run=False,
        ):
            try:
                valid_all = True
                for p in required_outputs:
                    df_check = pd.read_csv(p)
                    if len(df_check) == 0:
                        valid_all = False
                        break
                if os.path.exists(e1_json_path) and not is_experiment_completed(e1_json_path, force=args.force):
                    valid_all = False

                if valid_all:
                    print(
                        f"[SKIP] Validated Phase A results matching manifest found in {model_dir} for dataset '{args.dataset}'. "
                        f"Skipping model loading & probing for {args.model_prefix}. Use --force to rerun."
                    )
                    return
            except Exception as e:
                print(f"Warning: Corrupt existing Phase A results in {model_dir} ({e}). Rerunning.")

    if args.dry_run:
        print(f"[DRY-RUN] V1 Phase A Probing for Model: {args.model_id} (Instruct={is_instruct})")
        # Generate dummy outputs for downstream testing
        dummy_layers = 4
        e1_records = [
            {
                "layer": l,
                "relative_depth": l / (dummy_layers - 1),
                "dataset": "EmoBank",
                "target_V_human_r2_reader": 0.5,
                "target_V_human_r2_self": 0.4,
                "target_V_human_pr_reader": 0.7,
                "target_V_human_pr_self": 0.6,
                "target_A_human_r2_reader": 0.3,
                "target_A_human_r2_self": 0.2,
                "target_A_human_pr_reader": 0.5,
                "target_A_human_pr_self": 0.4,
            }
            for l in range(dummy_layers)
        ]
        from affective_empathy_eval.io import save_experiment_result
        pd.DataFrame(e1_records).to_csv(
            os.path.join(model_dir, "v1_e1_emobank_decodability.csv"), index=False
        )
        pd.DataFrame(e1_records).to_csv(
            os.path.join(model_dir, "e1_emobank_decodability.csv"), index=False
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e1_decodability_{args.model_prefix}.json"),
            {"emobank": e1_records},
            stage="v1",
            experiment_id="v1_e1_decodability",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix, "is_instruct": is_instruct},
        )

        e2_cd_records = [
            {"layer": l, "relative_depth": l / (dummy_layers - 1), "target": "Valence_human", "direct_transfer_score": 0.5, "direct_transfer_score_raw": 0.5, "direct_transfer_score_clipped": 0.5}
            for l in range(dummy_layers)
        ]
        e2_rsa_records = [
            {"layer": l, "relative_depth": l / (dummy_layers - 1), "target": "Valence_human", "rsa_correlation": 0.7}
            for l in range(dummy_layers)
        ]
        e2_align_records = [
            {"layer": l, "relative_depth": l / (dummy_layers - 1), "target": "Valence_human", "r2_aligned_transfer": 0.55, "geometry_pattern": "Operational: Shared Geometry"}
            for l in range(dummy_layers)
        ]
        e2_records = [
            {**e2_cd_records[l], **e2_rsa_records[l], **e2_align_records[l]}
            for l in range(dummy_layers)
        ]
        pd.DataFrame(e2_records).to_csv(
            os.path.join(model_dir, "e2_emobank_geometry.csv"), index=False
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_cross_decoding_{args.model_prefix}.json"),
            {"cross_decoding": e2_cd_records},
            stage="v1",
            experiment_id="v1_e2_cross_decoding",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_rsa_{args.model_prefix}.json"),
            {"rsa": e2_rsa_records},
            stage="v1",
            experiment_id="v1_e2_rsa",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_alignment_{args.model_prefix}.json"),
            {"alignment": e2_align_records},
            stage="v1",
            experiment_id="v1_e2_alignment",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )

        aipsy_records = [
            {
                "layer": l,
                "relative_depth": l / (dummy_layers - 1),
                "dataset": "AIPsy",
                "target": "clinical_vs_matched_neutral",
                "reader_balanced_acc": 0.85,
                "self_balanced_acc": 0.82,
                "reader_f1_macro": 0.84,
                "self_f1_macro": 0.81,
                "reader_roc_auc": 0.92,
                "self_roc_auc": 0.89,
            }
            for l in range(dummy_layers)
        ]
        pd.DataFrame(aipsy_records).to_csv(
            os.path.join(model_dir, "v1_e1_aipsy_classification.csv"), index=False
        )
        pd.DataFrame(aipsy_records).to_csv(
            os.path.join(model_dir, "e1_aipsy_classification.csv"), index=False
        )
        aipsy_intensity_records = [
            {
                "layer": l,
                "relative_depth": l / (dummy_layers - 1),
                "dataset": "AIPsy",
                "target": "intensity_none_moderate_peak",
                "reader_spearman_rho": 0.70,
                "self_spearman_rho": 0.65,
                "reader_pearson_r": 0.68,
                "self_pearson_r": 0.62,
                "reader_r2": 0.45,
                "self_r2": 0.38,
            }
            for l in range(dummy_layers)
        ]
        pd.DataFrame(aipsy_intensity_records).to_csv(
            os.path.join(model_dir, "v1_e1_aipsy_intensity.csv"), index=False
        )
        pd.DataFrame(aipsy_intensity_records).to_csv(
            os.path.join(model_dir, "e1_aipsy_intensity.csv"), index=False
        )
        aipsy_sec_records = [
            {
                "layer": l,
                "relative_depth": l / (dummy_layers - 1),
                "dataset": "AIPsy",
                "target": "emotion_category_secondary",
                "reader_balanced_acc": 0.65,
                "self_balanced_acc": 0.60,
                "reader_f1_macro": 0.62,
                "self_f1_macro": 0.58,
                "reader_roc_auc": 0.80,
                "self_roc_auc": 0.76,
            }
            for l in range(dummy_layers)
        ]
        pd.DataFrame(aipsy_sec_records).to_csv(
            os.path.join(model_dir, "v1_e1_aipsy_emotion_secondary.csv"), index=False
        )
        pd.DataFrame(aipsy_sec_records).to_csv(
            os.path.join(model_dir, "e1_aipsy_emotion_secondary.csv"), index=False
        )
        dry_cfg = dict(manifest_config)
        dry_cfg["dry_run"] = True
        manifest = create_run_manifest(
            run_type="v1_phase_a",
            model_name=args.model_id,
            model_revision=args.model_revision or "main",
            config=dry_cfg,
            dataset_path=dataset_paths[0] if dataset_paths else None,
            dataset_hash=dataset_hash,
            candidate_space="N/A",
            measurement_space="prompt_end_hidden_state",
            seed=phase_a_seed,
            intervention_version="none",
            run_id=args.run_id,
            dry_run=True,
        )
        manifest.save(os.path.join(model_dir, "manifest.json"))
        print(f"[DRY-RUN] Completed Phase A mock output in {model_dir}")
        return

    print(
        f"=== Starting V1 Phase A Probing for Model: {args.model_id} (Instruct={is_instruct}, revision={args.model_revision}) ==="
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    actual_torch_dtype = (
        torch.bfloat16
        if is_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if is_cuda else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=actual_torch_dtype,
        device_map=args.device if is_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

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

            geom_a = evaluate_cross_decoding_and_geometry(
                H_R_l, H_S_l, y_human_a, group_ids=group_ids
            )
            geom_a["layer"] = l
            geom_a["relative_depth"] = rel_d
            geom_a["target"] = "Arousal_human"
            e2_emobank_records.append(geom_a)

            # 逐次保存 (レイヤー完了ごと)
            pd.DataFrame(e1_emobank_records).to_csv(
                os.path.join(model_dir, "v1_e1_emobank_decodability.csv"), index=False
            )
            pd.DataFrame(e1_emobank_records).to_csv(
                os.path.join(model_dir, "e1_emobank_decodability.csv"), index=False
            )
            pd.DataFrame(e2_emobank_records).to_csv(
                os.path.join(model_dir, "e2_emobank_geometry.csv"), index=False
            )

        from affective_empathy_eval.io import save_experiment_result
        save_experiment_result(
            os.path.join(model_dir, f"v1_e1_decodability_{args.model_prefix}.json"),
            {"emobank": e1_emobank_records},
            stage="v1",
            experiment_id="v1_e1_decodability",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix, "is_instruct": is_instruct},
        )

        e2_cd_records = [
            {"layer": r["layer"], "relative_depth": r["relative_depth"], "target": r["target"], "direct_transfer_score": r.get("direct_transfer_score"), "direct_transfer_score_raw": r.get("direct_transfer_score_raw"), "direct_transfer_score_clipped": r.get("direct_transfer_score_clipped")}
            for r in e2_emobank_records
        ]
        e2_rsa_records = [
            {"layer": r["layer"], "relative_depth": r["relative_depth"], "target": r["target"], "rsa_correlation": r.get("rsa_correlation")}
            for r in e2_emobank_records
        ]
        e2_align_records = [
            {"layer": r["layer"], "relative_depth": r["relative_depth"], "target": r["target"], "r2_aligned_transfer": r.get("r2_aligned_transfer"), "geometry_pattern": r.get("geometry_pattern")}
            for r in e2_emobank_records
        ]
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_cross_decoding_{args.model_prefix}.json"),
            {"cross_decoding": e2_cd_records},
            stage="v1",
            experiment_id="v1_e2_cross_decoding",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_rsa_{args.model_prefix}.json"),
            {"rsa": e2_rsa_records},
            stage="v1",
            experiment_id="v1_e2_rsa",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )
        save_experiment_result(
            os.path.join(model_dir, f"v1_e2_alignment_{args.model_prefix}.json"),
            {"alignment": e2_align_records},
            stage="v1",
            experiment_id="v1_e2_alignment",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix},
        )

    # Part 2: AIPsy Probing (Primary: Clinical vs Neutral; Intensity; Secondary: Emotion Category)
    if args.dataset in ["aipsy", "both"]:
        aipsy_path = Path("v1/data/processed/aipsy_4split_all.csv")
        if not aipsy_path.exists():
            aipsy_path = Path("data/processed/aipsy_4split_all.csv")
        if not aipsy_path.exists():
            aipsy_path = Path("data/raw/aipsy/aipsy_split.csv")

        if aipsy_path.exists():
            print(f"Loading AIPsy stimuli for probing from {aipsy_path}...")
            df_aipsy = pd.read_csv(aipsy_path)
            if args.limit > 0:
                df_aipsy = df_aipsy.head(args.limit)

            # Extract representations for all loaded AIPsy texts once
            prompts_r_aip = [
                format_prompt(tokenizer, t, "reader", is_instruct)
                for t in df_aipsy["text"]
            ]
            prompts_s_aip = [
                format_prompt(tokenizer, t, "self", is_instruct)
                for t in df_aipsy["text"]
            ]
            reps_r_aip = extract_hidden_states_batched(
                model, tokenizer, prompts_r_aip, device=args.device, batch_size=args.batch_size
            )
            reps_s_aip = extract_hidden_states_batched(
                model, tokenizer, prompts_s_aip, device=args.device, batch_size=args.batch_size
            )
            num_aip_layers = len(reps_r_aip)

            # 2.1 Primary Analysis: Clinical vs Matched Neutral (Group on pair_id)
            if "split" in df_aipsy.columns and "pair_id" in df_aipsy.columns:
                mask_prim = df_aipsy["split"].isin(["clinical", "neutral"])
                df_prim = df_aipsy[mask_prim].copy().reset_index()
                if len(df_prim) >= 4 and df_prim["split"].nunique() >= 2:
                    print(f"Running Primary AIPsy Analysis (clinical vs matched neutral, n={len(df_prim)})...")
                    y_prim = (df_prim["split"] == "clinical").astype(int).values
                    grp_prim = df_prim["pair_id"].values
                    idx_prim = df_prim["index"].values

                    e1_aipsy_records = []
                    for l in range(num_aip_layers):
                        rel_d = l / (num_aip_layers - 1) if num_aip_layers > 1 else 0.0
                        res_r_cls = evaluate_classification_probe(
                            reps_r_aip[l][idx_prim], y_prim, group_ids=grp_prim
                        )
                        res_s_cls = evaluate_classification_probe(
                            reps_s_aip[l][idx_prim], y_prim, group_ids=grp_prim
                        )
                        e1_aipsy_records.append(
                            {
                                "layer": l,
                                "relative_depth": rel_d,
                                "dataset": "AIPsy",
                                "target": "clinical_vs_matched_neutral",
                                "reader_balanced_acc": res_r_cls["balanced_acc"],
                                "self_balanced_acc": res_s_cls["balanced_acc"],
                                "reader_f1_macro": res_r_cls["f1_macro"],
                                "self_f1_macro": res_s_cls["f1_macro"],
                                "reader_roc_auc": res_r_cls["roc_auc"],
                                "self_roc_auc": res_s_cls["roc_auc"],
                            }
                        )
                    df_e1_aipsy = pd.DataFrame(e1_aipsy_records)
                    df_e1_aipsy.to_csv(
                        os.path.join(model_dir, "v1_e1_aipsy_classification.csv"), index=False
                    )
                    df_e1_aipsy.to_csv(
                        os.path.join(model_dir, "e1_aipsy_classification.csv"), index=False
                    )
                    print(f"Saved Primary AIPsy probing to {os.path.join(model_dir, 'v1_e1_aipsy_classification.csv')}")

            # 2.2 Intensity Analysis: none < moderate < peak (samples with triplet_id)
            if "triplet_id" in df_aipsy.columns:
                mask_int = df_aipsy["triplet_id"].notna() & (df_aipsy["triplet_id"].astype(str).str.strip() != "")
                df_int = df_aipsy[mask_int].copy().reset_index()
                if len(df_int) >= 6:
                    print(f"Running AIPsy Intensity Analysis (none < moderate < peak, n={len(df_int)})...")
                    # Map intensity: none/neutral -> 0, moderate -> 1, peak/clinical -> 2
                    intensity_map = {
                        "none": 0.0, "neutral": 0.0,
                        "moderate": 1.0,
                        "peak": 2.0, "clinical": 2.0
                    }
                    target_series = df_int["intensity"] if "intensity" in df_int.columns else df_int["split"]
                    mapped = target_series.astype(str).str.lower().map(intensity_map)
                    bad = mapped.isna()
                    if bad.any():
                        bad_values = sorted(target_series[bad].astype(str).unique())
                        if not args.dry_run:
                            raise ValueError(f"Unknown intensity labels: {bad_values}")
                    y_int = mapped.fillna(0.0).to_numpy(dtype=float)

                    grp_int = df_int["triplet_id"].values
                    idx_int = df_int["index"].values

                    e1_intensity_records = []
                    for l in range(num_aip_layers):
                        rel_d = l / (num_aip_layers - 1) if num_aip_layers > 1 else 0.0
                        res_r_int = evaluate_regression_probe(
                            reps_r_aip[l][idx_int], y_int, group_ids=grp_int,
                            cv=phase_a_cv, alpha=phase_a_alpha, seed=phase_a_seed,
                        )
                        res_s_int = evaluate_regression_probe(
                            reps_s_aip[l][idx_int], y_int, group_ids=grp_int,
                            cv=phase_a_cv, alpha=phase_a_alpha, seed=phase_a_seed,
                        )
                        e1_intensity_records.append(
                            {
                                "layer": l,
                                "relative_depth": rel_d,
                                "dataset": "AIPsy",
                                "target": "intensity_none_moderate_peak",
                                "reader_spearman_rho": res_r_int["spearman_rho"],
                                "self_spearman_rho": res_s_int["spearman_rho"],
                                "reader_pearson_r": res_r_int["pearson_r"],
                                "self_pearson_r": res_s_int["pearson_r"],
                                "reader_r2": res_r_int["r2"],
                                "self_r2": res_s_int["r2"],
                            }
                        )
                    df_e1_int = pd.DataFrame(e1_intensity_records)
                    df_e1_int.to_csv(
                        os.path.join(model_dir, "v1_e1_aipsy_intensity.csv"), index=False
                    )
                    df_e1_int.to_csv(
                        os.path.join(model_dir, "e1_aipsy_intensity.csv"), index=False
                    )
                    print(f"Saved AIPsy Intensity analysis to {os.path.join(model_dir, 'v1_e1_aipsy_intensity.csv')}")

            # 2.3 Secondary Analysis: 8 Emotion Category Decoding (Clinical only)
            if "emotion" in df_aipsy.columns and "pair_id" in df_aipsy.columns:
                mask_sec = df_aipsy["emotion"].notna() & (df_aipsy["split"] == "clinical")
                df_sec = df_aipsy[mask_sec].copy().reset_index()
                if len(df_sec) >= 8 and df_sec["emotion"].nunique() >= 2:
                    print(f"Running Secondary Emotion-Category Probing (n={len(df_sec)}, {df_sec['emotion'].nunique()} categories)...")
                    y_sec = df_sec["emotion"].values
                    grp_sec = df_sec["pair_id"].values
                    idx_sec = df_sec["index"].values

                    e1_sec_records = []
                    for l in range(num_aip_layers):
                        rel_d = l / (num_aip_layers - 1) if num_aip_layers > 1 else 0.0
                        res_r_sec = evaluate_classification_probe(
                            reps_r_aip[l][idx_sec], y_sec, group_ids=grp_sec,
                            cv=phase_a_cv, seed=phase_a_seed,
                        )
                        res_s_sec = evaluate_classification_probe(
                            reps_s_aip[l][idx_sec], y_sec, group_ids=grp_sec,
                            cv=phase_a_cv, seed=phase_a_seed,
                        )
                        e1_sec_records.append(
                            {
                                "layer": l,
                                "relative_depth": rel_d,
                                "dataset": "AIPsy",
                                "target": "emotion_category_secondary",
                                "reader_balanced_acc": res_r_sec["balanced_acc"],
                                "self_balanced_acc": res_s_sec["balanced_acc"],
                                "reader_f1_macro": res_r_sec["f1_macro"],
                                "self_f1_macro": res_s_sec["f1_macro"],
                                "reader_roc_auc": res_r_sec["roc_auc"],
                                "self_roc_auc": res_s_sec["roc_auc"],
                            }
                        )
                    df_e1_sec = pd.DataFrame(e1_sec_records)
                    df_e1_sec.to_csv(
                        os.path.join(model_dir, "v1_e1_aipsy_emotion_secondary.csv"), index=False
                    )
                    df_e1_sec.to_csv(
                        os.path.join(model_dir, "e1_aipsy_emotion_secondary.csv"), index=False
                    )
                    print(f"Saved Secondary AIPsy Emotion Decoding to {os.path.join(model_dir, 'v1_e1_aipsy_emotion_secondary.csv')}")

        from affective_empathy_eval.io import save_experiment_result
        save_experiment_result(
            os.path.join(model_dir, f"v1_e1_decodability_{args.model_prefix}.json"),
            {
                "aipsy_primary": e1_aipsy_records if "e1_aipsy_records" in locals() else [],
                "aipsy_intensity": e1_intensity_records if "e1_intensity_records" in locals() else [],
                "aipsy_secondary": e1_sec_records if "e1_sec_records" in locals() else [],
            },
            stage="v1",
            experiment_id="v1_e1_decodability",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix, "dataset": args.dataset},
        )

    # Save manifest
    # Item 32: Phase A/B は内部表現抽出のため candidate_space="N/A", measurement_space="prompt_end_hidden_state"
    manifest_path = os.path.join(model_dir, "manifest.json")
    manifest = create_run_manifest(
        run_type="v1_phase_a",
        model_name=args.model_id,
        model_revision=args.model_revision or "main",
        config=manifest_config,
        dataset_path=dataset_paths[0] if dataset_paths else None,
        dataset_hash=dataset_hash,
        candidate_space="N/A",
        measurement_space="prompt_end_hidden_state",
        actual_dtype=str(actual_torch_dtype).replace("torch.", ""),
        seed=phase_a_seed,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    manifest.save(manifest_path)
    print(f"Saved Phase A run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
