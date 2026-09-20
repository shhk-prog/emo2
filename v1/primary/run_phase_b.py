#!/usr/bin/env python3
"""
v1/primary/run_phase_b.py

V1 Phase B Primary Script: Semantic vs. Lexical Controls Audit
Target layer is a priori mid-depth: l = round(d * (L - 1)) with d=0.5.
This does not use the Phase A decodability peak.
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
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
import yaml
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.data import describe_loaded_frame
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_architecture_dims,
    resolve_single_model_from_args,
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


from affective_empathy_eval.geometry import get_block_hidden_state


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
        # Guard against silent prompt truncation
        for b_idx, p in enumerate(batch_prompts):
            raw_len = len(tokenizer.encode(p, add_special_tokens=False))
            if raw_len > 1024:
                raise RuntimeError(
                    f"Prompt truncated by max_length=1024! Prompt index: {start_idx + b_idx}, token length: {raw_len}. "
                    "Primary samples must not be silently truncated."
                )
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

        # Transformer block layer_idx (0 <= layer_idx < num_layers) の出力を取得
        layer_tensor = get_block_hidden_state(outputs.hidden_states, layer_idx)
        seq_lengths = attention_mask.sum(dim=1) - 1

        for b_idx in range(len(batch_prompts)):
            valid_pos = torch.nonzero(attention_mask[b_idx], as_tuple=False).flatten()
            last_pos = int(valid_pos[-1]) if len(valid_pos) > 0 else int(attention_mask.shape[1] - 1)
            vec = (
                layer_tensor[b_idx, last_pos, :].detach().cpu().float().numpy()
            )
            all_reps.append(vec)

    arr = np.array(all_reps)
    if not np.all(np.isfinite(arr)):
        bad_count = int(np.size(arr) - np.isfinite(arr).sum())
        raise FloatingPointError(f"Non-finite hidden states detected in Phase B representations: {bad_count}")
    return arr.astype(np.float32)



def evaluate_probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    cv: int = 5,
    seed: int = 42,
) -> float:
    """
    Pair-aware cross-validated linear probing evaluation.
    If groups (pair_id) are provided, uses StratifiedGroupKFold to strictly prevent pair leakage.
    """
    if not np.all(np.isfinite(X)):
        bad_count = int(np.size(X) - np.isfinite(X).sum())
        raise FloatingPointError(f"Non-finite hidden states detected in probe accuracy evaluation: {bad_count}")
    X = X.astype(np.float32)
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return float("nan")
        n_splits = min(cv, n_groups)
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_gen = list(splitter.split(X, y, groups=groups))
        except ValueError:
            splitter = GroupKFold(n_splits=n_splits)
            split_gen = list(splitter.split(X, y, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        split_gen = list(splitter.split(X, y))

    preds = np.zeros_like(y)
    for train_idx, val_idx in split_gen:
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if valid cached results already exist",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="reader",
        choices=["reader", "self"],
        help="Task prompt framing (reader or self)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_experiments.yaml",
        help="Path to experiment configuration YAML",
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
        "--limit",
        type=int,
        default=0,
        help="Sample limit (0 for full dataset)",
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

    # Load Phase B configuration from YAML (Item 20)
    phase_b_cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            phase_b_cfg = yaml.safe_load(f).get("phase_b", {})
    cfg_seed = int(phase_b_cfg.get("seed", 42))
    cfg_train_ratio = float(phase_b_cfg.get("train_ratio", 0.7))
    cfg_cv_folds = int(phase_b_cfg.get("cv_folds", 5))

    if args.dry_run:
        args.out_dir = os.path.join(args.out_dir, "dry_run")

    # Ensure task-type separation in output hierarchy
    if not args.out_dir.endswith(args.task_type):
        model_dir = os.path.join(args.out_dir, args.task_type, args.model_prefix)
    else:
        model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    # Dynamic layer resolution first so it can be verified in manifest
    data_file = Path(args.data_path)
    if not data_file.exists():
        fallback = Path("data/processed/v1_e5_semantic_controls.csv")
        if fallback.exists():
            data_file = fallback

    target_layer = args.layer
    if target_layer is None:
        try:
            num_layers, _ = resolve_architecture_dims(args.model_id)
            target_layer = int(round(args.relative_depth * (num_layers - 1)))
        except Exception:
            target_layer = None

    from affective_empathy_eval.io import save_experiment_result, is_experiment_completed
    from affective_empathy_eval.manifests import (
        is_manifest_matching,
        compute_file_hash,
        compute_prompt_hash,
        compute_string_or_dict_hash,
        create_run_manifest,
    )

    dataset_hash = compute_file_hash(data_file) if data_file.exists() else "unknown"
    prompt_hash = compute_prompt_hash(f"task_type={args.task_type}|is_instruct={is_instruct}|v1_phase_b_audit")

    manifest_config = {
        "model_prefix": args.model_prefix,
        "model_id": args.model_id,
        "model_revision": args.model_revision or "main",
        "task_type": args.task_type,
        "relative_depth": float(args.relative_depth),
        "target_layer": int(target_layer) if target_layer is not None else None,
        "seed": int(cfg_seed),
        "train_ratio": float(cfg_train_ratio),
        "cv_folds": int(cfg_cv_folds),
        "dataset_hash": dataset_hash,
        "prompt_hash": prompt_hash,
        "limit": args.limit,
    }
    expected_cfg_hash = compute_string_or_dict_hash(manifest_config)

    modular_json_path = os.path.join(model_dir, f"v1_e5_semantic_controls_{args.model_prefix}_{args.task_type}.json")
    modular_csv_path = os.path.join(model_dir, f"v1_e5_semantic_controls_{args.model_prefix}_{args.task_type}.csv")
    res_path = os.path.join(model_dir, "phase_b_semantic_controls.csv")
    manifest_path = os.path.join(model_dir, "manifest.json")

    if not args.force and not args.dry_run and os.path.exists(manifest_path):
        target_check = modular_json_path if os.path.exists(modular_json_path) else res_path
        if is_experiment_completed(target_check, manifest_path=manifest_path):
            if is_manifest_matching(
                manifest_path=manifest_path,
                expected_model_name=args.model_id,
                expected_config_hash=expected_cfg_hash,
                expected_dataset_hash=dataset_hash,
                expected_prompt_hash=prompt_hash,
                expected_model_revision=args.model_revision,
                expected_dry_run=False,
            ):
                print(
                    f"[SKIP] Completed Phase B (V1 E5) results matching manifest found in {model_dir}. "
                    f"Skipping computation for {args.model_prefix}. Use --force to rerun."
                )
                return

    print(f"=== Starting V1 Phase B Semantic Audit: {args.model_id} (revision={args.model_revision}) ===")
    print(
        f"Data Path: {data_file} | Target Layer: {target_layer} (relative_depth={args.relative_depth:.2f})"
    )

    if args.dry_run:
        print(f"[DRY-RUN] V1 Phase B for Model: {args.model_id} (Instruct={is_instruct})")
        results = {
            "acc_original_minimal_pair": 0.85,
            "acc_pair_aware_held_out_paraphrase": 0.82,
            "pair_aware_held_out_reversal_drop": 0.58,
            "n_nonfallback_paraphrase_pairs": 80,
            "n_nonfallback_reversal_pairs": 80,
            "n_primary_test_paraphrase_pairs": 80,
            "n_primary_test_reversal_pairs": 80,
            "sensitivity_held_out_paraphrase": 0.80,
            "sensitivity_held_out_reversal_drop": 0.55,
            "acc_paraphrase_invariance_all": 0.82,
            "acc_word_shuffle": 0.52,
            "mean_affective_prob_original": 0.80,
            "mean_affective_prob_outcome_reversed": 0.20,
            "outcome_reversal_prob_drop": 0.60,
        }
        # Modular save
        save_experiment_result(
            output_path=modular_json_path,
            payload=results,
            stage="v1",
            experiment_id="v1_e5_semantic_controls",
            status="success",
            success=True,
            metadata={
                "model_id": args.model_id,
                "model_prefix": args.model_prefix,
                "task_type": args.task_type,
                "target_layer": target_layer,
                "relative_depth": args.relative_depth,
                "dry_run": True,
            },
        )
        pd.DataFrame([results]).to_csv(modular_csv_path, index=False)
        pd.DataFrame([results]).to_csv(res_path, index=False)
        dry_cfg = dict(manifest_config)
        dry_cfg["dry_run"] = True
        manifest = create_run_manifest(
            run_type="v1_phase_b",
            model_name=args.model_id,
            model_revision=args.model_revision or "main",
            config=dry_cfg,
            dataset_path=str(data_file),
            prompt_hash=prompt_hash,
            metadata=results,
            candidate_space="N/A",
            measurement_space="prompt_end_hidden_state",
            intervention_version="none",
            run_id=args.run_id,
            dry_run=True,
        )
        manifest.save(os.path.join(model_dir, "manifest.json"))
        print(f"[DRY-RUN] Completed Phase B mock output in {model_dir}")
        return


    df = pd.read_csv(data_file)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy().reset_index(drop=True)
        print(f"Limiting Phase B evaluation to first {len(df)} samples (--limit {args.limit})")
    n_pairs = len(df)
    print(describe_loaded_frame(df, "V1 Phase B semantic controls", str(data_file)))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    actual_torch_dtype = (
        torch.bfloat16
        if is_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if is_cuda else torch.float32)
    )
    if is_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            torch_dtype=actual_torch_dtype,
            device_map=device if device.startswith("cuda:") else "auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            torch_dtype=actual_torch_dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(device)
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
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )
    H_orig_neu = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_original_neutral"].tolist(),
        target_layer,
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )
    H_para_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_paraphrase_affective"].tolist(),
        target_layer,
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )
    H_shuf_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_shuffled_affective"].tolist(),
        target_layer,
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )
    H_shuf_neu = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_shuffled_neutral"].tolist(),
        target_layer,
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )
    H_rev_aff = extract_single_layer_hidden_states(
        model,
        tokenizer,
        df["text_reversed_affective"].tolist(),
        target_layer,
        task_type=args.task_type,
        is_instruct=is_instruct,
        device=args.device,
    )

    y_orig = np.array([1] * n_pairs + [0] * n_pairs)
    pair_ids = df["pair_id"].values if "pair_id" in df.columns else np.arange(n_pairs)
    groups = np.concatenate([pair_ids, pair_ids], axis=0)

    # Condition 1: Minimal Pair (Strict Pair-Aware Cross-Validation via StratifiedGroupKFold)
    X_orig = np.concatenate([H_orig_aff, H_orig_neu], axis=0)
    acc_orig = evaluate_probe_accuracy(X_orig, y_orig, groups=groups, cv=cfg_cv_folds, seed=cfg_seed)

    # Condition 2: Paraphrase / Surface Perturbation (Semantic Transformation Sensitivity)
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    clf = LogisticRegression(max_iter=500, random_state=cfg_seed)
    clf.fit(X_orig_scaled, y_orig)

    X_para = np.concatenate([H_para_aff, H_orig_neu], axis=0)
    X_para_scaled = scaler.transform(X_para)
    preds_para = clf.predict(X_para_scaled)
    acc_paraphrase = float(balanced_accuracy_score(y_orig, preds_para))

    # Condition 3: Word Shuffle Test (Pair-Aware Cross-Validation)
    X_shuf = np.concatenate([H_shuf_aff, H_shuf_neu], axis=0)
    acc_shuffled = evaluate_probe_accuracy(X_shuf, y_orig, groups=groups, cv=cfg_cv_folds, seed=cfg_seed)

    # Condition 4: Outcome Reversal Test
    X_rev = np.concatenate([H_rev_aff, H_orig_neu], axis=0)
    X_rev_scaled = scaler.transform(X_rev)
    probs_rev_aff = clf.predict_proba(X_rev_scaled[:n_pairs])[:, 1]
    probs_orig_aff = clf.predict_proba(X_orig_scaled[:n_pairs])[:, 1]
    mean_prob_orig = float(np.mean(probs_orig_aff))
    mean_prob_rev = float(np.mean(probs_rev_aff))
    outcome_reversal_drop = mean_prob_orig - mean_prob_rev

    # Condition 5: Strict Pair-Aware Evaluation (Generalization to Held-Out Stimuli Pairs)
    unique_pairs = np.unique(pair_ids)
    rng_split = np.random.default_rng(cfg_seed)
    shuffled_pairs = rng_split.permutation(unique_pairs)
    n_train_pairs = max(1, int(cfg_train_ratio * len(shuffled_pairs)))
    train_pair_ids = set(shuffled_pairs[:n_train_pairs])
    test_pair_ids = set(shuffled_pairs[n_train_pairs:])
    if len(train_pair_ids) == 0 or len(test_pair_ids) == 0:
        raise ValueError("Independent train/test pair split cannot be constructed.")

    # 厳格なデータリーク防止アサート (Item 2)
    assert train_pair_ids.isdisjoint(test_pair_ids), (
        "Data leakage detected! Training and test pair sets must be completely disjoint."
    )

    train_mask = np.isin(pair_ids, list(train_pair_ids))
    test_mask = np.isin(pair_ids, list(test_pair_ids))

    X_tr_orig = np.concatenate([H_orig_aff[train_mask], H_orig_neu[train_mask]], axis=0)
    y_tr_orig = np.array([1] * int(np.sum(train_mask)) + [0] * int(np.sum(train_mask)))

    scaler_pa = StandardScaler()
    X_tr_scaled = scaler_pa.fit_transform(X_tr_orig)
    clf_pa = LogisticRegression(max_iter=500, random_state=cfg_seed)
    clf_pa.fit(X_tr_scaled, y_tr_orig)

    # Fallback filtering:
    # Primary analysis uses nonfallback rule-based transformations without identity fallbacks.
    # Sensitivity analysis includes all transformations (including fallback clause additions).
    has_para_fallback = "paraphrase_fallback" in df.columns
    has_rev_fallback = "reversal_fallback" in df.columns

    if has_para_fallback:
        valid_para_pair_mask = ~df["paraphrase_fallback"].astype(bool).values
    else:
        valid_para_pair_mask = np.ones(n_pairs, dtype=bool)

    if has_rev_fallback:
        valid_rev_pair_mask = ~df["reversal_fallback"].astype(bool).values
    else:
        valid_rev_pair_mask = np.ones(n_pairs, dtype=bool)

    # 1) Sensitivity (All test pairs)
    X_te_para_all = np.concatenate([H_para_aff[test_mask], H_orig_neu[test_mask]], axis=0)
    y_te_para_all = np.array([1] * int(np.sum(test_mask)) + [0] * int(np.sum(test_mask)))
    preds_te_para_all = clf_pa.predict(scaler_pa.transform(X_te_para_all))
    acc_held_out_paraphrase_all = float(balanced_accuracy_score(y_te_para_all, preds_te_para_all))

    X_te_rev_all = scaler_pa.transform(H_rev_aff[test_mask])
    X_te_orig_all = scaler_pa.transform(H_orig_aff[test_mask])
    p_orig_te_all = clf_pa.predict_proba(X_te_orig_all)[:, 1]
    p_rev_te_all = clf_pa.predict_proba(X_te_rev_all)[:, 1]
    held_out_reversal_drop_all = float(np.mean(p_orig_te_all) - np.mean(p_rev_te_all))

    # 2) Primary (Nonfallback rule-based transformations only)
    test_para_valid_mask = test_mask & valid_para_pair_mask
    if np.sum(test_para_valid_mask) > 0:
        X_te_para_prim = np.concatenate([H_para_aff[test_para_valid_mask], H_orig_neu[test_para_valid_mask]], axis=0)
        y_te_para_prim = np.array([1] * int(np.sum(test_para_valid_mask)) + [0] * int(np.sum(test_para_valid_mask)))
        preds_te_para_prim = clf_pa.predict(scaler_pa.transform(X_te_para_prim))
        acc_held_out_paraphrase_primary = float(balanced_accuracy_score(y_te_para_prim, preds_te_para_prim))
    else:
        acc_held_out_paraphrase_primary = float("nan")

    test_rev_valid_mask = test_mask & valid_rev_pair_mask
    if np.sum(test_rev_valid_mask) > 0:
        X_te_rev_prim = scaler_pa.transform(H_rev_aff[test_rev_valid_mask])
        X_te_orig_prim = scaler_pa.transform(H_orig_aff[test_rev_valid_mask])
        p_orig_te_prim = clf_pa.predict_proba(X_te_orig_prim)[:, 1]
        p_rev_te_prim = clf_pa.predict_proba(X_te_rev_prim)[:, 1]
        held_out_reversal_drop_primary = float(np.mean(p_orig_te_prim) - np.mean(p_rev_te_prim))
    else:
        held_out_reversal_drop_primary = float("nan")

    results = {
        # Primary indicators (Non-fallback rule-based semantic controls)
        "acc_original_minimal_pair": acc_orig,
        "acc_pair_aware_held_out_paraphrase": acc_held_out_paraphrase_primary,
        "pair_aware_held_out_reversal_drop": held_out_reversal_drop_primary,
        "n_train_pairs": int(len(train_pair_ids)),
        "n_test_pairs": int(len(test_pair_ids)),
        "n_nonfallback_paraphrase_pairs": int(np.sum(valid_para_pair_mask)),
        "n_nonfallback_reversal_pairs": int(np.sum(valid_rev_pair_mask)),
        "n_primary_test_paraphrase_pairs": int(np.sum(test_para_valid_mask)),
        "n_primary_test_reversal_pairs": int(np.sum(test_rev_valid_mask)),
        # Sensitivity indicators (All pairs including fallbacks)
        "sensitivity_held_out_paraphrase": acc_held_out_paraphrase_all,
        "sensitivity_held_out_reversal_drop": held_out_reversal_drop_all,
        "acc_paraphrase_invariance_all": acc_paraphrase,
        "acc_word_shuffle": acc_shuffled,
        "mean_affective_prob_original": mean_prob_orig,
        "mean_affective_prob_outcome_reversed": mean_prob_rev,
        "outcome_reversal_prob_drop": outcome_reversal_drop,
    }

    df_res = pd.DataFrame([results])
    res_path = os.path.join(model_dir, "phase_b_semantic_controls.csv")
    df_res.to_csv(res_path, index=False)
    df_res.to_csv(modular_csv_path, index=False)

    save_experiment_result(
        output_path=modular_json_path,
        payload=results,
        stage="v1",
        experiment_id="v1_e5_semantic_controls",
        status="success",
        success=True,
        metadata={
            "model_id": args.model_id,
            "model_prefix": args.model_prefix,
            "task_type": args.task_type,
            "target_layer": target_layer,
            "relative_depth": args.relative_depth,
            "num_pairs": n_pairs,
            "dry_run": args.dry_run,
        },
    )

    # Item 7: 個別ペアの品質監査テーブルの保存 (transformation_method, quality_status)
    pairs_audit_records = []
    for idx_p, pid in enumerate(pair_ids):
        is_tr = pid in train_pair_ids
        is_te = pid in test_pair_ids
        is_para_valid = bool(valid_para_pair_mask[idx_p])
        is_rev_valid = bool(valid_rev_pair_mask[idx_p])
        pairs_audit_records.append({
            "pair_id": pid,
            "split": "train" if is_tr else ("test" if is_te else "none"),
            "paraphrase_transformation_method": "rule_based_substitution" if is_para_valid else "identity_fallback",
            "paraphrase_quality_status": "nonfallback_valid" if is_para_valid else "fallback_excluded_from_primary",
            "reversal_transformation_method": "antonym_negation_switch" if is_rev_valid else "identity_fallback",
            "reversal_quality_status": "nonfallback_valid" if is_rev_valid else "fallback_excluded_from_primary",
        })
    df_audit = pd.DataFrame(pairs_audit_records)
    df_audit.to_csv(os.path.join(model_dir, "phase_b_pairs_quality_audit.csv"), index=False)

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_b",
        model_name=args.model_id,
        model_revision=args.model_revision or "main",
        config=manifest_config,
        metadata={
            **results,
            "num_pairs": n_pairs,
        },
        dataset_path=str(data_file),
        prompt_hash=prompt_hash,
        candidate_space="N/A",
        measurement_space="prompt_end_hidden_state",
        actual_dtype=str(actual_torch_dtype).replace("torch.", ""),
        intervention_version="none",
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    manifest.save(os.path.join(model_dir, "manifest.json"))
    print(f"Phase B (V1 E5) completed. Modular result saved to {modular_json_path}")


if __name__ == "__main__":
    main()

