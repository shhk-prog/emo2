#!/usr/bin/env python3
"""
behavioral/primary/run_behavioral_aipsy.py

Evaluate LLMs on AIPsy-Affect 4-split stimuli (N=480) across:
  - Task 1: Writer-State Estimation (W)
  - Task 2: Reader-Response Prediction (R)
  - Task 3: Self-Report (S)

Computes expected values E[V], E[A], E[D] and greedy argmax predictions over
all 729 joint candidate triplets (V, A, D) in {1..9}^3 using canonical Sequence Likelihood.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
from typing import Optional

from affective_empathy_eval.likelihood import (
    build_vad_candidates,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.manifests import create_run_manifest


def build_candidates():
    """Builds canonical VAD candidates using the unified likelihood module."""
    cand_dicts = build_vad_candidates()
    candidates = [c["json_str"] for c in cand_dicts]
    triplets = [(c["valence"], c["arousal"], c["dominance"]) for c in cand_dicts]
    return candidates, triplets


def construct_prompt(tokenizer, text, task_type, is_instruct=False):
    if task_type == "writer":
        instruction = "Read the following text and estimate the affective state of the writer who wrote it."
    elif task_type == "reader":
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


def evaluate_aipsy_stimuli(
    model,
    tokenizer,
    device,
    candidates,
    vad_triplets,
    stimuli_path,
    is_instruct=False,
    limit=0,
    batch_size=81,
    checkpoint_path: Optional[str] = None,
    checkpoint_meta_path: Optional[str] = None,
    expected_meta: Optional[dict] = None,
):
    df = pd.read_csv(stimuli_path)
    if limit > 0:
        df = df.head(limit)

    print(f"Starting AIPsy 4-Split evaluation on {len(df)} stimuli...")

    tasks = ["writer", "reader", "self"]
    results = []
    processed_ids = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        can_resume = False
        if checkpoint_meta_path and os.path.exists(checkpoint_meta_path) and expected_meta:
            can_resume = True
            try:
                with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                for k, v in expected_meta.items():
                    if saved_meta.get(k) != v:
                        print(f"Checkpoint meta mismatch ({k}: saved={saved_meta.get(k)} vs exp={v}). Discarding old checkpoint.")
                        can_resume = False
                        break
            except Exception as e:
                print(f"Failed to read checkpoint meta: {e}. Starting fresh.")
                can_resume = False
        else:
            print(f"Checkpoint exists at {checkpoint_path} but expected metadata file ({checkpoint_meta_path}) is missing. Discarding unverified checkpoint.")
            can_resume = False

        if can_resume:
            try:
                ckpt_df = pd.read_csv(checkpoint_path)
                results = ckpt_df.to_dict("records")
                if "id" in ckpt_df.columns:
                    processed_ids = set(ckpt_df["id"].tolist())
                print(f"Resuming from checkpoint: {len(processed_ids)} samples already completed.")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                results = []
                processed_ids = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="AIPsy Evaluation"):
        s_id = row["id"] if "id" in row else idx
        if s_id in processed_ids:
            continue

        text = row["text"]
        row_dict = row.to_dict()

        for task in tasks:
            prompt = construct_prompt(
                tokenizer, text, task, is_instruct=is_instruct
            )
            _, probs = compute_sequence_likelihoods_for_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                device=device,
                batch_size=batch_size,
                normalize_length=False,
            )

            # Expected values
            ev = sum(p * t[0] for p, t in zip(probs, vad_triplets))
            ea = sum(p * t[1] for p, t in zip(probs, vad_triplets))
            ed = sum(p * t[2] for p, t in zip(probs, vad_triplets))

            # Greedy argmax
            argmax_idx = int(np.argmax(probs))
            best_v, best_a, best_d = vad_triplets[argmax_idx]

            # Probability of exact (5,5,5) neutral
            neutral_idx = vad_triplets.index((5, 5, 5))
            p555 = probs[neutral_idx]

            prefix = task[0]  # 'w', 'r', 's'
            row_dict[f"{prefix}_ev"] = float(ev)
            row_dict[f"{prefix}_ea"] = float(ea)
            row_dict[f"{prefix}_ed"] = float(ed)
            row_dict[f"{prefix}_gv"] = int(best_v)
            row_dict[f"{prefix}_ga"] = int(best_a)
            row_dict[f"{prefix}_gd"] = int(best_d)
            row_dict[f"{prefix}_p555"] = float(p555)

        results.append(row_dict)
        processed_ids.add(s_id)

        # 逐次保存（10サンプルごと）
        if len(results) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if checkpoint_path:
                try:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                except Exception:
                    pass

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Primary: AIPsy-Affect 4-Split Evaluation"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HF name"
    )
    parser.add_argument(
        "--tag", type=str, required=True, help="Short tag for the model"
    )
    parser.add_argument(
        "--is-instruct",
        action="store_true",
        help="Whether to apply chat template",
    )
    parser.add_argument(
        "--stimuli-path",
        type=str,
        default="v1/data/processed/aipsy_4split_all.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="behavioral/results/raw/aipsy_4split",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional limit for dry-run"
    )
    parser.add_argument("--batch-size", type=int, default=81)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mock mode without loading model weights",
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
        help="Force recomputation even if output CSV already exists",
    )
    args = parser.parse_args()

    # Production safety valve: ensure fixed model revision is resolved
    if args.model_revision is None and not getattr(args, "dry_run", False):
        from affective_empathy_eval.models.registry import get_registry
        registry = get_registry()
        fam_cfg = registry.get_family_by_model_id(args.model)
        if fam_cfg:
            for spec in (fam_cfg.base_model, fam_cfg.instruct_model):
                if spec.model_id == args.model and spec.revision:
                    args.model_revision = spec.revision
                    break
        if not args.model_revision:
            raise ValueError(
                f"Production run requires explicit fixed model revision for '{args.model}', but none was provided or resolved from registry."
            )

    stim_path = Path(args.stimuli_path)
    if not stim_path.exists():
        fallback = Path("data/processed/aipsy_4split_all.csv")
        if fallback.exists():
            stim_path = fallback
        else:
            raise FileNotFoundError(
                f"Stimuli dataset not found: {args.stimuli_path} or {fallback}"
            )

    if getattr(args, "dry_run", False):
        args.out_dir = str(Path(args.out_dir) / "dry_run")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"behavioral_aipsy_{args.tag}_4split.csv")
    out_csv_compat = os.path.join(args.out_dir, f"{args.tag}_4split.csv")
    out_json = os.path.join(args.out_dir, f"behavioral_aipsy_{args.tag}_summary.json")
    manifest_path = os.path.join(args.out_dir, f"behavioral_aipsy_{args.tag}_manifest.json")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_csv = os.path.join(ckpt_dir, f"{args.tag}_aipsy_4split_checkpoint.csv")
    ckpt_meta = os.path.join(ckpt_dir, f"{args.tag}_aipsy_4split_meta.json")

    from affective_empathy_eval.manifests import (
        compute_file_hash,
        compute_prompt_hash,
        compute_string_or_dict_hash,
        is_manifest_matching,
        create_run_manifest,
    )
    from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

    device_str = args.device
    actual_dtype_str = "bfloat16" if (device_str != "cpu" and torch.cuda.is_available()) else "float32"
    prompt_template_desc = (
        "v1_aipsy_4split_vad_json|"
        f"is_instruct={args.is_instruct}|"
        "tasks=writer,reader,self|aipsy_prompts"
    )
    prompt_hash = compute_prompt_hash(prompt_template_desc)
    dataset_hash = compute_file_hash(stim_path) if stim_path.exists() else "unknown"

    all_candidates, _ = build_candidates()
    candidate_hash = compute_string_or_dict_hash(all_candidates)

    manifest_config = {
        "model_id": args.model,
        "model_revision": args.model_revision or "main",
        "tag": args.tag,
        "is_instruct": args.is_instruct,
        "limit": args.limit,
        "dataset_hash": dataset_hash,
        "candidate_space": "VAD_729",
        "candidate_hash": candidate_hash,
        "prompt_hash": prompt_hash,
        "actual_dtype": actual_dtype_str,
    }
    expected_config_hash = compute_string_or_dict_hash(manifest_config)

    expected_meta = {
        "model": args.model,
        "model_revision": args.model_revision or "main",
        "limit": args.limit,
        "is_instruct": args.is_instruct,
        "stimuli_hash": dataset_hash,
        "prompt_hash": prompt_hash,
        "candidate_hash": candidate_hash,
        "dtype": actual_dtype_str,
    }

    # Early skip if already evaluated and valid (Resume support)
    skip_eval = False
    if not args.force and (os.path.exists(out_csv) or os.path.exists(out_csv_compat)):
        check_path = out_csv if os.path.exists(out_csv) else out_csv_compat
        try:
            cached_df = pd.read_csv(check_path)
            expected_min = args.limit if (args.limit and args.limit > 0) else 1
            if len(cached_df) >= expected_min and "s_ev" in cached_df.columns:
                if os.path.exists(out_json) and is_experiment_completed(out_json, manifest_path=manifest_path, force=args.force):
                    if is_manifest_matching(
                        manifest_path=manifest_path,
                        expected_model_name=args.model,
                        expected_config_hash=expected_config_hash,
                        expected_dataset_hash=dataset_hash,
                        expected_prompt_hash=prompt_hash,
                        expected_model_revision=args.model_revision,
                        expected_dry_run=args.dry_run,
                    ):
                        print(
                            f"[SKIP] Existing validated results matching manifest found at {check_path} (n={len(cached_df)}). "
                            f"Skipping model loading & evaluation for {args.tag}. Use --force to rerun."
                        )
                        res_df = cached_df
                        skip_eval = True
        except Exception as e:
            print(f"Warning: Corrupt or unreadable output at {check_path} ({e}). Rerunning.")
            skip_eval = False
    else:
        skip_eval = False

    if not skip_eval:
        if args.dry_run:
            print(f"[DRY-RUN] Simulating AIPsy 4-Split evaluation for {args.tag}...")
            records = []
            quads = ["high_v_high_a", "high_v_low_a", "low_v_high_a", "low_v_low_a"]
            for i in range(16):
                q = quads[i % 4]
                records.append({
                    "id": f"dry_aipsy_{i}",
                    "text": f"Mock AIPsy text {i}",
                    "quad": q,
                    "valence": 7.0 if "high_v" in q else 3.0,
                    "arousal": 7.0 if "high_a" in q else 3.0,
                    "dominance": 5.0,
                    "s_ev": 6.8 if "high_v" in q else 3.2,
                    "s_ea": 6.5 if "high_a" in q else 3.5,
                    "s_ed": 5.0,
                    "r_ev": 6.9 if "high_v" in q else 3.1,
                    "r_ea": 6.7 if "high_a" in q else 3.3,
                    "r_ed": 5.0,
                })
            res_df = pd.DataFrame(records)
            res_df.to_csv(out_csv, index=False)
            res_df.to_csv(out_csv_compat, index=False)
        else:
            # Save checkpoint meta
            with open(ckpt_meta, "w", encoding="utf-8") as f:
                json.dump(expected_meta, f, indent=2)

            print("=" * 60)
            print(
                f"Evaluating Model: {args.model} (Tag: {args.tag}, Instruct: {args.is_instruct})"
            )
            print(f"Stimuli Path: {stim_path} (Device: {args.device})")
            print("=" * 60)

            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.model_revision,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = args.device
            torch_dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32

            if device.startswith("cuda:"):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    revision=args.model_revision,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
            elif device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    revision=args.model_revision,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    revision=args.model_revision,
                    torch_dtype=torch_dtype,
                    device_map=None,
                    trust_remote_code=True,
                ).to(device)
            model.eval()

            candidates, vad_triplets = build_candidates()
            print(f"Generated {len(candidates)} VAD candidate triplets in {{1..9}}^3.")

            res_df = evaluate_aipsy_stimuli(
                model,
                tokenizer,
                args.device,
                candidates,
                vad_triplets,
                stimuli_path=str(stim_path),
                is_instruct=args.is_instruct,
                limit=args.limit,
                batch_size=args.batch_size,
                checkpoint_path=ckpt_csv,
                checkpoint_meta_path=ckpt_meta,
                expected_meta=expected_meta,
            )

            res_df.to_csv(out_csv, index=False)
            try:
                res_df.to_csv(out_csv_compat, index=False)
            except Exception:
                pass
    if os.path.exists(ckpt_csv):
        try:
            os.remove(ckpt_csv)
        except Exception:
            pass
    if os.path.exists(ckpt_meta):
        try:
            os.remove(ckpt_meta)
        except Exception:
            pass

    # Save summary with execution_success: true
    summary = {
        "tag": args.tag,
        "model": args.model,
        "is_instruct": args.is_instruct,
        "num_samples": len(res_df),
        "mean_s_ev": float(res_df["s_ev"].mean()) if "s_ev" in res_df.columns else None,
        "mean_s_ea": float(res_df["s_ea"].mean()) if "s_ea" in res_df.columns else None,
    }
    save_experiment_result(
        output_path=out_json,
        payload=summary,
        stage="behavioral",
        experiment_id="behavioral_aipsy_4split",
        success=True,
        metadata={"tag": args.tag, "model": args.model, "is_instruct": args.is_instruct},
    )

    # Save manifest
    manifest = create_run_manifest(
        run_type="behavioral_aipsy_4split",
        model_name=args.model,
        model_revision=args.model_revision or "main",
        config=manifest_config,
        metadata=summary,
        dataset_path=str(stim_path),
        prompt_hash=prompt_hash,
        candidate_space="VAD_729",
        measurement_space="VAD_expectation_from_VAD_729",
        actual_dtype=actual_dtype_str,
        intervention_version="none",
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    manifest.save(manifest_path)
    try:
        manifest.save(os.path.join(args.out_dir, f"{args.tag}_manifest.json"))
    except Exception:
        pass

    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved {len(res_df)} evaluation results to {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
