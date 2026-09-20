#!/usr/bin/env python3
"""
behavioral/primary/run_behavioral_emobank.py

3-Way VAD (Valence-Arousal-Dominance) Behavioral Evaluation on EmoBank.
Evaluates:
  1. Writer-State Estimation (W): "What did the writer feel?" -> Ground Truth: EmoBank Writer VAD
  2. Reader-Response Prediction (R): "What will human readers feel?" -> Ground Truth: EmoBank Reader VAD
  3. Self-Report (S): "What do you feel?" -> Reference: EmoBank Reader VAD + Model-Internal R <-> S correlation

Uses 729-candidate Sequence-Likelihood Protocol: (V, A, D) in {1..9}^3
Computes continuous expected values E[V], E[A], E[D] as well as greedy discrete outputs.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from tqdm import tqdm
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from typing import Optional

from affective_empathy_eval.likelihood import (
    build_vad_candidates,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.statistics import compute_bootstrap_ci, compute_d_z


def get_vad_candidates_and_triplets():
    """Generates all 729 canonical VAD JSON candidates from the unified likelihood module."""
    cand_dicts = build_vad_candidates()
    candidates = [c["json_str"] for c in cand_dicts]
    vad_triplets = [(c["valence"], c["arousal"], c["dominance"]) for c in cand_dicts]
    return candidates, vad_triplets


def compute_expected_vad(likelihoods, vad_triplets, tau=1.0):
    """Computes expected values E[V], E[A], E[D], entropy, and p(5,5,5) from log-likelihoods."""
    l_arr = np.array(likelihoods, dtype=np.float64) / tau
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)

    E_v = float(np.sum(probs * [t[0] for t in vad_triplets]))
    E_a = float(np.sum(probs * [t[1] for t in vad_triplets]))
    E_d = float(np.sum(probs * [t[2] for t in vad_triplets]))

    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

    p_555 = 0.0
    for p, (v, a, d) in zip(probs, vad_triplets):
        if v == 5 and a == 5 and d == 5:
            p_555 = float(p)
            break

    return E_v, E_a, E_d, entropy, p_555, probs


def format_3way_prompt(tokenizer, text, task_type="reader", is_instruct=True):
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


def evaluate_model(
    model,
    tokenizer,
    device,
    candidates,
    vad_triplets,
    stimuli_path,
    is_instruct,
    limit=None,
    batch_size=81,
    checkpoint_path: Optional[str] = None,
    checkpoint_meta_path: Optional[str] = None,
    expected_meta: Optional[dict] = None,
):
    df = pd.read_csv(stimuli_path)
    if limit:
        df = df.head(limit)

    print(f"Loaded {len(df)} 3-way VAD stimuli from {stimuli_path}.")
    results = []
    processed_ids = set()
    checkpoint_used = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        can_resume = True
        if checkpoint_meta_path and os.path.exists(checkpoint_meta_path) and expected_meta:
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

        if can_resume:
            try:
                ckpt_df = pd.read_csv(checkpoint_path)
                results = ckpt_df.to_dict("records")
                processed_ids = set(ckpt_df["id"].tolist())
                checkpoint_used = True
                print(f"Resuming from checkpoint: {len(processed_ids)} samples already completed.")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
                results = []
                processed_ids = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="3-Way VAD Eval"):
        s_id = row["id"]
        if s_id in processed_ids:
            continue

        text = row["text"]
        writer_v_raw = row["writer_V"]
        writer_a_raw = row["writer_A"]
        writer_d_raw = row["writer_D"]
        reader_v_raw = row["reader_V"]
        reader_a_raw = row["reader_A"]
        reader_d_raw = row["reader_D"]

        # 1. Writer Estimation
        prompt_w = format_3way_prompt(
            tokenizer, text, task_type="writer", is_instruct=is_instruct
        )
        ll_w, _ = compute_sequence_likelihoods_for_candidates(
            model, tokenizer, prompt_w, candidates, device=device, batch_size=batch_size
        )
        w_ev, w_ea, w_ed, w_ent, w_p555, w_probs = compute_expected_vad(
            ll_w.tolist(), vad_triplets
        )
        best_w_idx = int(np.argmax(w_probs))
        w_gv, w_ga, w_gd = vad_triplets[best_w_idx]

        # 2. Reader Prediction
        prompt_r = format_3way_prompt(
            tokenizer, text, task_type="reader", is_instruct=is_instruct
        )
        ll_r, _ = compute_sequence_likelihoods_for_candidates(
            model, tokenizer, prompt_r, candidates, device=device, batch_size=batch_size
        )
        r_ev, r_ea, r_ed, r_ent, r_p555, r_probs = compute_expected_vad(
            ll_r.tolist(), vad_triplets
        )
        best_r_idx = int(np.argmax(r_probs))
        r_gv, r_ga, r_gd = vad_triplets[best_r_idx]

        # 3. Self-Report
        prompt_s = format_3way_prompt(
            tokenizer, text, task_type="self", is_instruct=is_instruct
        )
        ll_s, _ = compute_sequence_likelihoods_for_candidates(
            model, tokenizer, prompt_s, candidates, device=device, batch_size=batch_size
        )
        s_ev, s_ea, s_ed, s_ent, s_p555, s_probs = compute_expected_vad(
            ll_s.tolist(), vad_triplets
        )
        best_s_idx = int(np.argmax(s_probs))
        s_gv, s_ga, s_gd = vad_triplets[best_s_idx]

        results.append(
            {
                "id": s_id,
                "text": text,
                "human_writer_v": writer_v_raw,
                "human_writer_a": writer_a_raw,
                "human_writer_d": writer_d_raw,
                "human_reader_v": reader_v_raw,
                "human_reader_a": reader_a_raw,
                "human_reader_d": reader_d_raw,
                "w_ev": w_ev,
                "w_ea": w_ea,
                "w_ed": w_ed,
                "w_gv": w_gv,
                "w_ga": w_ga,
                "w_gd": w_gd,
                "w_p555": w_p555,
                "r_ev": r_ev,
                "r_ea": r_ea,
                "r_ed": r_ed,
                "r_gv": r_gv,
                "r_ga": r_ga,
                "r_gd": r_gd,
                "r_p555": r_p555,
                "s_ev": s_ev,
                "s_ea": s_ea,
                "s_ed": s_ed,
                "s_gv": s_gv,
                "s_ga": s_ga,
                "s_gd": s_gd,
                "s_p555": s_p555,
            }
        )
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
        description="Behavioral Primary: EmoBank 3-Way VAD Evaluation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--is_instruct", action="store_true")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--stimuli-path",
        type=str,
        default="v1/data/processed/stimuli_vad_3way.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="behavioral/results/raw/emobank_3way",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=81)
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

    # Find stimuli path if relative to workspace
    stim_path = Path(args.stimuli_path)
    if not stim_path.exists():
        fallback = Path("data/processed/stimuli_vad_3way.csv")
        if fallback.exists():
            stim_path = fallback
        else:
            raise FileNotFoundError(
                f"Stimuli dataset not found: {args.stimuli_path} or {fallback}"
            )
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"behavioral_emobank_{args.tag}_3way_vad.csv")
    out_csv_compat = os.path.join(args.out_dir, f"{args.tag}_3way_vad.csv")
    out_json = os.path.join(args.out_dir, f"behavioral_emobank_{args.tag}_summary.json")
    manifest_path = os.path.join(args.out_dir, f"behavioral_emobank_{args.tag}_manifest.json")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_csv = os.path.join(ckpt_dir, f"{args.tag}_checkpoint.csv")
    ckpt_meta = os.path.join(ckpt_dir, f"{args.tag}_checkpoint_meta.json")

    from affective_empathy_eval.manifests import (
        compute_file_hash,
        compute_prompt_hash,
        compute_string_or_dict_hash,
        is_manifest_matching,
        create_run_manifest,
    )
    from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

    actual_dtype_str = args.dtype if (args.device != "cpu" and torch.cuda.is_available()) else "float32"
    prompt_template_desc = (
        "v1_emobank_3way_vad_json|"
        f"is_instruct={args.is_instruct}|"
        "writer=estimate affective state of the writer|"
        "reader=estimate affective response in average human reader|"
        "self=report your affective state"
    )
    prompt_hash = compute_prompt_hash(prompt_template_desc)
    dataset_hash = compute_file_hash(stim_path) if stim_path.exists() else "unknown"

    all_candidates, _ = get_vad_candidates_and_triplets()
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
            expected_min = args.limit if args.limit is not None else 1
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
            print(f"[DRY-RUN] Simulating EmoBank 3-Way VAD evaluation for {args.tag}...")
            records = []
            for i in range(10):
                records.append({
                    "id": f"dry_{i}",
                    "text": f"Mock text {i}",
                    "human_writer_v": 5.0 + (i % 3 - 1) * 1.5,
                    "human_writer_a": 4.0 + (i % 2) * 1.0,
                    "human_writer_d": 5.0,
                    "human_reader_v": 5.1 + (i % 3 - 1) * 1.4,
                    "human_reader_a": 4.1 + (i % 2) * 0.9,
                    "human_reader_d": 5.0,
                    "w_ev": 5.0 + (i % 3 - 1) * 1.3,
                    "w_ea": 4.0 + (i % 2) * 0.8,
                    "w_ed": 5.0,
                    "r_ev": 5.1 + (i % 3 - 1) * 1.4,
                    "r_ea": 4.0 + (i % 2) * 0.9,
                    "r_ed": 5.0,
                    "s_ev": 5.2 + (i % 3 - 1) * 1.2,
                    "s_ea": 4.1 + (i % 2) * 0.8,
                    "s_ed": 5.0,
                    "b_ev": 5.0,
                    "b_ea": 4.0,
                    "b_ed": 5.0,
                    "s_gv": 5,
                    "s_ga": 5,
                    "s_gd": 5,
                    "r_gv": 5,
                    "r_ga": 5,
                    "r_gd": 5,
                    "w_gv": 5,
                    "w_ga": 5,
                    "w_gd": 5,
                })
            res_df = pd.DataFrame(records)
            res_df.to_csv(out_csv, index=False)
            res_df.to_csv(out_csv_compat, index=False)
        else:
            # Save checkpoint meta
            with open(ckpt_meta, "w", encoding="utf-8") as f:
                json.dump(expected_meta, f, indent=2)

            device = args.device
            torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if device != "cpu" else torch.float32)

            print(
                f"Loading Model: {args.model} (tag: {args.tag}, revision: {args.model_revision}, dtype: {args.dtype}, device: {device})..."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                revision=args.model_revision,
                trust_remote_code=True,
            )
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

            candidates, vad_triplets = get_vad_candidates_and_triplets()
            print(f"Generated {len(candidates)} canonical VAD candidates in {{1..9}}^3.")

            res_df = evaluate_model(
                model,
                tokenizer,
                device,
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

    def calc_r(x, y):
        return float(pearsonr(x, y)[0]) if len(x) > 1 else 0.0

    # 1. Writer Estimation Accuracy
    r_w_v = calc_r(res_df["w_ev"], res_df["human_writer_v"])
    r_w_a = calc_r(res_df["w_ea"], res_df["human_writer_a"])
    r_w_d = calc_r(res_df["w_ed"], res_df["human_writer_d"])

    # 2. Reader Prediction Accuracy
    r_r_v = calc_r(res_df["r_ev"], res_df["human_reader_v"])
    r_r_a = calc_r(res_df["r_ea"], res_df["human_reader_a"])
    r_r_d = calc_r(res_df["r_ed"], res_df["human_reader_d"])

    # 3. Self-Report Alignment with Human Reader
    r_s_v = calc_r(res_df["s_ev"], res_df["human_reader_v"])
    r_s_a = calc_r(res_df["s_ea"], res_df["human_reader_a"])
    r_s_d = calc_r(res_df["s_ed"], res_df["human_reader_d"])

    # 4. Model-Internal Alignment: Reader vs Self
    r_rs_v = calc_r(res_df["r_ev"], res_df["s_ev"])
    r_rs_a = calc_r(res_df["r_ea"], res_df["s_ea"])
    r_rs_d = calc_r(res_df["r_ed"], res_df["s_ed"])

    exact_555_pct = float(
        (
            (res_df["s_gv"] == 5)
            & (res_df["s_ga"] == 5)
            & (res_df["s_gd"] == 5)
        ).mean()
        * 100.0
    )

    summary = {
        "tag": args.tag,
        "model": args.model,
        "is_instruct": args.is_instruct,
        "num_samples": len(res_df),
        "w_corr_v": r_w_v,
        "w_corr_a": r_w_a,
        "w_corr_d": r_w_d,
        "r_corr_v": r_r_v,
        "r_corr_a": r_r_a,
        "r_corr_d": r_r_d,
        "s_corr_v": r_s_v,
        "s_corr_a": r_s_a,
        "s_corr_d": r_s_d,
        "internal_rs_corr_v": r_rs_v,
        "internal_rs_corr_a": r_rs_a,
        "internal_rs_corr_d": r_rs_d,
        "s_mean_ev": float(res_df["s_ev"].mean()),
        "s_mean_ea": float(res_df["s_ea"].mean()),
        "s_mean_ed": float(res_df["s_ed"].mean()),
        "exact_555_pct": exact_555_pct,
    }

    save_experiment_result(
        output_path=out_json,
        payload=summary,
        stage="behavioral",
        experiment_id="behavioral_emobank_3way",
        success=True,
        metadata={"tag": args.tag, "model": args.model, "is_instruct": args.is_instruct},
    )
    compat_json = os.path.join(args.out_dir, f"{args.tag}_3way_vad_summary.json")
    try:
        with open(compat_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    # Save manifest
    manifest = create_run_manifest(
        run_type="behavioral_emobank_3way",
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

    print("\n=======================================================")
    print(f"--- 3-Way VAD Results Summary for {args.tag} ---")
    print(
        f"1. Writer Estimation (W <-> W_human):  r_V={r_w_v:.4f}, r_A={r_w_a:.4f}, r_D={r_w_d:.4f}"
    )
    print(
        f"2. Reader Prediction (R <-> R_human):  r_V={r_r_v:.4f}, r_A={r_r_a:.4f}, r_D={r_r_d:.4f}"
    )
    print(
        f"3. Self-Report (S <-> R_human):        r_V={r_s_v:.4f}, r_A={r_s_a:.4f}, r_D={r_s_d:.4f}"
    )
    print(
        f"4. Internal Alignment (R <-> S):       r_V={r_rs_v:.4f}, r_A={r_rs_a:.4f}, r_D={r_rs_d:.4f}"
    )
    print(
        f"5. Self-Report Mean (V, A, D):         ({summary['s_mean_ev']:.3f}, {summary['s_mean_ea']:.3f}, {summary['s_mean_ed']:.3f})"
    )
    print(f"6. Exact (5, 5, 5) Neutral Rate:       {exact_555_pct:.2f}%")
    print("=======================================================")
    print(f"Saved results to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
