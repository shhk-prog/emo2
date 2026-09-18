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
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
from affective_empathy_eval.manifests import create_run_manifest


def build_candidates():
    candidates = []
    triplets = []
    for v, a, d in itertools.product(range(1, 10), range(1, 10), range(1, 10)):
        candidates.append(f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}')
        triplets.append((v, a, d))
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
    batch_size=243,
    checkpoint_path: Optional[str] = None,
):
    df = pd.read_csv(stimuli_path)
    if limit > 0:
        df = df.head(limit)

    print(f"Starting AIPsy 4-Split evaluation on {len(df)} stimuli...")

    tasks = ["writer", "reader", "self"]
    results = []
    processed_ids = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt_df = pd.read_csv(checkpoint_path)
            results = ckpt_df.to_dict("records")
            if "id" in ckpt_df.columns:
                processed_ids = set(ckpt_df["id"].tolist())
            print(f"Resuming from checkpoint: {len(processed_ids)} samples already completed.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")

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
        if checkpoint_path and (len(results) % 10 == 0):
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
        default="behavioral/results/aipsy_4split",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional limit for dry-run"
    )
    parser.add_argument("--batch-size", type=int, default=243)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    stim_path = Path(args.stimuli_path)
    if not stim_path.exists():
        fallback = Path("data/processed/aipsy_4split_all.csv")
        if fallback.exists():
            stim_path = fallback

    print("=" * 60)
    print(
        f"Evaluating Model: {args.model} (Tag: {args.tag}, Instruct: {args.is_instruct})"
    )
    print(f"Stimuli Path: {stim_path} (Device: {args.device})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device
    torch_dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32

    if device.startswith("cuda:"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(device)
    model.eval()

    candidates, vad_triplets = build_candidates()
    print(f"Generated {len(candidates)} VAD candidate triplets in {{1..9}}^3.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.tag}_aipsy_4split.csv")
    ckpt_csv = os.path.join(args.out_dir, f"{args.tag}_aipsy_4split_checkpoint.csv")

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
    )

    res_df.to_csv(out_csv, index=False)
    if os.path.exists(ckpt_csv):
        try:
            os.remove(ckpt_csv)
        except Exception:
            pass

    # Save manifest
    manifest = create_run_manifest(
        run_type="behavioral_aipsy_4split",
        model_name=args.model,
        config={
            "tag": args.tag,
            "is_instruct": args.is_instruct,
            "stimuli_path": str(stim_path),
            "limit": args.limit,
        },
        metadata={"num_samples": len(res_df)},
    )
    manifest.save(os.path.join(args.out_dir, f"{args.tag}_manifest.json"))

    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved {len(res_df)} evaluation results to {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
