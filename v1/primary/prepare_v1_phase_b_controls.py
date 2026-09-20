#!/usr/bin/env python3
"""
v1/primary/prepare_v1_phase_b_controls.py

Deterministic Preprocessing for V1 Phase B Semantic / contextual validity controls.
Extracts matched clinical (affective) and neutral pairs from AIPsy-Affect (aipsy_4split_all.csv)
and applies deterministic rule-based controlled perturbations (not LLM paraphrases):
  - Original Minimal Pair (Affective vs. Matched Neutral)
  - Paraphrase Invariance (fixed surface substitutions; fallback prefix if none match)
  - Word Shuffle (Syntax destroyed, 100% lexical tokens preserved)
  - Outcome Reversal (rule-based polarity inversion; fallback sentence adds substantial vocabulary
    and must not be over-interpreted as strong reversal evidence)

Output:
  v1/data/processed/v1_e5_semantic_controls.csv
"""

import argparse
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def shuffle_words(text: str, seed: int = 42) -> str:
    """E5 Word Shuffle: Randomizes word order to destroy compositional syntax."""
    words = text.split()
    if len(words) <= 3:
        return text
    rng = random.Random(seed)
    shuffled = words.copy()
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def generate_outcome_reversal(text: str) -> Tuple[str, str, bool]:
    """E5 Outcome Reversal: Reverses the affective resolution while retaining situational context."""
    replacements = [
        (r'\b(terminal|malignant|fatal)\b', 'benign'),
        (r'\b(failed|lost|ruined)\b', 'succeeded'),
        (r'\b(zero balance|emptied|stolen)\b', 'fully intact and safe'),
        (r'\b(died|passed away)\b', 'survived and stabilized'),
        (r'\b(rejected|denied)\b', 'approved and accepted'),
        (r'\b(guilty|convicted)\b', 'acquitted and exonerated'),
        (r'\b(won|succeeded|alive)\b', 'failed'),
        (r'\b(benign|safe|recovered)\b', 'critical')
    ]
    reversed_text = text
    changed = False
    used_method = ""
    for pat, rep in replacements:
        if re.search(pat, reversed_text, flags=re.IGNORECASE):
            reversed_text = re.sub(pat, rep, reversed_text, flags=re.IGNORECASE)
            changed = True
            used_method = f"regex:{pat}"
            break

    if not changed:
        if "lost" in text:
            reversed_text = text.replace("lost", "won")
            return reversed_text, "keyword_swap:lost->won", False
        elif "won" in text:
            reversed_text = text.replace("won", "lost")
            return reversed_text, "keyword_swap:won->lost", False
        else:
            reversed_text = text.rstrip(".") + ". Fortunately, everything was completely resolved without any harm."
            return reversed_text, "resolution_clause_fallback", True

    return reversed_text, used_method, False


def generate_paraphrase(text: str) -> Tuple[str, str, bool]:
    """E5 Paraphrase Invariance: Alters surface words while preserving situational meaning."""
    paraphrase_map = [
        ("The papers were spread across the floor", "Documents lay scattered across the room"),
        ("His daughter's college fund statement", "The savings portfolio meant for his daughter's education"),
        ("Zero balance", "The balance was completely depleted"),
        ("The retirement account had been emptied", "All money in the pension fund had been withdrawn"),
        ("She sat at her desk and calculated", "Sitting at her workstation, she tallied up the figures"),
        ("The investigator's report was three pages long", "A three-page dossier from the private investigator arrived"),
        ("There was no surgery", "The medical procedure had never existed"),
        ("The review board published its findings", "The evaluation panel released their formal determination"),
        ("The hospital had billed her for the full amount", "The healthcare facility invoiced her for the entire sum"),
        ("The lab was empty now", "The research laboratory stood completely deserted"),
        ("The kitchen counter still had the grocery list", "The shopping notes remained visible on the kitchen counter"),
    ]
    p_text = text
    applied_rules = []
    for orig, para in paraphrase_map:
        if orig in p_text:
            p_text = p_text.replace(orig, para)
            applied_rules.append(orig)

    if p_text == text:
        p_text = "It was documented that " + text[:1].lower() + text[1:]
        return p_text, "reporting_clause_fallback", True
    return p_text, f"lexical_substitution:{len(applied_rules)}_rules", False


def prepare_semantic_controls(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input AIPsy dataset not found at: {input_path}")

    df_raw = pd.read_csv(input_path)
    # Split clinical vs neutral
    condition_col = "split" if "split" in df_raw.columns else "condition"
    df_aff = df_raw[df_raw[condition_col].isin(["clinical", "affective"])].copy()
    df_neu = df_raw[df_raw[condition_col] == "neutral"].copy()

    # Match by pair_id
    merged = pd.merge(
        df_aff[["pair_id", "emotion", "domain", "text"]],
        df_neu[["pair_id", "text"]],
        on="pair_id",
        suffixes=("_aff", "_neu"),
    ).drop_duplicates(subset=["pair_id"]).reset_index(drop=True)

    print(f"Matched {len(merged)} clinical-neutral pairs from {input_path}.")

    records = []
    for idx, row in merged.iterrows():
        t_aff = str(row["text_aff"]).strip()
        t_neu = str(row["text_neu"]).strip()

        # Deterministic transformations
        t_para, para_method, para_fallback = generate_paraphrase(t_aff)
        t_shuf_aff = shuffle_words(t_aff, seed=seed + idx)
        t_shuf_neu = shuffle_words(t_neu, seed=seed + 1000 + idx)
        t_rev, rev_method, rev_fallback = generate_outcome_reversal(t_aff)

        records.append({
            "pair_id": row["pair_id"],
            "emotion": row.get("emotion", "unknown"),
            "domain": row.get("domain", 0.0),
            "text_original_affective": t_aff,
            "text_original_neutral": t_neu,
            "text_paraphrase_affective": t_para,
            "paraphrase_method": para_method,
            "paraphrase_fallback": para_fallback,
            "text_shuffled_affective": t_shuf_aff,
            "text_shuffled_neutral": t_shuf_neu,
            "text_reversed_affective": t_rev,
            "reversal_method": rev_method,
            "reversal_fallback": rev_fallback,
        })

    df_out = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df_out)} semantic controls -> {output_path}")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Prepare V1 Phase B Semantic Controls Dataset")
    parser.add_argument(
        "--input-path",
        type=str,
        default="v1/data/processed/aipsy_4split_all.csv",
        help="Path to source AIPsy-Affect dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="v1/data/processed/v1_e5_semantic_controls.csv",
        help="Path to output processed semantic controls CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for word shuffle")
    args = parser.parse_args()

    prepare_semantic_controls(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
