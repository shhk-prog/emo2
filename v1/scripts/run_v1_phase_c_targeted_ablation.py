#!/usr/bin/env python3
"""
V1 Phase C: Targeted Ablation & Double Dissociation Script (E6)
Evaluates whether Reader and Self recruit task-specific causal sub-circuits
through Targeted Ablation and Linear Mixed-Effects Model (LMM) interaction testing:
  Outcome ~ Task * SiteType + (1 | pair)
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from scipy import stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from affective_empathy_eval.intervention import PyTorchActivationPatcher


def format_prompt(tokenizer, text: str, task_type: str, is_instruct: bool) -> str:
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
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
    return (
        f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Output:\n"
    )


def build_vad_candidates():
    candidates = []
    vad_triplets = []
    for v in range(1, 10):
        for a in range(1, 10):
            for d in range(1, 10):
                cand_str = f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}'
                candidates.append(cand_str)
                vad_triplets.append((v, a, d))
    return candidates, np.array(vad_triplets)


@torch.no_grad()
def evaluate_expected_va(
    model, 
    tokenizer, 
    prompt: str, 
    candidates: List[str], 
    vad_triplets: np.ndarray,
    device: str = "cuda",
    sub_batch_size: int = 81
) -> Tuple[float, float]:
    """Computes expected Valence and Arousal using canonical Sequence-Likelihood Protocol."""
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
    log_ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=sub_batch_size,
        normalize_length=False,
    )
    ev = float(np.sum(probs * vad_triplets[:, 0]))
    ea = float(np.sum(probs * vad_triplets[:, 1]))
    return ev, ea



def fit_lmm_or_anova(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits Linear Mixed-Effects Model:
      Outcome ~ Task * SiteType + (1 | pair_id)
    Falls back to Repeated Measures Two-Way ANOVA via safe pair_id pivoting if statsmodels is not installed.
    """
    try:
        import statsmodels.formula.api as smf
        # Fit LMM with pair_id random intercepts
        model = smf.mixedlm("outcome ~ C(task, Treatment('Reader')) * C(site_type, Treatment('ReaderSite'))", 
                            df, groups=df["pair_id"])
        fit_res = model.fit()

        summary_dict = {
            "model_type": "Linear Mixed-Effects Model (LMM)",
            "p_interaction": float(fit_res.pvalues.get("C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite]", 1.0)),
            "coef_interaction": float(fit_res.params.get("C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite]", 0.0)),
            "p_task": float(fit_res.pvalues.get("C(task, Treatment('Reader'))[T.Self]", 1.0)),
            "p_site": float(fit_res.pvalues.get("C(site_type, Treatment('ReaderSite'))[T.SelfSite]", 1.0)),
            "t_stat_interaction": float(fit_res.tvalues.get("C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite]", 0.0)),
            "summary_text": str(fit_res.summary())
        }
        return summary_dict
    except Exception as e:
        # Fallback to Two-Way Repeated Measures ANOVA approximation using safe pivot matching
        print(f"Statsmodels LMM not available ({e}). Running Two-Way Repeated Measures ANOVA fallback with safe pair matching...")
        cell_means = df.groupby(["task", "site_type"])["outcome"].mean().to_dict()
        
        # Safely align across pair_id
        piv = df.pivot(index="pair_id", columns=["task", "site_type"], values="outcome").dropna()
        diff_reader_site = (piv[("Reader", "ReaderSite")] - piv[("Self", "ReaderSite")]).values
        diff_self_site = (piv[("Reader", "SelfSite")] - piv[("Self", "SelfSite")]).values

        t_stat, p_val = stats.ttest_rel(diff_reader_site, diff_self_site)
        return {
            "model_type": "Two-Way Repeated Measures Interaction (Paired t-test approximation)",
            "p_interaction": float(p_val),
            "t_stat_interaction": float(t_stat),
            "cell_means": {f"{k[0]}_{k[1]}": float(v) for k, v in cell_means.items()},
            "summary_text": f"Interaction Paired t-test: t={t_stat:.3f}, p={p_val:.4e}"
        }


def extract_layer_activation(model, tokenizer, prompt: str, layer: int, device: str) -> np.ndarray:
    """Extracts last token activation at specified layer using canonical encoding (add_special_tokens=False)."""
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], output_hidden_states=True)
        pos = encoded["attention_mask"].sum(dim=1) - 1
        vec = outputs.hidden_states[layer][0, pos[0].item(), :].detach().cpu().float().numpy()
    return vec



def main():
    parser = argparse.ArgumentParser(description="Run V1 Phase C: E6 Double Dissociation Script.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-prefix", type=str, default="qwen2.5_1.5b_instruct")
    parser.add_argument("--limit", type=int, default=20, help="Number of matched pairs to test (default: 20 for pilot; set 0 or use --full for all).")
    parser.add_argument("--full", action="store_true", help="Evaluate all available pairs.")
    parser.add_argument("--reader-layer", type=int, default=8, help="Layer identified as Reader-selective site.")
    parser.add_argument("--self-layer", type=int, default=20, help="Layer identified as Self-selective site.")
    parser.add_argument("--ablation-type", type=str, default="zero", choices=["zero", "matched_neutral", "neutral_mean"], 
                        help="Intervention type: zero out, replace with matched neutral stimulus activation, or mean neutral activation.")
    parser.add_argument("--no-split-eval", action="store_false", dest="split_eval", help="Disable 50/50 Discovery/Confirmation split evaluation (evaluate on all pairs directly).")
    parser.add_argument("--auto-discover", action="store_true", default=True, help="Automatically discover optimal ablation sites from Discovery split.")
    parser.add_argument("--discovery-sample-size", type=int, default=50, help="Number of pairs from Discovery split to use for auto-discovery screening (default: 50).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--sub-batch-size", type=int, default=81, help="Sub-batch size for 729-candidate evaluation to avoid OOM.")
    parser.add_argument("--force", action="store_true", help="Force re-run even if output files already exist.")
    parser.add_argument("--out-dir", type=str, default="v1/results/derived/v1_phase_c")
    args = parser.parse_args()

    if args.full:
        args.limit = 0

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    obs_out = os.path.join(model_dir, "e6_double_dissociation_observations.csv")
    lmm_out = os.path.join(model_dir, "e6_lmm_results.json")
    if not args.force and os.path.exists(obs_out) and os.path.exists(lmm_out):
        print(f"[SKIP] Results already exist in {model_dir}. Skipping execution (use --force to re-run).")
        return

    is_instruct = "instruct" in args.model_id.lower() or "chat" in args.model_id.lower() or "it" in args.model_id.lower()

    print(f"=== Starting V1 Phase C E6 Double Dissociation for Model: {args.model_id} ===")
    print(f"Reader-Site Layer: {args.reader_layer} | Self-Site Layer: {args.self_layer} | Ablation: {args.ablation_type} | Limit: {args.limit} (full={args.full})")

    candidates, vad_triplets = build_vad_candidates()

    # Load Paired Data
    aipsy_path = "v1/data/processed/aipsy_4split_all.csv"
    df_aipsy = pd.read_csv(aipsy_path)
    df_clin = df_aipsy[df_aipsy["split"] == "clinical"].copy()
    df_neut = df_aipsy[df_aipsy["split"] == "neutral"].copy()
    merged = pd.merge(df_clin, df_neut, on="pair_id", suffixes=('_aff', '_neu')).dropna(subset=["text_aff", "text_neu"]).reset_index(drop=True)

    if args.limit > 0:
        merged = merged.head(args.limit)

    n_pairs = len(merged)
    print(f"Loaded {n_pairs} matched clinical-neutral pairs.")

    # Load Model
    print("\nLoading Model and Tokenizer...")
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

    # Prompts
    prompts_r_aff = [format_prompt(tokenizer, t, "reader", is_instruct) for t in merged["text_aff"]]
    prompts_s_aff = [format_prompt(tokenizer, t, "self", is_instruct) for t in merged["text_aff"]]
    prompts_r_neu = [format_prompt(tokenizer, t, "reader", is_instruct) for t in merged["text_neu"]]
    prompts_s_neu = [format_prompt(tokenizer, t, "self", is_instruct) for t in merged["text_neu"]]

    # Pre-extract neutral activations if needed
    mean_neut_vecs = {}
    if args.ablation_type == "neutral_mean":
        print("Pre-computing mean neutral activations at target layers...")
        for l in [args.reader_layer, args.self_layer]:
            r_neu_vecs = [extract_layer_activation(model, tokenizer, p, l, args.device) for p in prompts_r_neu]
            mean_neut_vecs[l] = np.mean(r_neu_vecs, axis=0)

    # 1. Clean Baselines
    print("\n[1/3] Computing Clean Baselines...")
    clean_ev_r = [evaluate_expected_va(model, tokenizer, p, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)[0] for p in tqdm(prompts_r_aff, desc="Clean Reader")]
    clean_ev_s = [evaluate_expected_va(model, tokenizer, p, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)[0] for p in tqdm(prompts_s_aff, desc="Clean Self")]

    # 2. Targeted Ablation Cross-Intervention
    print(f"\n[2/3] Performing Targeted Ablations (ReaderSite vs SelfSite) using type='{args.ablation_type}'...")
    observations = []
    sites = [("ReaderSite", args.reader_layer), ("SelfSite", args.self_layer)]

    # Assign discovery / confirmation split tag using seed-fixed random group split (by pair_id)
    unique_pairs = merged["pair_id"].unique()
    rng = np.random.default_rng(42)
    perm_pairs = rng.permutation(unique_pairs)
    half_pt = len(perm_pairs) // 2
    disc_pair_set = set(perm_pairs[:half_pt])
    conf_pair_set = set(perm_pairs[half_pt:])
    split_tags = ["confirmation" if (args.split_eval and pid in conf_pair_set) else "discovery" for pid in merged["pair_id"]]

    # Auto-discovery of Reader-Site and Self-Site on Discovery split if requested
    if args.auto_discover and args.split_eval:
        print("\n[Auto-Discovery] Screening candidate layers on Discovery split...")
        disc_indices = [i for i, tag in enumerate(split_tags) if tag == "discovery"]
        # 代表層候補 (0.25, 0.50, 0.70, 0.85 深度付近)
        n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else 28
        candidate_layers = sorted(list(set([int(n_layers * d) for d in [0.25, 0.40, 0.55, 0.70, 0.85]])))
        best_r_layer = args.reader_layer
        best_r_spec = -999.0
        best_s_layer = args.self_layer
        best_s_spec = -999.0

        for l in candidate_layers:
            # Discovery サンプル (指定数、デフォルト最大 50 ペア) でスクリーニング
            d_size = getattr(args, "discovery_sample_size", 50)
            sub_idx = disc_indices[:min(d_size, len(disc_indices))]
            r_diffs, s_diffs = [], []
            for p_i in sub_idx:
                r_prompt = prompts_r_aff[p_i]
                s_prompt = prompts_s_aff[p_i]
                r_pos = tokenizer(r_prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1] - 1
                s_pos = tokenizer(s_prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1] - 1
                zero_vec = np.zeros(model.config.hidden_size, dtype=np.float32)

                with PyTorchActivationPatcher(model, l, zero_vec, position=r_pos, intervention_type="replace"):
                    ab_r, _ = evaluate_expected_va(model, tokenizer, r_prompt, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
                with PyTorchActivationPatcher(model, l, zero_vec, position=s_pos, intervention_type="replace"):
                    ab_s, _ = evaluate_expected_va(model, tokenizer, s_prompt, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)

                r_diffs.append(abs(ab_r - clean_ev_r[p_i]))
                s_diffs.append(abs(ab_s - clean_ev_s[p_i]))

            mean_r_diff = float(np.mean(r_diffs))
            mean_s_diff = float(np.mean(s_diffs))
            # Reader 特異度: Reader への影響 - Self への影響
            r_spec = mean_r_diff - mean_s_diff
            if r_spec > best_r_spec:
                best_r_spec = r_spec
                best_r_layer = l
            # Self 特異度: Self への影響 - Reader への影響
            s_spec = mean_s_diff - mean_r_diff
            if s_spec > best_s_spec:
                best_s_spec = s_spec
                best_s_layer = l

        args.reader_layer = best_r_layer
        args.self_layer = best_s_layer
        print(f"[Auto-Discovery Completed] Selected Reader-Site: Layer {args.reader_layer}, Self-Site: Layer {args.self_layer}")

    sites = [("ReaderSite", args.reader_layer), ("SelfSite", args.self_layer)]

    for site_name, target_layer in sites:
        for p_idx in tqdm(range(n_pairs), desc=f"Ablating {site_name} (Layer {target_layer})"):
            pair_id = merged.loc[p_idx, "pair_id"]
            split_tag = split_tags[p_idx]

            # Determine replacement vector
            if args.ablation_type == "zero":
                replace_vec_r = np.zeros(model.config.hidden_size, dtype=np.float32)
                replace_vec_s = np.zeros(model.config.hidden_size, dtype=np.float32)
            elif args.ablation_type == "matched_neutral":
                replace_vec_r = extract_layer_activation(model, tokenizer, prompts_r_neu[p_idx], target_layer, args.device)
                replace_vec_s = extract_layer_activation(model, tokenizer, prompts_s_neu[p_idx], target_layer, args.device)
            elif args.ablation_type == "neutral_mean":
                replace_vec_r = mean_neut_vecs[target_layer]
                replace_vec_s = mean_neut_vecs[target_layer]
            else:
                raise ValueError(f"Unknown ablation type: {args.ablation_type}")

            # Intervene on Reader Task
            encoded_r = tokenizer(prompts_r_aff[p_idx], add_special_tokens=False, return_tensors="pt").to(args.device)
            pos_r = encoded_r.input_ids.shape[1] - 1

            with PyTorchActivationPatcher(model, target_layer, replace_vec_r, patch_weight=1.0, position=pos_r, intervention_type="replace"):
                ablated_ev_r, _ = evaluate_expected_va(model, tokenizer, prompts_r_aff[p_idx], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
            
            # Intervene on Self Task
            encoded_s = tokenizer(prompts_s_aff[p_idx], add_special_tokens=False, return_tensors="pt").to(args.device)
            pos_s = encoded_s.input_ids.shape[1] - 1

            with PyTorchActivationPatcher(model, target_layer, replace_vec_s, patch_weight=1.0, position=pos_s, intervention_type="replace"):
                ablated_ev_s, _ = evaluate_expected_va(model, tokenizer, prompts_s_aff[p_idx], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)

            # Outcome: Absolute shift caused by ablation
            outcome_r = abs(ablated_ev_r - clean_ev_r[p_idx])
            outcome_s = abs(ablated_ev_s - clean_ev_s[p_idx])

            observations.append({"pair_id": pair_id, "task": "Reader", "site_type": site_name, "outcome": outcome_r, "eval_split": split_tag})
            observations.append({"pair_id": pair_id, "task": "Self", "site_type": site_name, "outcome": outcome_s, "eval_split": split_tag})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_obs = pd.DataFrame(observations)
    df_obs.to_csv(os.path.join(model_dir, "e6_double_dissociation_observations.csv"), index=False)

    # 3. Statistical Testing (LMM Interaction on Confirmation Split Only)
    print("\n[3/3] Running Linear Mixed-Effects Model Interaction Analysis on Confirmation Split...")
    eval_df = df_obs[df_obs["eval_split"] == "confirmation"] if args.split_eval else df_obs
    stat_results = fit_lmm_or_anova(eval_df)

    # Evaluate Double Dissociation Crossover Criterion
    cell_means = eval_df.groupby(["task", "site_type"])["outcome"].mean().unstack()
    mean_r_rs = float(cell_means.loc['Reader', 'ReaderSite'])
    mean_s_rs = float(cell_means.loc['Self', 'ReaderSite'])
    mean_r_ss = float(cell_means.loc['Reader', 'SelfSite'])
    mean_s_ss = float(cell_means.loc['Self', 'SelfSite'])

    has_crossover = (mean_r_rs > mean_s_rs) and (mean_s_ss > mean_r_ss)
    is_significant = stat_results["p_interaction"] < 0.05

    if is_significant and has_crossover:
        dissociation_status = "Evidence consistent with task-specific causal specialization"
        interpretation_msg = "Ablating Reader-site disproportionately impairs Reader prediction, while ablating Self-site disproportionately impairs Self-report, providing evidence consistent with task-specific causal specialization."
    elif is_significant and not has_crossover:
        dissociation_status = "Single Dissociation / Asymmetric Specialization (Significant Interaction without Full Crossover)"
        interpretation_msg = "Interaction is statistically significant, but the full crossover condition is not satisfied across both directions."
    else:
        dissociation_status = "No Significant Dissociation (Shared Causal Readout)"
        interpretation_msg = "Interaction did not reach significance, suggesting shared or distributed causal utilization."


    stat_results["dissociation_status"] = dissociation_status
    stat_results["has_crossover"] = bool(has_crossover)
    stat_results["interpretation"] = interpretation_msg
    stat_results["cell_means"] = {
        "Reader_ReaderSite": mean_r_rs,
        "Self_ReaderSite": mean_s_rs,
        "Reader_SelfSite": mean_r_ss,
        "Self_SelfSite": mean_s_ss
    }

    # Save Results
    with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
        json.dump({k: v for k, v in stat_results.items() if k != "summary_text"}, f, indent=2)

    # Summary Report
    summary_path = os.path.join(model_dir, "e6_double_dissociation_summary.md")

    with open(summary_path, "w") as f:
        f.write(f"# V1 Phase C: E6 Double Dissociation Report: {args.model_id}\n\n")
        f.write(f"- **Reader-Site**: Layer {args.reader_layer}\n")
        f.write(f"- **Self-Site**: Layer {args.self_layer}\n")
        f.write(f"- **Ablation Type**: `{args.ablation_type}`\n")
        f.write(f"- **Pairs Tested**: {len(eval_df['pair_id'].unique())} (Split-eval: {args.split_eval})\n\n")

        f.write("## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)\n\n")
        f.write("| Site \\ Task | Reader Task Impact | Self Task Impact |\n")
        f.write("|:---|:---:|:---:|\n")
        f.write(f"| **Reader-Site (Layer {args.reader_layer})** | **{mean_r_rs:.3f}** | {mean_s_rs:.3f} |\n")
        f.write(f"| **Self-Site (Layer {args.self_layer})** | {mean_r_ss:.3f} | **{mean_s_ss:.3f}** |\n\n")

        f.write("## 2. Statistical Interaction Test (Task x SiteType)\n\n")
        f.write(f"- **Model**: `{stat_results['model_type']}`\n")
        f.write(f"- **Interaction p-value**: **`p = {stat_results['p_interaction']:.4e}`**\n")
        f.write(f"- **Crossover Condition Satisfied**: `{has_crossover}`\n")
        f.write(f"- **Outcome Classification**: **{dissociation_status}**\n\n")

        f.write("```text\n")
        f.write(stat_results.get("summary_text", "") + "\n")
        f.write("```\n\n")

        f.write("## 3. Scientific Interpretation\n\n")
        f.write(f"> [!NOTE]\n")
        f.write(f"> **{dissociation_status}**: {interpretation_msg}\n")

    print(f"\nE6 Double Dissociation analysis complete! Results saved to {model_dir}/")
    print(f"Summary report written to {summary_path}")


if __name__ == "__main__":
    main()
