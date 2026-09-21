import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.geometry import compute_relative_depth
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_distribution_metrics,
    compute_emd_recovery_ratio,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.data import describe_loaded_frame
from affective_empathy_eval.manifests import (
    create_run_manifest,
    is_manifest_matching,
    compute_string_or_dict_hash,
    DEFAULT_CODE_VERSION,
)

from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    get_registry,
    resolve_models_from_args,
)
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
)
from affective_empathy_eval.statistics import compute_bootstrap_ci, paired_family_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V2-RQ4 Distribution Recovery Patching")
    parser.add_argument("--config", type=str, default="configs/v2_experiments.yaml", help="Path to V2 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if valid cached results exist")
    add_model_selection_args(parser)
    return parser.parse_args()


def run_recovery_patching_for_task(
    task: TaskType,
    fam_id: str,
    fam_cfg: Any,
    df: pd.DataFrame,
    device: str = "cpu",
    is_dry_run: bool = False,
    n_boot: int = 1000,
    model_base: Any = None,
    model_inst: Any = None,
    tok_base: Any = None,
    tok_inst: Any = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    train_indices: Optional[list[int]] = None,
    eval_indices: Optional[list[int]] = None,
    procrustes_pca_dim: int = 64,
) -> dict[str, Any]:
    """
    指定タスク（Reader または Self）において、Base 活性化の Instruct への層別パッチングを実施。
    刺激ごと（sample-wise）の EMD_VA および回復率 Recovery_{i,l} を算出し、Bootstrap CI を付与。
    """
    num_layers = min(fam_cfg.num_layers, 4) if is_dry_run else fam_cfg.num_layers
    depths = [compute_relative_depth(l, num_layers) for l in range(num_layers)]
    N = len(df)

    # trapezoid 統合関数 (numpy 2.0+ 対応)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

    if is_dry_run:
        rng = np.random.default_rng(42 if task == TaskType.SELF else 142)
        # サンプルごとの初期 EMD
        sample_initial_emds = [float(rng.uniform(1.2, 2.5)) for _ in range(N)]
        initial_emd_mean = float(np.mean(sample_initial_emds))

        layer_mean_emds = []
        layer_mean_ratios = []
        layer_ratios_ci = []
        layer_mean_ratios_plain = []
        layer_mean_ratios_aligned = []

        sample_ratios_direct_by_layer = {l: [] for l in range(num_layers)}
        sample_ratios_plain_by_layer = {l: [] for l in range(num_layers)}
        sample_ratios_aligned_by_layer = {l: [] for l in range(num_layers)}
        sample_delta_emds_plain_by_layer = {l: [] for l in range(num_layers)}

        # 中間層 (d ~ 0.5 - 0.7) で最も Base 分布への回復効果が高いシミュレーション
        peak_amp = 0.85 if task == TaskType.SELF else 0.70
        for l_idx, d in enumerate(depths):
            rec_base = float(peak_amp * np.exp(-((d - 0.6) ** 2) / 0.05))
            sample_ratios = [
                float(np.clip(rec_base + rng.normal(0, 0.08), 0.0, 1.0))
                for _ in range(N)
            ]
            sample_r_plain = [float(np.clip(r * 1.05 + rng.normal(0, 0.02), 0.0, 1.0)) for r in sample_ratios]
            sample_r_aligned = [float(np.clip(r * 0.95 + rng.normal(0, 0.02), 0.0, 1.0)) for r in sample_ratios]

            sample_ratios_direct_by_layer[l_idx] = sample_ratios
            sample_ratios_plain_by_layer[l_idx] = sample_r_plain
            sample_ratios_aligned_by_layer[l_idx] = sample_r_aligned

            sample_patched_emds = [
                sample_initial_emds[i] * (1.0 - sample_ratios[i])
                for i in range(N)
            ]
            sample_delta_emds_plain_by_layer[l_idx] = [
                sample_initial_emds[i] * sample_r_plain[i]
                for i in range(N)
            ]

            mean_emd = float(np.mean(sample_patched_emds))
            mean_ratio = float(np.mean(sample_ratios))
            pt_r, r_low, r_up = compute_bootstrap_ci(sample_ratios, n_boot=n_boot)

            layer_mean_emds.append(mean_emd)
            layer_mean_ratios.append(mean_ratio)
            layer_ratios_ci.append({"mean": pt_r, "ci_lower": r_low, "ci_upper": r_up})
            layer_mean_ratios_plain.append(float(np.mean(sample_r_plain)))
            layer_mean_ratios_aligned.append(float(np.mean(sample_r_aligned)))

        best_l_native = int(np.argmax(layer_mean_ratios))
        best_l_matched = int(np.argmax(layer_mean_ratios_plain))
        best_l_aligned = int(np.argmax(layer_mean_ratios_aligned))

        auc_recovery = float(trapz_func(layer_mean_ratios, depths))
        auc_recovery_plain = float(trapz_func(layer_mean_ratios_plain, depths))
        auc_recovery_aligned = float(trapz_func(layer_mean_ratios_aligned, depths))

        # Item 9: サンプル単位 Delta EMD の層平均および台形積分 AUC
        layer_mean_delta_emds_plain = [
            float(np.mean([sample_delta_emds_plain_by_layer[l][i] for i in range(N)]))
            for l in range(num_layers)
        ]
        auc_delta_emd_plain = float(trapz_func(layer_mean_delta_emds_plain, depths))

        # サンプルごとの層方向 AUC および各サンプルの best layer での指標
        sample_records = []
        for i in range(N):
            sample_r_direct = [sample_ratios_direct_by_layer[l][i] for l in range(num_layers)]
            sample_r_plain = [sample_ratios_plain_by_layer[l][i] for l in range(num_layers)]
            sample_r_aligned = [sample_ratios_aligned_by_layer[l][i] for l in range(num_layers)]
            sample_d_plain = [sample_delta_emds_plain_by_layer[l][i] for l in range(num_layers)]
            s_auc = float(trapz_func(sample_r_direct, depths))
            s_auc_plain = float(trapz_func(sample_r_plain, depths))
            s_auc_aligned = float(trapz_func(sample_r_aligned, depths))
            s_auc_delta_plain = float(trapz_func(sample_d_plain, depths))
            pair_id = str(df.iloc[i].get("pair_id", f"pair_{i}")) if "pair_id" in df.columns else f"pair_{i}"
            item_id = str(df.iloc[i].get("item_id", f"item_{i}")) if "item_id" in df.columns else f"item_{i}"
            sample_records.append({
                "family": fam_id,
                "task": task.value,
                "sample_idx": i,
                "pair_id": pair_id,
                "item_id": item_id,
                "initial_emd": sample_initial_emds[i],
                "best_layer": best_l_matched,  # Matched-Plain が Primary
                "best_depth": depths[best_l_matched],
                "best_layer_native": best_l_native,
                "best_layer_matched_plain": best_l_matched,
                "best_layer_aligned": best_l_aligned,
                "recovery_ratio": sample_r_direct[best_l_native],
                "recovery_ratio_matched_plain": sample_r_plain[best_l_matched],
                "recovery_ratio_aligned": sample_r_aligned[best_l_aligned],
                "auc_recovery": s_auc,
                "auc_recovery_matched_plain": s_auc_plain,
                "auc_recovery_aligned": s_auc_aligned,
                "delta_emd_matched_plain": sample_d_plain[best_l_matched],
                "auc_delta_emd_matched_plain": s_auc_delta_plain,
            })

        return {
            "task": task.value,
            "initial_emd_mean": initial_emd_mean,
            "sample_initial_emds": sample_initial_emds,
            "recovery_emd_va": layer_mean_emds,
            "recovery_ratios": layer_mean_ratios,  # Condition A: Direct Base -> Instruct
            "recovery_ratios_bootstrap_ci": layer_ratios_ci,
            "recovery_ratios_matched_plain": layer_mean_ratios_plain,
            "recovery_ratios_aligned": layer_mean_ratios_aligned,  # Condition B: Aligned Base -> Instruct
            "ci_lower": [max(0.0, r - 0.05) for r in layer_mean_ratios],
            "ci_upper": [min(1.0, r + 0.05) for r in layer_mean_ratios],
            "best_recovery_layer": best_l_matched,
            "best_recovery_depth": depths[best_l_matched],
            "best_recovery_layer_native": best_l_native,
            "best_recovery_layer_matched_plain": best_l_matched,
            "best_recovery_layer_aligned": best_l_aligned,
            "max_recovery_ratio": max(layer_mean_ratios),
            "max_recovery_ratio_matched_plain": max(layer_mean_ratios_plain),
            "max_recovery_ratio_aligned": max(layer_mean_ratios_aligned),
            "auc_recovery": auc_recovery,
            "auc_recovery_matched_plain": auc_recovery_plain,
            "auc_recovery_aligned": auc_recovery_aligned,
            "delta_emd_matched_plain_by_layer": layer_mean_delta_emds_plain,
            "auc_delta_emd_matched_plain": auc_delta_emd_plain,
            "primary_matched_plain": {
                "auc_recovery": auc_recovery_plain,
                "delta_emd_auc": auc_delta_emd_plain,
                "delta_emd_by_layer": layer_mean_delta_emds_plain,
                "layer_recovery_ratios": layer_mean_ratios_plain,
            },
            "secondary_peak_localization": {
                "best_recovery_layer": best_l_matched,
                "best_recovery_depth": depths[best_l_matched],
                "max_recovery_ratio": max(layer_mean_ratios_plain),
                "best_recovery_layer_native": best_l_native,
                "best_recovery_layer_aligned": best_l_aligned,
            },
            "sample_records": sample_records,
            "summary_by_control_type": {
                "direct_native": {
                    "max_recovery_ratio": max(layer_mean_ratios),
                    "auc_recovery": auc_recovery,
                    "layer_recovery_ratios": layer_mean_ratios
                },
                "matched_plain": {
                    "max_recovery_ratio": max(layer_mean_ratios_plain),
                    "auc_recovery": auc_recovery_plain,
                    "layer_recovery_ratios": layer_mean_ratios_plain
                },
                "aligned_procrustes": {
                    "max_recovery_ratio": max(layer_mean_ratios_aligned),
                    "auc_recovery": auc_recovery_aligned,
                    "layer_recovery_ratios": layer_mean_ratios_aligned
                }
            }
        }


    adapter_inst = get_model_adapter(model_inst)
    adapter_base = get_model_adapter(model_base)

    # Runtime assertion: Base と Instruct の hidden state 次元一致
    assert adapter_base.hidden_size == adapter_inst.hidden_size, (
        f"Hidden dimension mismatch between Base ({adapter_base.hidden_size}) and Instruct ({adapter_inst.hidden_size})"
    )

    candidates = build_va_candidates()


    # 1. 未介入時の出力確率分布の取得および Base/Instruct 活性化の全層一括抽出
    base_probs_list = []
    inst_probs_list = []
    base_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
    inst_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
    sample_initial_emds = []
    sample_initial_emds_plain = []

    model_base.eval()
    model_inst.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            text = str(row["text"])
            # Base (plain)
            p_base = build_prompt(text, task, format_type="plain")
            enc_b = encode_prompt_canonical(tok_base, p_base, device=device)
            anchors_b = find_semantic_anchors(enc_b["input_ids"][0].tolist(), tok_base, text)
            patch_pos_b = anchors_b["prompt_end"]

            # 1回の forward で全層の活性化を capture (Base)
            with ActivationHookManager(adapter_base) as hook_mgr_b:
                for l in range(num_layers):
                    hook_mgr_b.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=patch_pos_b,
                        key=f"layer_{l}",
                    )
                _ = model_base(**enc_b)
                for l in range(num_layers):
                    base_activations[l].append(hook_mgr_b.captured_activations[f"layer_{l}"].detach().clone())

            _, probs_b = compute_sequence_likelihoods_for_candidates(
                model=model_base, tokenizer=tok_base, prompt=p_base, candidates=candidates, device=device
            )
            base_probs_list.append(probs_b)

            # Instruct (chat)
            p_inst = build_prompt(text, task, format_type="chat", tokenizer=tok_inst)
            enc_i = encode_prompt_canonical(tok_inst, p_inst, device=device)
            anchors_i = find_semantic_anchors(enc_i["input_ids"][0].tolist(), tok_inst, text)
            patch_pos_i = anchors_i["prompt_end"]

            # 1回の forward で全層の活性化を capture (Instruct)
            with ActivationHookManager(adapter_inst) as hook_mgr_i_cap:
                for l in range(num_layers):
                    hook_mgr_i_cap.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=patch_pos_i,
                        key=f"layer_{l}",
                    )
                _ = model_inst(**enc_i)
                for l in range(num_layers):
                    inst_activations[l].append(hook_mgr_i_cap.captured_activations[f"layer_{l}"].detach().clone())

            _, probs_i = compute_sequence_likelihoods_for_candidates(
                model=model_inst, tokenizer=tok_inst, prompt=p_inst, candidates=candidates, device=device
            )
            inst_probs_list.append(probs_i)

            # Instruct (plain) baseline for matched-plain control
            p_inst_plain_clean = build_prompt(text, task, format_type="plain")
            _, probs_i_plain = compute_sequence_likelihoods_for_candidates(
                model=model_inst, tokenizer=tok_inst, prompt=p_inst_plain_clean, candidates=candidates, device=device
            )

            # サンプルごとの初期 EMD
            init_metrics = compute_distribution_metrics(probs_i, probs_b)
            sample_initial_emds.append(init_metrics["emd_va"])
            init_metrics_plain = compute_distribution_metrics(probs_i_plain, probs_b)
            sample_initial_emds_plain.append(init_metrics_plain["emd_va"])

    initial_emd_mean = float(np.mean(sample_initial_emds))

    # 2. 各層への Base 活性化パッチング (Direct vs Aligned Procrustes)
    layer_mean_emds = []
    layer_mean_ratios = []
    layer_ratios_ci = []
    layer_mean_ratios_plain = []
    layer_mean_ratios_aligned = []
    sample_ratios_by_layer = {l: {} for l in range(num_layers)}
    sample_ratios_plain_by_layer = {l: {} for l in range(num_layers)}
    sample_ratios_aligned_by_layer = {l: {} for l in range(num_layers)}
    sample_delta_emds_plain_by_layer = {l: {} for l in range(num_layers)}

    # Train / Eval 分割 (Procrustes alignment 学習に評価サンプルを含めない: seeded group/permutation split)
    if train_indices is None or eval_indices is None:
        rng_split = np.random.default_rng(seed)
        if "pair_id" in df.columns:
            unique_pairs = list(df["pair_id"].unique())
            rng_split.shuffle(unique_pairs)
            n_train_pairs = max(1, int(train_ratio * len(unique_pairs)))
            train_pairs = set(unique_pairs[:n_train_pairs])
            eval_pairs = set(unique_pairs[n_train_pairs:])
            train_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in train_pairs]
            eval_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in eval_pairs]
            if len(train_indices) == 0 or len(eval_indices) == 0:
                raise ValueError("Independent evaluation split could not be constructed.")
            assert train_pairs.isdisjoint(eval_pairs), "Train and eval pair sets must be disjoint!"
        else:
            perm = list(rng_split.permutation(N))
            n_train = max(1, int(train_ratio * N))
            train_indices = perm[:n_train]
            eval_indices = perm[n_train:]
            if len(train_indices) == 0 or len(eval_indices) == 0:
                raise ValueError("Independent evaluation split could not be constructed.")
            assert set(train_indices).isdisjoint(set(eval_indices)), "Train and eval indices must be disjoint!"

    for l in range(num_layers):
        # Train-only PCA -> Procrustes (Rank-deficiency 防止: k = min(procrustes_pca_dim, n_train - 1, hidden_dim))
        H_b_train = torch.cat([base_activations[l][idx].squeeze() for idx in train_indices], dim=0).view(len(train_indices), -1).cpu().float().numpy()
        H_i_train = torch.cat([inst_activations[l][idx].squeeze() for idx in train_indices], dim=0).view(len(train_indices), -1).cpu().float().numpy()

        mu_b = np.mean(H_b_train, axis=0, keepdims=True)
        mu_i = np.mean(H_i_train, axis=0, keepdims=True)
        X_b = H_b_train - mu_b
        X_i = H_i_train - mu_i

        hidden_dim = H_b_train.shape[1]
        n_train_pts = len(train_indices)
        k = min(int(procrustes_pca_dim), max(1, n_train_pts - 1), hidden_dim)

        # 共通 PCA: 結合データ X_joint = [X_b; X_i] から共通主成分基底 V_k (D, k) を抽出
        X_joint = np.vstack([X_b, X_i])
        _, _, Vh_joint = np.linalg.svd(X_joint, full_matrices=False)
        V_k = Vh_joint[:k, :].T  # (D, k)

        # k 次元部分空間に射影して直交 Procrustes を学習
        Z_b = X_b @ V_k  # (n_train, k)
        Z_i = X_i @ V_k  # (n_train, k)
        M_k = Z_b.T @ Z_i  # (k, k)
        U_k, _, Vh_k = np.linalg.svd(M_k, full_matrices=False)
        R_k = U_k @ Vh_k  # (k, k)

        V_k_torch = torch.tensor(V_k, dtype=torch.float32, device=device)
        R_k_torch = torch.tensor(R_k, dtype=torch.float32, device=device)
        mu_b_torch = torch.tensor(mu_b, dtype=torch.float32, device=device)
        mu_i_torch = torch.tensor(mu_i, dtype=torch.float32, device=device)

        sample_patched_emds = []
        sample_ratios = []
        sample_ratios_plain = []
        sample_ratios_aligned = []

        for i in eval_indices:
            row = df.iloc[i]
            text = str(row["text"])
            p_inst = build_prompt(text, task, format_type="chat", tokenizer=tok_inst)
            enc_i = encode_prompt_canonical(tok_inst, p_inst, device=device)
            anchors_i = find_semantic_anchors(enc_i["input_ids"][0].tolist(), tok_inst, text)
            patch_pos_i = anchors_i["prompt_end"]

            base_act_tensor = base_activations[l][i].to(device)

            # Condition A: Direct Base -> Instruct patch
            with ActivationHookManager(adapter_inst) as hook_mgr_i:
                hook_mgr_i.register_patch_hook(
                    layer_idx=l,
                    patch_tensor=base_act_tensor,
                    token_indices=patch_pos_i,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                _, probs_patched = compute_sequence_likelihoods_for_candidates(
                    model=model_inst, tokenizer=tok_inst, prompt=p_inst, candidates=candidates, device=device
                )

            # サンプル単位の EMD と回復率
            p_emd = compute_distribution_metrics(probs_patched, base_probs_list[i])["emd_va"]
            ratio = compute_emd_recovery_ratio(sample_initial_emds[i], p_emd)
            sample_patched_emds.append(p_emd)
            sample_ratios.append(ratio)
            sample_ratios_by_layer[l][i] = ratio

            # Condition B: Aligned Base -> Instruct patch (PCA-Procrustes on subspace, identity on residual)
            base_flat = base_act_tensor.view(1, -1).float()
            x = base_flat - mu_b_torch
            z = x @ V_k_torch
            x_sub = z @ V_k_torch.T
            x_resid = x - x_sub
            z_aligned = z @ R_k_torch
            base_aligned = z_aligned @ V_k_torch.T + x_resid + mu_i_torch
            base_aligned_tensor = base_aligned.view_as(base_act_tensor)

            with ActivationHookManager(adapter_inst) as hook_mgr_aligned:
                hook_mgr_aligned.register_patch_hook(
                    layer_idx=l,
                    patch_tensor=base_aligned_tensor,
                    token_indices=patch_pos_i,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                _, probs_aligned = compute_sequence_likelihoods_for_candidates(
                    model=model_inst, tokenizer=tok_inst, prompt=p_inst, candidates=candidates, device=device
                )
            p_emd_aligned = compute_distribution_metrics(probs_aligned, base_probs_list[i])["emd_va"]
            ratio_aligned = compute_emd_recovery_ratio(sample_initial_emds[i], p_emd_aligned)
            sample_ratios_aligned.append(ratio_aligned)
            sample_ratios_aligned_by_layer[l][i] = ratio_aligned

            # Prompt-format control: Base plain -> Instruct matched-plain へのパッチング
            p_inst_plain = build_prompt(text, task, format_type="plain")
            enc_ip = encode_prompt_canonical(tok_inst, p_inst_plain, device=device)
            anchors_ip = find_semantic_anchors(enc_ip["input_ids"][0].tolist(), tok_inst, text)
            patch_pos_ip = anchors_ip["prompt_end"]
            with ActivationHookManager(adapter_inst) as hook_mgr_ip:
                hook_mgr_ip.register_patch_hook(
                    layer_idx=l,
                    patch_tensor=base_act_tensor,
                    token_indices=patch_pos_ip,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                _, probs_patched_plain = compute_sequence_likelihoods_for_candidates(
                    model=model_inst, tokenizer=tok_inst, prompt=p_inst_plain, candidates=candidates, device=device
                )
            p_emd_plain = compute_distribution_metrics(probs_patched_plain, base_probs_list[i])["emd_va"]
            ratio_plain = compute_emd_recovery_ratio(sample_initial_emds_plain[i], p_emd_plain)
            sample_ratios_plain.append(ratio_plain)
            sample_ratios_plain_by_layer[l][i] = ratio_plain
            # Item 9: サンプル単位の真の Delta EMD = initial_emd_i - patched_emd_i
            sample_delta_emds_plain_by_layer[l][i] = float(sample_initial_emds_plain[i] - p_emd_plain)

        mean_emd = float(np.mean(sample_patched_emds))
        mean_ratio = float(np.mean(sample_ratios))
        pt_r, r_low, r_up = compute_bootstrap_ci(sample_ratios, n_boot=n_boot)

        layer_mean_emds.append(mean_emd)
        layer_mean_ratios.append(mean_ratio)
        layer_ratios_ci.append({"mean": pt_r, "ci_lower": r_low, "ci_upper": r_up})
        layer_mean_ratios_plain.append(float(np.mean(sample_ratios_plain)))
        layer_mean_ratios_aligned.append(float(np.mean(sample_ratios_aligned)))

    best_l_native = int(np.argmax(layer_mean_ratios))
    best_l_matched = int(np.argmax(layer_mean_ratios_plain))
    best_l_aligned = int(np.argmax(layer_mean_ratios_aligned))

    auc_recovery = float(trapz_func(layer_mean_ratios, depths))
    auc_recovery_plain = float(trapz_func(layer_mean_ratios_plain, depths))
    auc_recovery_aligned = float(trapz_func(layer_mean_ratios_aligned, depths))

    # Item 9: サンプル単位 Delta EMD の層平均および台形積分 AUC
    layer_mean_delta_emds_plain = [
        float(np.mean([sample_delta_emds_plain_by_layer[l][idx] for idx in eval_indices]))
        for l in range(num_layers)
    ]
    auc_delta_emd_plain = float(trapz_func(layer_mean_delta_emds_plain, depths))

    # サンプルごとのレコード構築
    sample_records = []
    for i in eval_indices:
        sample_r_direct = [sample_ratios_by_layer[l][i] for l in range(num_layers)]
        sample_r_plain = [sample_ratios_plain_by_layer[l][i] for l in range(num_layers)]
        sample_r_aligned = [sample_ratios_aligned_by_layer[l][i] for l in range(num_layers)]
        sample_d_plain = [sample_delta_emds_plain_by_layer[l][i] for l in range(num_layers)]
        s_auc = float(trapz_func(sample_r_direct, depths))
        s_auc_plain = float(trapz_func(sample_r_plain, depths))
        s_auc_aligned = float(trapz_func(sample_r_aligned, depths))
        s_auc_delta_plain = float(trapz_func(sample_d_plain, depths))
        pair_id = str(df.iloc[i].get("pair_id", f"pair_{i}")) if "pair_id" in df.columns else f"pair_{i}"
        item_id = str(df.iloc[i].get("item_id", f"item_{i}")) if "item_id" in df.columns else f"item_{i}"
        sample_records.append({
            "family": fam_id,
            "task": task.value,
            "sample_idx": i,
            "pair_id": pair_id,
            "item_id": item_id,
            "initial_emd": sample_initial_emds[i],
            "best_layer": best_l_matched,  # Matched-Plain が Primary
            "best_depth": depths[best_l_matched],
            "best_layer_native": best_l_native,
            "best_layer_matched_plain": best_l_matched,
            "best_layer_aligned": best_l_aligned,
            "recovery_ratio": sample_r_direct[best_l_native],
            "recovery_ratio_matched_plain": sample_r_plain[best_l_matched],
            "recovery_ratio_aligned": sample_r_aligned[best_l_aligned],
            "auc_recovery": s_auc,
            "auc_recovery_matched_plain": s_auc_plain,
            "auc_recovery_aligned": s_auc_aligned,
            "delta_emd_matched_plain": sample_d_plain[best_l_matched],
            "auc_delta_emd_matched_plain": s_auc_delta_plain,
        })

    return {
        "task": task.value,
        "initial_emd_mean": initial_emd_mean,
        "sample_initial_emds": sample_initial_emds,
        "recovery_emd_va": layer_mean_emds,
        "recovery_ratios": layer_mean_ratios,
        "recovery_ratios_bootstrap_ci": layer_ratios_ci,
        "recovery_ratios_matched_plain": layer_mean_ratios_plain,
        "recovery_ratios_aligned": layer_mean_ratios_aligned,
        "auc_recovery": auc_recovery,
        "auc_recovery_matched_plain": auc_recovery_plain,
        "auc_recovery_aligned": auc_recovery_aligned,
        "delta_emd_matched_plain_by_layer": layer_mean_delta_emds_plain,
        "auc_delta_emd_matched_plain": auc_delta_emd_plain,
        # Item 10: Primary (Matched-Plain AUC) と Secondary (Peak localization) の明確な分離
        "primary_matched_plain": {
            "auc_recovery": auc_recovery_plain,
            "delta_emd_auc": auc_delta_emd_plain,
            "delta_emd_by_layer": layer_mean_delta_emds_plain,
            "layer_recovery_ratios": layer_mean_ratios_plain,
        },
        "secondary_peak_localization": {
            "best_recovery_layer": best_l_matched,
            "best_recovery_depth": depths[best_l_matched],
            "max_recovery_ratio": max(layer_mean_ratios_plain),
            "best_recovery_layer_native": best_l_native,
            "best_recovery_layer_aligned": best_l_aligned,
        },
        "best_recovery_layer": best_l_matched,
        "best_recovery_depth": depths[best_l_matched],
        "best_recovery_layer_native": best_l_native,
        "best_recovery_layer_matched_plain": best_l_matched,
        "best_recovery_layer_aligned": best_l_aligned,
        "max_recovery_ratio": layer_mean_ratios[best_l_native],
        "max_recovery_ratio_matched_plain": max(layer_mean_ratios_plain),
        "max_recovery_ratio_aligned": max(layer_mean_ratios_aligned),
        "sample_records": sample_records,
        "summary_by_control_type": {
            "direct_native": {
                "max_recovery_ratio": max(layer_mean_ratios),
                "auc_recovery": auc_recovery,
                "layer_recovery_ratios": layer_mean_ratios,
            },
            "matched_plain": {
                "max_recovery_ratio": max(layer_mean_ratios_plain),
                "auc_recovery": auc_recovery_plain,
                "layer_recovery_ratios": layer_mean_ratios_plain,
                "delta_emd_by_layer": layer_mean_delta_emds_plain,
                "auc_delta_emd": auc_delta_emd_plain,
            },
            "aligned_procrustes": {
                "max_recovery_ratio": max(layer_mean_ratios_aligned),
                "auc_recovery": auc_recovery_aligned,
                "layer_recovery_ratios": layer_mean_ratios_aligned,
            },
        },
    }




def run_recovery_patching_for_family(
    fam_id: str,
    fam_cfg: Any,
    df: pd.DataFrame,
    device: str = "cpu",
    is_dry_run: bool = False,
    n_boot: int = 1000,
    seed: int = 42,
    train_ratio: float = 0.7,
    procrustes_pca_dim: int = 64,
) -> dict[str, Any]:
    """
    1ファミリーについて Base/Instruct モデルをロードし、Reader と Self の両タスクで回復パッチングを実行
    """
    logger.info(f"=== Starting RQ4 Recovery Patching for Family: {fam_id} ===")
    num_layers = min(fam_cfg.num_layers, 4) if is_dry_run else fam_cfg.num_layers
    depths = [compute_relative_depth(l, num_layers) for l in range(num_layers)]
    N = len(df)

    # 評価サンプルと Procrustes 学習サンプルの分割をファミリー単位で固定 (Reader と Self で厳密に一致)
    rng_split = np.random.default_rng(seed)
    if "pair_id" in df.columns:
        unique_pairs = list(df["pair_id"].unique())
        rng_split.shuffle(unique_pairs)
        n_train_pairs = max(1, int(train_ratio * len(unique_pairs)))
        train_pairs = set(unique_pairs[:n_train_pairs])
        eval_pairs = set(unique_pairs[n_train_pairs:])
        train_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in train_pairs]
        eval_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in eval_pairs]
        if len(train_indices) == 0 or len(eval_indices) == 0:
            raise ValueError("Independent evaluation split could not be constructed.")
        assert train_pairs.isdisjoint(eval_pairs), "Train and eval pair sets must be disjoint!"
    else:
        perm = list(rng_split.permutation(N))
        n_train = max(1, int(train_ratio * N))
        train_indices = perm[:n_train]
        eval_indices = perm[n_train:]
        if len(train_indices) == 0 or len(eval_indices) == 0:
            raise ValueError("Independent evaluation split could not be constructed.")
        assert set(train_indices).isdisjoint(set(eval_indices)), "Train and eval indices must be disjoint!"

    if is_dry_run:
        model_base, model_inst, tok_base, tok_inst = None, None, None, None
    else:
        spec_base = fam_cfg.get_model_spec("base")
        spec_inst = fam_cfg.get_model_spec("instruct")
        torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
        logger.info(f"Loading Base model: {spec_base.model_id} (revision={spec_base.revision})...")
        tok_base = AutoTokenizer.from_pretrained(
            spec_base.model_id,
            revision=spec_base.revision,
        )
        model_base = AutoModelForCausalLM.from_pretrained(
            spec_base.model_id,
            revision=spec_base.revision,
            torch_dtype=torch_dtype,
            device_map=device if "cuda" in device else None,
        )
        logger.info(f"Loading Instruct model: {spec_inst.model_id} (revision={spec_inst.revision})...")
        tok_inst = AutoTokenizer.from_pretrained(
            spec_inst.model_id,
            revision=spec_inst.revision,
        )
        model_inst = AutoModelForCausalLM.from_pretrained(
            spec_inst.model_id,
            revision=spec_inst.revision,
            torch_dtype=torch_dtype,
            device_map=device if "cuda" in device else None,
        )

    # 1. Reader タスクでの回復パッチング
    logger.info(f"[{fam_id}] Running Recovery Patching for Reader task...")
    res_reader = run_recovery_patching_for_task(
        task=TaskType.READER,
        fam_id=fam_id,
        fam_cfg=fam_cfg,
        df=df,
        device=device,
        is_dry_run=is_dry_run,
        n_boot=n_boot,
        model_base=model_base,
        model_inst=model_inst,
        tok_base=tok_base,
        tok_inst=tok_inst,
        seed=seed,
        train_ratio=train_ratio,
        train_indices=train_indices,
        eval_indices=eval_indices,
        procrustes_pca_dim=procrustes_pca_dim,
    )

    # 2. Self タスクでの回復パッチング
    logger.info(f"[{fam_id}] Running Recovery Patching for Self task...")
    res_self = run_recovery_patching_for_task(
        task=TaskType.SELF,
        fam_id=fam_id,
        fam_cfg=fam_cfg,
        df=df,
        device=device,
        is_dry_run=is_dry_run,
        n_boot=n_boot,
        model_base=model_base,
        model_inst=model_inst,
        tok_base=tok_base,
        tok_inst=tok_inst,
        seed=seed,
        train_ratio=train_ratio,
        train_indices=train_indices,
        eval_indices=eval_indices,
        procrustes_pca_dim=procrustes_pca_dim,
    )

    # メモリ解放
    if model_base is not None:
        del model_base, model_inst, tok_base, tok_inst
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # タスク間比較 (Self vs Reader 回復率の差)
    # 1. Primary: Matched-Plain
    diff_max_ratio_matched = float(
        res_self["max_recovery_ratio_matched_plain"]
        - res_reader["max_recovery_ratio_matched_plain"]
    )
    diff_auc_matched = float(
        res_self["auc_recovery_matched_plain"]
        - res_reader["auc_recovery_matched_plain"]
    )

    # 2. Secondary: Direct Native-Chat
    diff_max_ratio_native = float(
        res_self["max_recovery_ratio"]
        - res_reader["max_recovery_ratio"]
    )
    diff_auc_native = float(
        res_self["auc_recovery"]
        - res_reader["auc_recovery"]
    )

    diff_best_depth = float(res_self["best_recovery_depth"] - res_reader["best_recovery_depth"])

    sample_records = res_reader.get("sample_records", []) + res_self.get("sample_records", [])

    return {
        "family_id": fam_id,
        "num_layers": num_layers,
        "relative_depths": depths,
        "reader": res_reader,
        "self": res_self,
        "sample_records": sample_records,
        "task_comparison": {
            "primary_matched_plain": {
                "diff_max_recovery_self_vs_reader": diff_max_ratio_matched,
                "diff_auc_recovery_self_vs_reader": diff_auc_matched,
                "self_exceeds_reader": bool(diff_max_ratio_matched > 0),
            },
            "secondary_native_chat": {
                "diff_max_recovery_self_vs_reader": diff_max_ratio_native,
                "diff_auc_recovery_self_vs_reader": diff_auc_native,
                "self_exceeds_reader": bool(diff_max_ratio_native > 0),
            },
            # Generic keys explicitly point to primary matched-plain for clarity and backwards compatibility
            "diff_max_recovery_self_vs_reader": diff_max_ratio_matched,
            "diff_auc_recovery_self_vs_reader": diff_auc_matched,
            "diff_best_recovery_depth_self_vs_reader": diff_best_depth,
            "self_exceeds_reader": bool(diff_max_ratio_matched > 0),
        },
    }


def main():
    args = parse_args()
    logger.info(f"Starting V2-RQ4 Distribution Recovery Patching (dry_run={args.dry_run})")

    with open(args.config, "r", encoding="utf-8") as f:
        v2_config = yaml.safe_load(f)

    target_models = resolve_models_from_args(args, Path(args.models_config))
    raw_dir = Path(v2_config["output"]["raw_dir"])
    derived_dir = Path(v2_config["output"]["derived_dir"])
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(v2_config["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V2-RQ4 dataset", v2_config["dataset"]["path"]))
    if args.dry_run:
        df = df.head(32).copy()
        logger.info(f"[DRY-RUN] Scaled down dataset to N={len(df)} for fast smoke testing.")
    elif args.max_samples is not None:
        df = df.iloc[:args.max_samples].copy()
        logger.info(f"Applied max_samples={args.max_samples}: n_rows={len(df)}")

    n_boot = 10 if args.dry_run else (v2_config.get("bootstrap", {}).get("n_boot") or v2_config.get("statistics", {}).get("n_boot", 1000))
    all_recovery_results = {}
    all_sample_records: List[Dict[str, Any]] = []

    for fam_id, fam_cfg in target_models.items():
        out_path = raw_dir / f"v2_recovery_{fam_id}.json"
        modular_rq4_path = raw_dir / f"v2_rq4_recovery_patching_{fam_id}.json"
        manifest_path = raw_dir / f"manifest_recovery_{fam_id}.json"

        spec_base = fam_cfg.get_model_spec("base")
        spec_inst = fam_cfg.get_model_spec("instruct")
        manifest_config = {
            "v2_config": v2_config,
            "family_id": fam_id,
            "family_name": fam_cfg.family_name,
            "base_model_id": spec_base.model_id if spec_base else None,
            "base_revision": spec_base.revision if spec_base else None,
            "instruct_model_id": spec_inst.model_id if spec_inst else None,
            "instruct_revision": spec_inst.revision if spec_inst else None,
            "dtype": getattr(fam_cfg, "inference_dtype", "bfloat16"),
            "dataset_path": str(v2_config["dataset"]["path"]),
            "seed": v2_config.get("seed", 42),
            "n_boot": n_boot,
            "max_samples": args.max_samples,
            "dry_run": bool(args.dry_run),
        }

        from affective_empathy_eval.manifests import (
            is_manifest_matching,
            compute_file_hash,
            compute_string_or_dict_hash,
        )
        from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

        ds_path_p = Path(v2_config["dataset"]["path"])
        exp_ds_hash = compute_file_hash(ds_path_p) if ds_path_p.exists() else compute_string_or_dict_hash(str(ds_path_p))
        exp_cfg_hash = compute_string_or_dict_hash(manifest_config)

        if not args.force and not args.dry_run and manifest_path.exists():
            manifest_valid = is_manifest_matching(
                manifest_path=str(manifest_path),
                expected_config_hash=exp_cfg_hash,
                expected_dataset_hash=exp_ds_hash,
                expected_dry_run=False,
            )
            if manifest_valid:
                target_check = modular_rq4_path if modular_rq4_path.exists() else out_path
                if is_experiment_completed(str(target_check), manifest_path=str(manifest_path)):
                    try:
                        read_p = modular_rq4_path if modular_rq4_path.exists() else out_path
                        with open(read_p, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        logger.info(f"Loaded existing results matching manifest for {fam_id} from {read_p}. Skipping computation.")
                        all_recovery_results[fam_id] = cached
                        fam_csv = raw_dir / f"v2_recovery_samples_{fam_id}.csv"
                        if fam_csv.exists():
                            df_cached_samples = pd.read_csv(fam_csv)
                            all_sample_records.extend(df_cached_samples.to_dict(orient="records"))
                        continue
                    except Exception as e:
                        logger.warning(f"Cache check failed for {fam_id}: {e}")

        train_ratio = float(v2_config.get("dataset", {}).get("train_ratio", 0.7))
        logger.info(f"--- Running Recovery Patching for Family: {fam_id} (train_ratio={train_ratio}) ---")
        procrustes_pca_dim = int(v2_config.get("rq4", {}).get("procrustes_pca_dim", 64))
        res = run_recovery_patching_for_family(
            fam_id=fam_id,
            fam_cfg=fam_cfg,
            df=df,
            device=args.device,
            is_dry_run=args.dry_run,
            n_boot=n_boot,
            seed=v2_config.get("seed", 42),
            train_ratio=train_ratio,
            procrustes_pca_dim=procrustes_pca_dim,
        )
        res["dry_run"] = bool(args.dry_run)
        all_recovery_results[fam_id] = res

        if "sample_records" in res and res["sample_records"]:
            df_fam_samples = pd.DataFrame(res["sample_records"])
            fam_csv_path = raw_dir / f"v2_recovery_samples_{fam_id}.csv"
            df_fam_samples.to_csv(fam_csv_path, index=False)
            all_sample_records.extend(res["sample_records"])
            logger.info(f"Saved family sample recovery records to {fam_csv_path}")

        out_path = raw_dir / f"v2_recovery_{fam_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        save_experiment_result(
            output_path=str(modular_rq4_path),
            payload=res,
            stage="v2",
            experiment_id="v2_rq4_recovery_patching",
            status="success",
            success=True,
            metadata={"family_id": fam_id, "dry_run": bool(args.dry_run)},
        )
        logger.info(f"Saved family recovery result to {out_path} and {modular_rq4_path}")

        # Manifest 保存
        manifest = create_run_manifest(
            run_type="v2_recovery",
            model_name=fam_cfg.family_name,
            config=manifest_config,
            metadata={
                "best_self_layer": res["self"]["best_recovery_layer"],
                "best_reader_layer": res["reader"]["best_recovery_layer"],
                "primary_self_auc_matched": res["self"].get("auc_recovery_matched_plain"),
                "primary_reader_auc_matched": res["reader"].get("auc_recovery_matched_plain"),
            },
            dry_run=bool(args.dry_run),
        )
        manifest.save(str(raw_dir / f"manifest_recovery_{fam_id}.json"))

        logger.info(
            f"  [Self] Best Recovery (Matched-Plain): Layer {res['self']['best_recovery_layer']} (d={res['self']['best_recovery_depth']:.2f}) -> {res['self'].get('max_recovery_ratio_matched_plain', res['self']['max_recovery_ratio'])*100:.1f}%"
        )
        logger.info(
            f"  [Reader] Best Recovery (Matched-Plain): Layer {res['reader']['best_recovery_layer']} (d={res['reader']['best_recovery_depth']:.2f}) -> {res['reader'].get('max_recovery_ratio_matched_plain', res['reader']['max_recovery_ratio'])*100:.1f}%"
        )

    if all_sample_records:
        df_all_samples = pd.DataFrame(all_sample_records)
        df_all_samples.to_csv(raw_dir / "v2_recovery_sample_level_all.csv", index=False)
        df_all_samples.to_csv(derived_dir / "v2_recovery_sample_level_all.csv", index=False)
        logger.info(f"Saved cross-family sample recovery records (N={len(df_all_samples)}) to {derived_dir / 'v2_recovery_sample_level_all.csv'}")

    # 4ファミリー統合サマリー（Primary: Matched-Plain, Secondary: Native-Chat, Mechanistic Control: Aligned）
    # 1. Primary: Matched-Plain
    self_max_matched = [all_recovery_results[f]["self"]["max_recovery_ratio_matched_plain"] for f in all_recovery_results]
    reader_max_matched = [all_recovery_results[f]["reader"]["max_recovery_ratio_matched_plain"] for f in all_recovery_results]
    self_auc_matched = [all_recovery_results[f]["self"]["auc_recovery_matched_plain"] for f in all_recovery_results]
    reader_auc_matched = [all_recovery_results[f]["reader"]["auc_recovery_matched_plain"] for f in all_recovery_results]

    pt_self_m, s_m_low, s_m_up = compute_bootstrap_ci(self_max_matched, n_boot=n_boot)
    pt_reader_m, r_m_low, r_m_up = compute_bootstrap_ci(reader_max_matched, n_boot=n_boot)
    paired_comp_matched = paired_family_comparison(self_max_matched, reader_max_matched)

    pt_auc_self_m, a_s_m_low, a_s_m_up = compute_bootstrap_ci(self_auc_matched, n_boot=n_boot)
    pt_auc_reader_m, a_r_m_low, a_r_m_up = compute_bootstrap_ci(reader_auc_matched, n_boot=n_boot)
    paired_auc_comp_matched = paired_family_comparison(self_auc_matched, reader_auc_matched)

    # 2. Secondary: Direct Native
    self_max_native = [all_recovery_results[f]["self"]["max_recovery_ratio"] for f in all_recovery_results]
    reader_max_native = [all_recovery_results[f]["reader"]["max_recovery_ratio"] for f in all_recovery_results]
    self_auc_native = [all_recovery_results[f]["self"]["auc_recovery"] for f in all_recovery_results]
    reader_auc_native = [all_recovery_results[f]["reader"]["auc_recovery"] for f in all_recovery_results]

    pt_self_n, s_n_low, s_n_up = compute_bootstrap_ci(self_max_native, n_boot=n_boot)
    pt_reader_n, r_n_low, r_n_up = compute_bootstrap_ci(reader_max_native, n_boot=n_boot)
    paired_comp_native = paired_family_comparison(self_max_native, reader_max_native)

    pt_auc_self_n, a_s_n_low, a_s_n_up = compute_bootstrap_ci(self_auc_native, n_boot=n_boot)
    pt_auc_reader_n, a_r_n_low, a_r_n_up = compute_bootstrap_ci(reader_auc_native, n_boot=n_boot)
    paired_auc_comp_native = paired_family_comparison(self_auc_native, reader_auc_native)

    # 3. Mechanistic Control: Aligned Procrustes
    self_max_aligned = [all_recovery_results[f]["self"]["max_recovery_ratio_aligned"] for f in all_recovery_results]
    reader_max_aligned = [all_recovery_results[f]["reader"]["max_recovery_ratio_aligned"] for f in all_recovery_results]
    self_auc_aligned = [all_recovery_results[f]["self"]["auc_recovery_aligned"] for f in all_recovery_results]
    reader_auc_aligned = [all_recovery_results[f]["reader"]["auc_recovery_aligned"] for f in all_recovery_results]

    pt_self_al, s_al_low, s_al_up = compute_bootstrap_ci(self_max_aligned, n_boot=n_boot)
    pt_reader_al, r_al_low, r_al_up = compute_bootstrap_ci(reader_max_aligned, n_boot=n_boot)
    paired_comp_aligned = paired_family_comparison(self_max_aligned, reader_max_aligned)

    pt_auc_self_al, a_s_al_low, a_s_al_up = compute_bootstrap_ci(self_auc_aligned, n_boot=n_boot)
    pt_auc_reader_al, a_r_al_low, a_r_al_up = compute_bootstrap_ci(reader_auc_aligned, n_boot=n_boot)
    paired_auc_comp_aligned = paired_family_comparison(self_auc_aligned, reader_auc_aligned)

    summary_data = {
        "per_family": all_recovery_results,
        "primary_matched_plain": {
            "cross_family_bootstrap_ci_95": {
                "self_max_recovery_ratio": {"mean": pt_self_m, "ci_lower": s_m_low, "ci_upper": s_m_up},
                "reader_max_recovery_ratio": {"mean": pt_reader_m, "ci_lower": r_m_low, "ci_upper": r_m_up},
                "self_auc_recovery": {"mean": pt_auc_self_m, "ci_lower": a_s_m_low, "ci_upper": a_s_m_up},
                "reader_auc_recovery": {"mean": pt_auc_reader_m, "ci_lower": a_r_m_low, "ci_upper": a_r_m_up},
            },
            "paired_task_comparison_max_ratio": paired_comp_matched,
            "paired_task_comparison_auc": paired_auc_comp_matched,
        },
        "secondary_native_chat": {
            "cross_family_bootstrap_ci_95": {
                "self_max_recovery_ratio": {"mean": pt_self_n, "ci_lower": s_n_low, "ci_upper": s_n_up},
                "reader_max_recovery_ratio": {"mean": pt_reader_n, "ci_lower": r_n_low, "ci_upper": r_n_up},
                "self_auc_recovery": {"mean": pt_auc_self_n, "ci_lower": a_s_n_low, "ci_upper": a_s_n_up},
                "reader_auc_recovery": {"mean": pt_auc_reader_n, "ci_lower": a_r_n_low, "ci_upper": a_r_n_up},
            },
            "paired_task_comparison_max_ratio": paired_comp_native,
            "paired_task_comparison_auc": paired_auc_comp_native,
        },
        "mechanistic_control_aligned": {
            "cross_family_bootstrap_ci_95": {
                "self_max_recovery_ratio": {"mean": pt_self_al, "ci_lower": s_al_low, "ci_upper": s_al_up},
                "reader_max_recovery_ratio": {"mean": pt_reader_al, "ci_lower": r_al_low, "ci_upper": r_al_up},
                "self_auc_recovery": {"mean": pt_auc_self_al, "ci_lower": a_s_al_low, "ci_upper": a_s_al_up},
                "reader_auc_recovery": {"mean": pt_auc_reader_al, "ci_lower": a_r_al_low, "ci_upper": a_r_al_up},
            },
            "paired_task_comparison_max_ratio": paired_comp_aligned,
            "paired_task_comparison_auc": paired_auc_comp_aligned,
        },
    }

    summary_path = derived_dir / "v2_distribution_recovery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"All recovery experiments completed! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

