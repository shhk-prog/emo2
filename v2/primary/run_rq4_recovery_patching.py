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
    train_indices: Optional[list[int]] = None,
    eval_indices: Optional[list[int]] = None,
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

        # 中間層 (d ~ 0.5 - 0.7) で最も Base 分布への回復効果が高いシミュレーション
        # Self は Reader より回復ピークが高い傾向をシミュレート
        peak_amp = 0.85 if task == TaskType.SELF else 0.70
        for d in depths:
            rec_base = float(peak_amp * np.exp(-((d - 0.6) ** 2) / 0.05))
            sample_ratios = [
                float(np.clip(rec_base + rng.normal(0, 0.08), 0.0, 1.0))
                for _ in range(N)
            ]
            sample_patched_emds = [
                sample_initial_emds[i] * (1.0 - sample_ratios[i])
                for i in range(N)
            ]

            mean_emd = float(np.mean(sample_patched_emds))
            mean_ratio = float(np.mean(sample_ratios))
            pt_r, r_low, r_up = compute_bootstrap_ci(sample_ratios, n_boot=n_boot)

            layer_mean_emds.append(mean_emd)
            layer_mean_ratios.append(mean_ratio)
            layer_ratios_ci.append({"mean": pt_r, "ci_lower": r_low, "ci_upper": r_up})

        best_layer = int(np.argmax(layer_mean_ratios))

        # Prompt-format control (Base plain -> Instruct matched-plain)
        layer_mean_ratios_plain = [float(np.clip(r * 1.05 + rng.normal(0, 0.02), 0.0, 1.0)) for r in layer_mean_ratios]

        # Aligned activation patch control (Procrustes aligned Base -> Instruct)
        # 幾何整列を行っても内部表現と出力写像の再編により完全回復しないことを検証
        layer_mean_ratios_aligned = [float(np.clip(r * 0.95 + rng.normal(0, 0.02), 0.0, 1.0)) for r in layer_mean_ratios]

        auc_recovery = float(trapz_func(layer_mean_ratios, depths))
        auc_recovery_plain = float(trapz_func(layer_mean_ratios_plain, depths))
        auc_recovery_aligned = float(trapz_func(layer_mean_ratios_aligned, depths))

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
            "best_recovery_layer": best_layer,
            "best_recovery_depth": depths[best_layer],
            "max_recovery_ratio": max(layer_mean_ratios),
            "max_recovery_ratio_matched_plain": max(layer_mean_ratios_plain),
            "max_recovery_ratio_aligned": max(layer_mean_ratios_aligned),
            "auc_recovery": auc_recovery,
            "auc_recovery_matched_plain": auc_recovery_plain,
            "auc_recovery_aligned": auc_recovery_aligned,
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

    # Train / Eval 分割 (Procrustes alignment 学習に評価サンプルを含めない: seeded group/permutation split)
    if train_indices is None or eval_indices is None:
        rng_split = np.random.default_rng(seed)
        if "pair_id" in df.columns:
            unique_pairs = list(df["pair_id"].unique())
            rng_split.shuffle(unique_pairs)
            n_train_pairs = max(1, int(0.7 * len(unique_pairs)))
            train_pairs = set(unique_pairs[:n_train_pairs])
            train_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in train_pairs]
            eval_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid not in train_pairs]
            if len(eval_indices) == 0:
                eval_indices = train_indices
        else:
            perm = list(rng_split.permutation(N))
            n_train = max(2, int(0.7 * N))
            train_indices = perm[:n_train]
            eval_indices = perm[n_train:] if N > n_train else perm

    for l in range(num_layers):
        # SVD による直交 Procrustes 行列 R の学習 (Train split のみ)
        H_b_train = torch.cat([base_activations[l][idx].squeeze() for idx in train_indices], dim=0).view(len(train_indices), -1).cpu().float().numpy()
        H_i_train = torch.cat([inst_activations[l][idx].squeeze() for idx in train_indices], dim=0).view(len(train_indices), -1).cpu().float().numpy()

        mu_b = np.mean(H_b_train, axis=0, keepdims=True)
        mu_i = np.mean(H_i_train, axis=0, keepdims=True)
        X_b = H_b_train - mu_b
        X_i = H_i_train - mu_i
        M = X_b.T @ X_i
        U, _, Vh = np.linalg.svd(M, full_matrices=False)
        R_l = U @ Vh  # (D, D)
        R_l_torch = torch.tensor(R_l, dtype=torch.float32, device=device)
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
            # Recovery = (W1(P_clean, P_target) - W1(P_patch, P_target)) / W1(P_clean, P_target)
            # P_clean=Instruct 未介入, P_target=Base, P_patch=Base活性化を注入した Instruct
            p_emd = compute_distribution_metrics(probs_patched, base_probs_list[i])["emd_va"]
            ratio = compute_emd_recovery_ratio(sample_initial_emds[i], p_emd)
            sample_patched_emds.append(p_emd)
            sample_ratios.append(ratio)

            # Condition B: Aligned Base -> Instruct patch (Procrustes aligned)
            # base_aligned = (base_act - mu_b) @ R + mu_i
            base_flat = base_act_tensor.view(1, -1).float()
            base_aligned = (base_flat - mu_b_torch) @ R_l_torch + mu_i_torch
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

        mean_emd = float(np.mean(sample_patched_emds))
        mean_ratio = float(np.mean(sample_ratios))
        pt_r, r_low, r_up = compute_bootstrap_ci(sample_ratios, n_boot=n_boot)

        layer_mean_emds.append(mean_emd)
        layer_mean_ratios.append(mean_ratio)
        layer_ratios_ci.append({"mean": pt_r, "ci_lower": r_low, "ci_upper": r_up})
        layer_mean_ratios_plain.append(float(np.mean(sample_ratios_plain)))
        layer_mean_ratios_aligned.append(float(np.mean(sample_ratios_aligned)))

    best_l = int(np.argmax(layer_mean_ratios))
    auc_recovery = float(trapz_func(layer_mean_ratios, depths))
    auc_recovery_plain = float(trapz_func(layer_mean_ratios_plain, depths))
    auc_recovery_aligned = float(trapz_func(layer_mean_ratios_aligned, depths))

    return {
        "task": task.value,
        "initial_emd_mean": initial_emd_mean,
        "sample_initial_emds": sample_initial_emds,
        "recovery_emd_va": layer_mean_emds,
        "recovery_ratios": layer_mean_ratios,  # Condition A: Direct Base -> Instruct
        "recovery_ratios_bootstrap_ci": layer_ratios_ci,
        "recovery_ratios_matched_plain": layer_mean_ratios_plain,
        "recovery_ratios_aligned": layer_mean_ratios_aligned,  # Condition B: Aligned Base -> Instruct
        "best_recovery_layer": best_l,
        "best_recovery_depth": depths[best_l],
        "max_recovery_ratio": layer_mean_ratios[best_l],
        "max_recovery_ratio_matched_plain": max(layer_mean_ratios_plain),
        "max_recovery_ratio_aligned": max(layer_mean_ratios_aligned),
        "auc_recovery": auc_recovery,
        "auc_recovery_matched_plain": auc_recovery_plain,
        "auc_recovery_aligned": auc_recovery_aligned,
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
        n_train_pairs = max(1, int(0.7 * len(unique_pairs)))
        train_pairs = set(unique_pairs[:n_train_pairs])
        train_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid in train_pairs]
        eval_indices = [idx for idx, pid in enumerate(df["pair_id"]) if pid not in train_pairs]
        if len(eval_indices) == 0:
            eval_indices = train_indices
    else:
        perm = list(rng_split.permutation(N))
        n_train = max(2, int(0.7 * N))
        train_indices = perm[:n_train]
        eval_indices = perm[n_train:] if N > n_train else perm

    if is_dry_run:
        model_base, model_inst, tok_base, tok_inst = None, None, None, None
    else:
        spec_base = fam_cfg.get_model_spec("base")
        spec_inst = fam_cfg.get_model_spec("instruct")
        logger.info(f"Loading Base model: {spec_base.model_id}...")
        tok_base = AutoTokenizer.from_pretrained(spec_base.model_id)
        model_base = AutoModelForCausalLM.from_pretrained(
            spec_base.model_id,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device if "cuda" in device else None,
        )
        logger.info(f"Loading Instruct model: {spec_inst.model_id}...")
        tok_inst = AutoTokenizer.from_pretrained(spec_inst.model_id)
        model_inst = AutoModelForCausalLM.from_pretrained(
            spec_inst.model_id,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
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
        train_indices=train_indices,
        eval_indices=eval_indices,
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
        train_indices=train_indices,
        eval_indices=eval_indices,
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

    return {
        "family_id": fam_id,
        "num_layers": num_layers,
        "relative_depths": depths,
        "reader": res_reader,
        "self": res_self,
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

    for fam_id, fam_cfg in target_models.items():
        out_path = raw_dir / f"v2_recovery_{fam_id}.json"
        manifest_path = raw_dir / f"manifest_recovery_{fam_id}.json"

        spec_base = fam_cfg.get_model_spec("base")
        spec_inst = fam_cfg.get_model_spec("instruct")
        manifest_config = {
            "v2_config": v2_config,
            "family_id": fam_id,
            "family_name": fam_cfg.family_name,
            "base_model_id": spec_base.model_id if spec_base else None,
            "instruct_model_id": spec_inst.model_id if spec_inst else None,
            "dataset_path": str(v2_config["dataset"]["path"]),
            "seed": v2_config.get("seed", 42),
            "n_boot": n_boot,
            "max_samples": args.max_samples,
            "dry_run": bool(args.dry_run),
        }

        if out_path.exists() and not args.dry_run:
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached and "self" in cached and "reader" in cached:
                    expected_config_hash = compute_string_or_dict_hash(manifest_config)
                    expected_dataset_hash = compute_string_or_dict_hash(str(v2_config["dataset"]["path"]))
                    if not cached.get("dry_run", False) and is_manifest_matching(
                        str(manifest_path),
                        expected_model_name=fam_cfg.family_name,
                        expected_config_hash=expected_config_hash,
                        expected_dataset_hash=expected_dataset_hash,
                        expected_code_version=DEFAULT_CODE_VERSION,
                        expected_dry_run=False,
                    ):
                        logger.info(f"Loaded existing results for {fam_id} from {out_path}. Skipping computation.")
                        all_recovery_results[fam_id] = cached
                        continue
            except Exception as e:
                logger.warning(f"Cache check failed for {fam_id}: {e}")

        logger.info(f"--- Running Recovery Patching for Family: {fam_id} ---")
        res = run_recovery_patching_for_family(
            fam_id=fam_id,
            fam_cfg=fam_cfg,
            df=df,
            device=args.device,
            is_dry_run=args.dry_run,
            n_boot=n_boot,
            seed=v2_config.get("seed", 42),
        )
        res["dry_run"] = bool(args.dry_run)
        all_recovery_results[fam_id] = res

        out_path = raw_dir / f"v2_recovery_{fam_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved family recovery result to {out_path}")

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

