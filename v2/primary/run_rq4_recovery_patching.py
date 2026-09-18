import argparse
import json
import logging
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    """
    指定タスク（Reader または Self）において、Base 活性化の Instruct への層別パッチングを実施。
    刺激ごと（sample-wise）の EMD_VA および回復率 Recovery_{i,l} を算出し、Bootstrap CI を付与。
    """
    num_layers = min(fam_cfg.num_layers, 4) if is_dry_run else fam_cfg.num_layers
    depths = [compute_relative_depth(l, num_layers) for l in range(num_layers)]
    N = len(df)

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
            "summary_by_control_type": {
                "direct_native": {
                    "max_recovery_ratio": max(layer_mean_ratios),
                    "layer_recovery_ratios": layer_mean_ratios
                },
                "matched_plain": {
                    "max_recovery_ratio": max(layer_mean_ratios_plain),
                    "layer_recovery_ratios": layer_mean_ratios_plain
                },
                "aligned_procrustes": {
                    "max_recovery_ratio": max(layer_mean_ratios_aligned),
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

    # Train / Eval 分割 (Procrustes alignment 学習に評価サンプルを含めない)
    n_train = max(2, int(0.7 * N))
    eval_indices = list(range(n_train, N)) if N > n_train else list(range(N))

    for l in range(num_layers):
        # SVD による直交 Procrustes 行列 R の学習 (Train split のみ)
        H_b_train = torch.cat([base_activations[l][idx].squeeze() for idx in range(n_train)], dim=0).view(n_train, -1).cpu().float().numpy()
        H_i_train = torch.cat([inst_activations[l][idx].squeeze() for idx in range(n_train)], dim=0).view(n_train, -1).cpu().float().numpy()

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
        "summary_by_control_type": {
            "direct_native": {
                "max_recovery_ratio": max(layer_mean_ratios),
                "layer_recovery_ratios": layer_mean_ratios,
            },
            "matched_plain": {
                "max_recovery_ratio": max(layer_mean_ratios_plain),
                "layer_recovery_ratios": layer_mean_ratios_plain,
            },
            "aligned_procrustes": {
                "max_recovery_ratio": max(layer_mean_ratios_aligned),
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
) -> dict[str, Any]:
    """
    同一 Family 内の Base 活性化を Instruct モデルへパッチングし、
    Reader および Self 両タスクで分布回復率を測定・比較
    """
    num_layers = fam_cfg.num_layers
    depths = [compute_relative_depth(l, num_layers) for l in range(num_layers)]

    model_base = None
    model_inst = None
    tok_base = None
    tok_inst = None

    if not is_dry_run:
        base_spec = fam_cfg.get_model_spec("base")
        inst_spec = fam_cfg.get_model_spec("instruct")
        logger.info(f"Loading Base ({base_spec.model_id}) and Instruct ({inst_spec.model_id})...")
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        tok_base = AutoTokenizer.from_pretrained(base_spec.model_id)
        tok_inst = AutoTokenizer.from_pretrained(inst_spec.model_id)
        model_base = AutoModelForCausalLM.from_pretrained(
            base_spec.model_id, torch_dtype=dtype, device_map=device if "cuda" in device else None
        )
        model_inst = AutoModelForCausalLM.from_pretrained(
            inst_spec.model_id, torch_dtype=dtype, device_map=device if "cuda" in device else None
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
    )

    # メモリ解放
    if model_base is not None:
        del model_base, model_inst, tok_base, tok_inst
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # タスク間比較 (Self vs Reader 回復率の差)
    diff_max_ratio = float(res_self["max_recovery_ratio"] - res_reader["max_recovery_ratio"])
    diff_best_depth = float(res_self["best_recovery_depth"] - res_reader["best_recovery_depth"])

    return {
        "family_id": fam_id,
        "num_layers": num_layers,
        "relative_depths": depths,
        "reader": res_reader,
        "self": res_self,
        "task_comparison": {
            "diff_max_recovery_self_vs_reader": diff_max_ratio,
            "diff_best_recovery_depth_self_vs_reader": diff_best_depth,
            "self_exceeds_reader": bool(diff_max_ratio > 0),
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
        if out_path.exists() and not args.dry_run:
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached and "self" in cached and "reader" in cached:
                    logger.info(f"Loaded existing results for {fam_id} from {out_path}. Skipping computation.")
                    all_recovery_results[fam_id] = cached
                    continue
            except Exception:
                pass

        logger.info(f"--- Running Recovery Patching for Family: {fam_id} ---")
        res = run_recovery_patching_for_family(
            fam_id=fam_id,
            fam_cfg=fam_cfg,
            df=df,
            device=args.device,
            is_dry_run=args.dry_run,
            n_boot=n_boot,
        )
        all_recovery_results[fam_id] = res

        out_path = raw_dir / f"v2_recovery_{fam_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved family recovery result to {out_path}")
        logger.info(
            f"  [Self] Best Recovery: Layer {res['self']['best_recovery_layer']} (d={res['self']['best_recovery_depth']:.2f}) -> {res['self']['max_recovery_ratio']*100:.1f}%"
        )
        logger.info(
            f"  [Reader] Best Recovery: Layer {res['reader']['best_recovery_layer']} (d={res['reader']['best_recovery_depth']:.2f}) -> {res['reader']['max_recovery_ratio']*100:.1f}%"
        )

    # 4ファミリー統合サマリー（Self vs Reader paired comparison）
    self_max_ratios = [all_recovery_results[f]["self"]["max_recovery_ratio"] for f in all_recovery_results]
    reader_max_ratios = [all_recovery_results[f]["reader"]["max_recovery_ratio"] for f in all_recovery_results]

    pt_self, s_low, s_up = compute_bootstrap_ci(self_max_ratios, n_boot=n_boot)
    pt_reader, r_low, r_up = compute_bootstrap_ci(reader_max_ratios, n_boot=n_boot)
    paired_comp = paired_family_comparison(self_max_ratios, reader_max_ratios)

    summary_data = {
        "per_family": all_recovery_results,
        "cross_family_bootstrap_ci_95": {
            "self_max_recovery_ratio": {"mean": pt_self, "ci_lower": s_low, "ci_upper": s_up},
            "reader_max_recovery_ratio": {"mean": pt_reader, "ci_lower": r_low, "ci_upper": r_up},
        },
        "paired_task_comparison": paired_comp,
    }

    summary_path = derived_dir / "v2_distribution_recovery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"All recovery experiments completed! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

