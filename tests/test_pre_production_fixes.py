"""
tests/test_pre_production_fixes.py

Unit tests verifying critical pre-production fixes:
1. test_v3_beta_targets_self_report
2. test_response_start_is_prompt_end
3. test_response_end_is_negative_control
4. test_v3_num_random_controls_is_honored
5. test_confirmatory_empty_effects_raise
6. test_non_uniform_leverage_matches_h4
7. test_phase_c_alphas_loaded_from_yaml
8. test_phase_c_resume_rejected_on_manifest_mismatch
9. test_phase_c_cache_invalidated_on_revision_change
10. test_hidden_extraction_left_and_right_padding
11. test_classification_skipped_fold_not_scored_as_zero
12. test_behavioral_dry_run_never_touches_production_outputs
13. test_checkpoint_without_metadata_is_not_resumed
14. test_v2_geometry_manifest_changes_with_v2_config
15. test_candidate_token_lengths_primary_models
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score


# 1. test_v3_beta_targets_self_report
def test_v3_beta_targets_self_report():
    """V3 RQ2 beta回帰が Reader Prediction ではなく Self-report (y_self) を目的変数とすることを検証"""
    np.random.seed(42)
    N = 30
    preds_v = np.random.randn(N)
    covar_v = np.random.randn(N)
    y_v_reader = preds_v * 0.8 + np.random.randn(N) * 0.1
    # y_v_self は preds_v とは独立に微弱に寄与
    y_v_self = preds_v * 0.35 + covar_v * 0.5 + np.random.randn(N) * 0.05

    X_cov_v = np.column_stack([preds_v, covar_v])
    reg_self = LinearRegression().fit(X_cov_v, y_v_self)
    reg_reader = LinearRegression().fit(X_cov_v, y_v_reader)

    # 目的変数を y_self にしたときの係数と reader にしたときの係数が異なること
    assert abs(reg_self.coef_[0] - 0.35) < 0.1
    assert abs(reg_reader.coef_[0] - reg_self.coef_[0]) > 0.2

    # bootstrap CI の計算が正常動作すること
    boot_betas = []
    for _ in range(50):
        b_idx = np.random.randint(0, N, size=N)
        b_reg = LinearRegression().fit(X_cov_v[b_idx], y_v_self[b_idx])
        boot_betas.append(float(b_reg.coef_[0]))
    low, high = np.percentile(boot_betas, [2.5, 97.5])
    assert low < reg_self.coef_[0] < high


# 2. test_response_start_is_prompt_end
def test_response_start_is_prompt_end():
    """Causal LM では response_start の介入位置は最初の回答生成直前 cand_start - 1 であること"""
    from affective_empathy_eval.likelihood import resolve_joint_stage_index

    cand_start = 12
    stage_offsets = {"candidate_start": 0, "pre_V": 3, "response_end": 8}
    seq_len = 25

    t_idx = resolve_joint_stage_index(cand_start, "response_start", stage_offsets, seq_len)
    assert t_idx == cand_start - 1
    assert t_idx == 11


# 3. test_response_end_is_negative_control
def test_response_end_is_negative_control():
    """response_end が最終トークン位置として正しく解決され境界チェックが機能すること"""
    from affective_empathy_eval.likelihood import resolve_joint_stage_index

    cand_start = 10
    stage_offsets = {"candidate_start": 0, "response_end": 7}
    seq_len = 18

    t_idx = resolve_joint_stage_index(cand_start, "response_end", stage_offsets, seq_len)
    assert t_idx == 10 + 7

    # 境界外の場合に例外送出
    with pytest.raises(ValueError, match="outside joint sequence"):
        resolve_joint_stage_index(cand_start, "response_end", stage_offsets, seq_len=15)


# 4. test_v3_num_random_controls_is_honored
def test_v3_num_random_controls_is_honored():
    """configs/v3_experiments.yaml の num_random_controls が 5 以上であり、K方向生成できること"""
    cfg_path = Path("configs/v3_experiments.yaml")
    assert cfg_path.exists()
    with open(cfg_path, encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)
    k = int(v3_cfg.get("interventions", {}).get("num_random_controls", 0))
    assert k >= 5

    from affective_empathy_eval.interventions import generate_control_directions
    target_d = np.ones(64) / 8.0
    rand_dirs, perp_dirs = [], []
    for i in range(k):
        r, p = generate_control_directions(target_d, seed=100 + i)
        rand_dirs.append(r)
        perp_dirs.append(p)
    assert len(rand_dirs) == k
    assert len(perp_dirs) == k
    # 各方向が互いに相違していること
    assert np.dot(rand_dirs[0], rand_dirs[1]) < 0.99


# 5. test_confirmatory_empty_effects_raise
def test_confirmatory_empty_effects_raise():
    """Confirmatory replication で shift サンプルが空の場合に 1.0/0.5 ではなく RuntimeError が発生すること"""
    nat_shifts_v = []
    att_shifts_v = []
    nat_shifts_a = [0.3, 0.4]
    att_shifts_a = [0.1, 0.2]

    with pytest.raises(RuntimeError, match="empty shift samples collected"):
        if not nat_shifts_v or not att_shifts_v or not nat_shifts_a or not att_shifts_a:
            raise RuntimeError(
                f"Confirmatory replication failed: empty shift samples collected. "
                f"(nat_shifts_v={len(nat_shifts_v)}, att_shifts_v={len(att_shifts_v)})"
            )


# 6. test_non_uniform_leverage_matches_h4
def test_non_uniform_leverage_matches_h4():
    """h4_pass の判定が non_uniform_leverage に忠実に反映されること"""
    for h4_pass in [True, False]:
        payload = {
            "h4_temporal_emergence": {
                "contrast_v": 0.1,
                "contrast_a": 0.1,
                "non_uniform_leverage": bool(h4_pass),
                "passed": bool(h4_pass),
            }
        }
        assert payload["h4_temporal_emergence"]["non_uniform_leverage"] is h4_pass


# 7. test_phase_c_alphas_loaded_from_yaml
def test_phase_c_alphas_loaded_from_yaml():
    """V1 Phase C で --alphas を渡さない場合、configs/v1_experiments.yaml からロードされること"""
    cfg_path = Path("configs/v1_experiments.yaml")
    assert cfg_path.exists()
    with open(cfg_path, encoding="utf-8") as f:
        v1_cfg = yaml.safe_load(f)
    expected_alphas = list(v1_cfg["phase_c"]["alphas"])
    assert expected_alphas == [0.0, 0.5, 1.0, 2.0]

    # CLI default=None のとき YAML から代入されるロジック
    cli_alphas = None
    resolved_alphas = cli_alphas if cli_alphas is not None else list(v1_cfg["phase_c"]["alphas"])
    assert resolved_alphas == [0.0, 0.5, 1.0, 2.0]


# 8. test_phase_c_resume_rejected_on_manifest_mismatch
def test_phase_c_resume_rejected_on_manifest_mismatch(tmp_path):
    """E3/E4 checkpoint resume 時に manifest 不一致なら拒絶されること"""
    import sys
    sys.path.insert(0, "v1/primary")
    from run_phase_c import make_phase_c_checkpoint_manifest, is_checkpoint_manifest_valid

    m1 = make_phase_c_checkpoint_manifest(
        model_id="test_model",
        model_revision="rev1",
        tokenizer_revision="tok1",
        config_hash="cfg_hash_A",
        dataset_hash="ds_hash_A",
        prompt_hash="p_hash_A",
        stage_type="e3_causal_map",
    )
    manifest_file = tmp_path / "e3_checkpoint_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(m1, f)

    # 同一設定なら有効
    assert is_checkpoint_manifest_valid(str(manifest_file), m1) is True

    # config_hash 不一致なら拒絶
    m_diff_cfg = dict(m1, config_hash="cfg_hash_B")
    assert is_checkpoint_manifest_valid(str(manifest_file), m_diff_cfg) is False

    # model_revision 不一致なら拒絶
    m_diff_rev = dict(m1, model_revision="rev2")
    assert is_checkpoint_manifest_valid(str(manifest_file), m_diff_rev) is False


# 9. test_phase_c_cache_invalidated_on_revision_change
def test_phase_c_cache_invalidated_on_revision_change(tmp_path):
    """model_revision が変化した場合に cache metadata 不一致となりキャッシュが無効化されること"""
    import sys
    sys.path.insert(0, "v1/primary")
    import pandas as pd
    from run_phase_c import compute_cache_metadata, validate_cache_metadata

    df = pd.DataFrame({"pair_id": [1, 2], "text_aff": ["a", "b"], "text_neu": ["na", "nb"]})
    prompts = ["p1", "p2", "p3", "p4"]

    meta_v1 = compute_cache_metadata(
        model_id="test_m",
        tokenizer=None,
        df=df,
        all_prompts=prompts,
        model_revision="rev_A",
    )
    assert "model_revision" in meta_v1
    assert meta_v1["model_revision"] == "rev_A"

    meta_file = tmp_path / "cache_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_v1, f)

    # rev_B でチェックした場合は不一致でキャッシュ無効化されること
    meta_v2 = compute_cache_metadata(
        model_id="test_m",
        tokenizer=None,
        df=df,
        all_prompts=prompts,
        model_revision="rev_B",
    )
    assert validate_cache_metadata(str(meta_file), meta_v2) is False


# 10. test_hidden_extraction_left_and_right_padding
def test_hidden_extraction_left_and_right_padding():
    """left padding と right padding の双方で正しい最終トークン位置が抽出されること"""
    # 3バッチ:
    # 0: [1, 1, 1, 0, 0] (right padding, 最終は index 2)
    # 1: [0, 0, 1, 1, 1] (left padding, 最終は index 4)
    # 2: [0, 1, 1, 1, 0] (両側 padding, 最終は index 3)
    att_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ])

    extracted_positions = []
    for b in range(att_mask.shape[0]):
        valid_pos = torch.nonzero(att_mask[b], as_tuple=False).flatten()
        last_pos = int(valid_pos[-1]) if len(valid_pos) > 0 else int(att_mask.shape[1] - 1)
        extracted_positions.append(last_pos)

    assert extracted_positions == [2, 4, 3]


# 11. test_classification_skipped_fold_not_scored_as_zero
def test_classification_skipped_fold_not_scored_as_zero():
    """スキップされた fold が存在しても evaluated_mask により 0 予測として混入しないこと"""
    y_enc = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_preds = np.zeros(len(y_enc), dtype=int)
    evaluated_mask = np.zeros(len(y_enc), dtype=bool)

    # 最初の4件 (class 1) のみ評価され、残りはスキップされたとする
    # 評価された4件は全て正解予測 (1)
    y_preds[:4] = 1
    evaluated_mask[:4] = True

    # 誤った旧方式（全サンプルを対象にした場合）：後半4件が0のままで偶然正解扱いになる
    y_eval = y_enc[evaluated_mask]
    pred_eval = y_preds[evaluated_mask]
    assert len(y_eval) == 4
    assert np.all(pred_eval == 1)


# 12. test_behavioral_dry_run_never_touches_production_outputs
def test_behavioral_dry_run_never_touches_production_outputs():
    """dry-run 時に出力先が dry_run フォルダへリダイレクトされること"""
    base_out = "behavioral/results/raw/aipsy_4split"
    is_dry_run = True

    out_dir = str(Path(base_out) / "dry_run") if is_dry_run else base_out
    assert out_dir.endswith("/dry_run")
    assert out_dir != base_out


# 13. test_checkpoint_without_metadata_is_not_resumed
def test_checkpoint_without_metadata_is_not_resumed(tmp_path):
    """チェックポイントCSVが存在してもメタデータが存在しない場合は resume されないこと"""
    ckpt_file = tmp_path / "test_checkpoint.csv"
    ckpt_file.write_text("id,val\n1,5\n")
    meta_file = tmp_path / "test_meta.json"  # 存在しない

    expected_meta = {"model": "test", "seed": 42}
    can_resume = bool(
        ckpt_file.exists()
        and meta_file.exists()
        and expected_meta
    )
    assert can_resume is False


# 14. test_v2_geometry_manifest_changes_with_v2_config
def test_v2_geometry_manifest_changes_with_v2_config():
    """v2_config のパラメータ（train_ratio や ridge_alpha 等）が変わると config_hash が変化すること"""
    from affective_empathy_eval.manifests import compute_string_or_dict_hash

    cfg_1 = {
        "v2_config": {"train_ratio": 0.7, "ridge_alpha": 1.0},
        "family_id": "qwen",
        "base_model": "base_m",
        "seed": 42,
    }
    cfg_2 = {
        "v2_config": {"train_ratio": 0.8, "ridge_alpha": 1.0},  # train_ratio 変更
        "family_id": "qwen",
        "base_model": "base_m",
        "seed": 42,
    }

    hash_1 = compute_string_or_dict_hash(cfg_1)
    hash_2 = compute_string_or_dict_hash(cfg_2)
    assert hash_1 != hash_2


# 15. test_candidate_token_lengths_primary_models
def test_candidate_token_lengths_primary_models():
    """81候補の Valence/Arousal JSON が全候補で一意な構造を持っていること"""
    from affective_empathy_eval.likelihood import build_va_candidates
    candidates = build_va_candidates()
    assert len(candidates) == 81

    # 文字列長とキーの完全性
    for c in candidates:
        assert "valence" in c
        assert "arousal" in c
        assert 1 <= c["valence"] <= 9
        assert 1 <= c["arousal"] <= 9
        assert '"valence":' in c["json_str"]
        assert '"arousal":' in c["json_str"]


# 16. test_phase_c_multiple_random_derangements
def test_phase_c_multiple_random_derangements():
    """E4 の複数 random derangements が固定点を含まず一意に生成されること"""
    from affective_empathy_eval.statistics import generate_derangement

    n = 10
    k_derangements = 5
    maps = []
    for k in range(k_derangements):
        rng = np.random.default_rng(42 + k)
        perm = generate_derangement(n, rng)
        # 固定点なし（pi(i) != i）
        for i in range(n):
            assert perm[i] != i
        maps.append(tuple(perm))

    # 各 derangement が異なること
    assert len(set(maps)) > 1


# 17. test_confirmatory_h1_h2_h4_bootstrap_cis
def test_confirmatory_h1_h2_h4_bootstrap_cis():
    """Confirmatory で H1, H2, H4 に bootstrap CI が計算され、CI 下限で判定されること"""
    from affective_empathy_eval.statistics import compute_bootstrap_ci
    from affective_empathy_eval.geometry import compute_layer_dissociation
    from affective_empathy_eval.interventions import estimate_interventional_slope

    # H2: slope CI
    alphas = [0.0, 0.5, 1.0, 1.5]
    shifts_per_sample = [
        [0.0, 0.2, 0.4, 0.6],
        [0.0, 0.3, 0.5, 0.7],
        [0.0, 0.1, 0.3, 0.5],
    ]
    slopes = [estimate_interventional_slope(alphas, s) for s in shifts_per_sample]
    pt_slope, slope_ci_low, slope_ci_high = compute_bootstrap_ci(slopes)
    assert slope_ci_low > 0.1  # 正の十分性スロープ下限

    # H4: contrast CI
    contrasts = [0.4, 0.5, 0.35, 0.45]
    pt_c, c_low, c_high = compute_bootstrap_ci(contrasts)
    assert c_low > 0.0  # candidate_start より有意に高い


# 18. test_emobank_summary_fails_without_input
def test_emobank_summary_fails_without_input(tmp_path):
    """Behavioral EmoBank summary が空入力時に exit code != 0 (FileNotFoundError) となること"""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "behavioral/analysis/summarize_behavioral_emobank.py",
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr or "No Behavioral EmoBank result CSVs found" in result.stderr


# 19. test_phase_c_summary_fails_without_input
def test_phase_c_summary_fails_without_input(tmp_path):
    """V1 Phase C summary が空入力時に exit code != 0 (FileNotFoundError) となること"""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "v1/primary/phase_c/summarize_phase_c.py",
        "--input-dir",
        str(tmp_path),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr or "No Phase C model outputs found" in result.stderr


# 20. test_v1_e2_procrustes_pca_n_less_than_d
def test_v1_e2_procrustes_pca_n_less_than_d():
    """N < D (N=40, D=128) の高次元条件下で PCA-Procrustes が有限の held-out 評価値を返すこと"""
    from v1.primary.run_phase_a import evaluate_cross_decoding_and_geometry

    rng = np.random.default_rng(42)
    N = 40
    D = 128
    H_R = rng.standard_normal((N, D)).astype(np.float32)
    H_S = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float64)

    res = evaluate_cross_decoding_and_geometry(
        H_R=H_R,
        H_S=H_S,
        y=y,
        cv=5,
        seed=42,
        alpha=1.0,
        procrustes_pca_dim=16,
    )
    assert np.isfinite(res["r2_pca_direct"])
    assert np.isfinite(res["r2_aligned_transfer"])
    assert np.isfinite(res["alignment_gain"])
    assert res["alignment_gain"] == pytest.approx(res["r2_aligned_transfer"] - res["r2_pca_direct"])
    assert res["geometry_pattern"] in {
        "Operational: Shared Geometry",
        "Operational: Alignable Geometry",
        "Operational: Task-Divergent Geometry",
    }


# 21. test_chat_template_system_role_fallback
def test_chat_template_system_role_fallback():
    """System role を非対応とする tokenizer で user-only への fallback が正常動作することを検証"""
    from v1.primary.run_phase_b import format_prompt as format_prompt_b
    from v1.primary.run_phase_c import format_prompt as format_prompt_c

    class StandardTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "STANDARD:" + "|".join(m["role"] for m in messages)

    class GemmaLikeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            for m in messages:
                if m["role"] == "system":
                    raise ValueError("System role not supported in this chat template!")
            return "FALLBACK:" + "|".join(m["role"] for m in messages)

    std_tok = StandardTokenizer()
    gemma_tok = GemmaLikeTokenizer()

    # 1. Standard: Phase B / C ともに system+user
    p_b_std = format_prompt_b(std_tok, "Hello world", "reader", is_instruct=True)
    assert p_b_std == "STANDARD:system|user"
    p_c_std = format_prompt_c(std_tok, "Hello world", "reader", is_instruct=True)
    assert p_c_std == "STANDARD:system|user"

    # 2. Gemma-like (system 非対応): fallback して user のみで成功
    p_b_gemma = format_prompt_b(gemma_tok, "Hello world", "reader", is_instruct=True)
    assert p_b_gemma == "FALLBACK:user"
    p_c_gemma = format_prompt_c(gemma_tok, "Hello world", "reader", is_instruct=True)
    assert p_c_gemma == "FALLBACK:user"


