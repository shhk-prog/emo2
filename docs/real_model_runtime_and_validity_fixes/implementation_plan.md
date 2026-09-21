# 実モデル実行経路 BLOCKER 解消および論文妥当性修正 (Implementation Plan)

## 概要
dry-run（モック経路）では表面化しないが、実モデル実行時に確実に発生する未定義変数・インポート漏れ・戻り値不整合（NameError）を全件解消し、同時に論文上の妥当性を担保するための重要数理修正（V2 Procrustes の PCA 部分空間化、V3 Confirmatory H3 の Random 部分空間統制、V1 E4 の Permutation 検定追加等）を実施します。

---

## 修正計画の詳細

### A. 実モデル実行 BLOCKER の修正

1. **V1 `torch_dtype` → `actual_torch_dtype`** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)):
   - 743行目の `dtype=str(torch_dtype)` を `dtype=str(actual_torch_dtype)` に修正。
2. **V1 `e3_patching_records` → `e3_causal_records`** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)):
   - 1726行目の `if e3_patching_records:` を `if e3_causal_records:` に修正。
3. **V1 `--data-path` の一本化** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)):
   - 647行目の `df_aipsy = pd.read_csv(aipsy_path)` を `df_aipsy = pd.read_csv(data_file)` に修正し、上流のハッシュ計算対象と実ロード元を厳密に一致させる。
4. **V3 RQ1 `v3_cfg` スコープ外参照の解消** ([run_rq1_state_induction.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)):
   - `run_real_state_induction()` のシグネチャに `train_ratio: float = 0.7`, `seed: int = 42`, `num_random_controls: int = 5` を追加し、関数内部の `v3_cfg.get(...)` 参照を引数経由に置換。
5. **V3 RQ2 `ref_alpha_idx` の事前算出** ([run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)):
   - `run_real_spatiotemporal_maps()` 冒頭で `ref_alpha_idx = alpha_sweep.index(causal_reference_alpha)` を算出（未存在時は ValueError）。
6. **V3 RQ2 実モデル関数内の未定義 `dry_run` 判定削除** ([run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)):
   - 実モデル関数から未定義の `if not dry_run:` を削除し、`n_groups < 2` や `pair_id` 欠損時は常に厳格な例外を送出。
7. **V3 RQ3 Hook インポート追加** ([run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)):
   - `from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint` を追加。
8. **V3 RQ3 random control list の初期化** ([run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)):
   - 500行目付近に `residual_rand_v_list, residual_rand_a_list = [], []` を追加。
9. **V3 RQ3 return 変数名修正** ([run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)):
   - 671行目の `return discovery_res, confirmation_res` を `return discovery_summary, confirmation_res` に修正。

---

### B. 論文妥当性のための修正

10. **V2 RQ4 Procrustes の高次元 rank-deficient 対策 (PCA → Procrustes)** ([run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)):
    - $D > N_{\mathrm{train}}$ による直交補空間の未同定回転を防止するため、train split のみから共通 PCA 基底 $V_k$ ($k=\min(64, n_{\mathrm{train}}-1, D)$) を抽出し、$k$ 次元空間上で Procrustes 回転行列 $R_k$ を学習。
    - パッチ適用時: $x_{\mathrm{aligned}} = (x - \mu_b) V_k R_k V_k^T + (x - x_{\mathrm{sub}}) + \mu_i$ として補空間は identity で保持。
    - [v2_experiments.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/v2_experiments.yaml) に `procrustes_pca_dim: 64` を明示。
11. **V3 Confirmatory H3 に random-subspace control を導入** ([run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)):
    - 各 fold の train split で $Q_{\mathrm{sub}}$ (2D) に加えて matched-rank random 2D subspace $Q_{\mathrm{rand}}$ を生成。
    - Held-out test で natural shift, affective residual, random residual を実測し、サンプル単位で `net_atten = atten_aff - atten_rand` を計算。
    - Bootstrap CI を算出し、H3 criterion を `absolute_attenuation_ci_lower > threshold and net_attenuation_vs_random_ci_lower > 0` に更新。
12. **V3 RQ3 Net attenuation に CI を保存** ([run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)):
    - `compute_bootstrap_ci(net_atten_v)` / `(net_atten_a)` で `mean`, `ci_lower`, `ci_upper` を算出して保存。
13. **V1 E4 Permutation p-value の実装** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py) & [statistics.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/statistics.py)):
    - `diff = matched - mean_random` に対する paired sign-flip permutation test (10,000回) を追加し、`aligned_permutation_p_V/A`, `permutation_p_V/A` を出力。README の記載と完全に一致させる。

---

### C. 再現性・QA・軽微修正

14. **`data.py` の `from pathlib import Path` 追加 & 例外ログ出力** ([data.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/data.py)).
15. **V1 E6 の `logger` 定義追加** ([run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)).
16. **`compute_cache_metadata()` の transformers import 安全化** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)).
17. **`likelihood.py` の温度パラメータ反映** ([likelihood.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)): Softmax 計算で `temperature` を反映。
18. **manifest config への Sequence-Likelihood 設定保存** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py), [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)).
19. **`README.md` の検定名修正** ([README.md](file:///mnt/nas/home/hiromi/src/emo2/README.md)): "Permutation Test" → "paired Wilcoxon signed-rank test".
20. **V3 Confirmatory サンプル数の config 化** ([v3_experiments.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml), [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)): `n_intervention_samples_per_fold: 5` を config 化。
21. **`ci_95_high_V` 重複キーの削除** ([run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)).
22. **`likelihood.py` docstring 更新** (token mean).

---

## 検証手順

1. **単体テスト (`pytest -q`)**: 全件 PASS の確認。
2. **実モデル小規模スモークテスト**:
   - `python v1/primary/run_phase_c.py --model-id Qwen/Qwen2.5-1.5B-Instruct --limit 2 --layers 0 --device cpu` 等で実モデル経路の NameError が完全に解消されたことを直接検証。
3. **全 Stage Dry-run 完走確認**: Behavioral, V1, V2, V3 の dry-run 実行。
4. **結果整理と Walkthrough 作成**: `docs/real_model_runtime_and_validity_fixes/walkthrough.md` を作成して報告。
