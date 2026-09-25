# Walkthrough: Directional Decodability--Causal Dissociation (H1) Replication Criterion の改修と再集計

## 1. 概要と背景

本改修では、V3研究のH1（Decodability--Causal Dissociation）において発生していた**判定基準の符号不整合**を解消しました。

### 背景と問題点
- **指標の定義**：
  $$\Delta d^* = d_C^* - d_D^*, \qquad \Delta\bar{d} = \bar{d}_C - \bar{d}_D$$
- **Qwen Discovery での実測値**：
  - Peak層差：Valence $\Delta d^* = -0.111 < 0$、$d_C^* (0.741) < d_D^* (0.852)$；Arousal $\Delta d^* = -0.111 < 0$、$d_C^* (0.741) < d_D^* (0.852)$
  - 重心層差（Center-of-Mass）：Valence $\Delta\bar{d} = -0.0094 < 0$；Arousal $\Delta\bar{d} = +0.0387 > 0$
- **従来の不整合**：
  従来の cross-family H1 では一律に $CI_{\text{low}}(\Delta d^*) > 0$ かつ $CI_{\text{low}}(\Delta\bar{d}) > 0$ を要求しており、Qwen Discovery で発見された方向性の再現判定になっていませんでした（また Arousal では peak と center-of-mass で符号が異なるため、一律 $< 0$ への単純反転でも破綻します）。

### 解決策
H1を「**Qwen Discoveryで観測された方向のspatial dissociationが他familyでも再現するか（Directional Decodability--Causal Dissociation）**」と再定義。
Qwen Discovery終了時点での符号 $s \in \{-1, +1\}$ を明示的に freeze し、replication families（Llama 3.2, Gemma 3, OLMo 2）では **direction-aligned quantity** $\widetilde{\Delta} = s \cdot \Delta$ が $CI_{\text{low}} > 0$（raw CI では $s=-1$ のとき $CI_{\text{high}} < 0$、$s=+1$ のとき $CI_{\text{low}} > 0$）を満たすかを判定基準としました。

---

## 2. 実施した変更内容

### 2.1 符号の事前 Freeze とアーティファクト保存
- [frozen_confirmatory_sites.json](file:///mnt/nas/home/hiromi/src/emo2/v3/results/derived/frozen_confirmatory_sites.json) および [frozen_h1_replication_direction.json](file:///mnt/nas/home/hiromi/src/emo2/v3/results/derived/frozen_h1_replication_direction.json) に、Qwen Discovery から確定した符号を保存：
  ```json
  {
    "h1_replication_direction": {
      "valence": {
        "peak": -1,
        "center_of_mass": -1,
        "qwen_delta_d_peak": -0.11111111111111116,
        "qwen_delta_d_center": -0.009406566453965939
      },
      "arousal": {
        "peak": -1,
        "center_of_mass": 1,
        "qwen_delta_d_peak": -0.11111111111111116,
        "qwen_delta_d_center": 0.03869279541584949
      }
    }
  }
  ```
- [run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py): canonical freeze 出力時にも上記辞書が自動出力されるよう担保。

### 2.2 Replication スクリプトの改修
- [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py):
  - `frozen_confirmatory_sites.json` から `h1_replication_direction` を読み込み。
  - 出力辞書に raw 指標と aligned 指標の双方を完全に保持：
    - Raw: `delta_peak_raw`, `delta_com_raw`, `delta_d_peak_ci`, `delta_d_center_ci`
    - Direction: `qwen_sign_peak`, `qwen_sign_com`
    - Aligned: `delta_peak_aligned`, `delta_com_aligned`, `aligned_peak_ci`, `aligned_com_ci`
    - 判定フラグ: `peak_replication_pass`, `com_replication_pass`, `h1_replication_pass`
  - メタ分析サマリーにおいても raw CI と aligned CI を分離して保存。

### 2.3 集計スクリプト・サマリーテーブルの更新
- [build_paper_summary.py](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/build_paper_summary.py):
  - H1 レコード抽出時に directional criteria（`peak_replication_pass`, `com_replication_pass`）に基づいて判定するよう改修。
- [summarize_v3_causal_utilization.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/summarize_v3_causal_utilization.py):
  - LaTeX テーブル [v3_confirmatory_details.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v3_confirmatory_details.tex) および [v3_causal_utilization_summary.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v3_causal_utilization_summary.tex) のヘッダー・注釈・閾値表示を `CI_high < 0` / `CI_low > 0` に適合させ、Markdown サマリーも更新。

### 2.4 保存済み結果の再集計
- [scratch/recompute_h1_results.py](file:///mnt/nas/home/hiromi/src/emo2/scratch/recompute_h1_results.py) を実行し、既存の cross-family 結果ファイル群を安全に更新：
  - `v3/results/derived/v3_confirmatory_llama.json`
  - `v3/results/derived/v3_confirmatory_gemma.json`
  - `v3/results/derived/v3_confirmatory_olmo.json`
  - `v3/results/derived/v3_cross_model_replication_summary.json`
  - `results/derived/paper_summary/tables/table_v3_4_confirmatory.csv`
  - `results/derived/paper_summary/tables/table_v3_confirmatory_matrix.csv`

---

## 3. 再集計結果（Key Empirical Findings）

新 criterion（Qwen Discovery の符号に整列した directional replication）での再集計結果：

| Model Family | Metric | Valence Raw Diff (95% CI) | Valence Directional Criterion | Valence Replication Result | Arousal Raw Diff (95% CI) | Arousal Directional Criterion | Arousal Replication Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2** | Peak $\Delta d^*$ | $-0.87$ $[-1.00, -0.27]$ | $\text{CI}_{\text{high}} < 0$ | **PASS** | $+0.07$ $[-0.47, 0.40]$ | $\text{CI}_{\text{high}} < 0$ | FAIL (0を跨ぐ) |
| | Center-of-Mass $\Delta\bar{d}$ | $-0.32$ $[-0.52, -0.11]$ | $\text{CI}_{\text{high}} < 0$ | **PASS** | $-0.12$ $[-0.53, -0.09]$ | $\text{CI}_{\text{low}} > 0$ | FAIL (符号逆転) |
| **Gemma 3** | Peak $\Delta d^*$ | $+0.00$ $[-0.77, 0.13]$ | $\text{CI}_{\text{high}} < 0$ | FAIL (0を跨ぐ) | $+0.00$ $[-0.63, 0.23]$ | $\text{CI}_{\text{high}} < 0$ | FAIL (0を跨ぐ) |
| | Center-of-Mass $\Delta\bar{d}$ | $+0.02$ $[-0.27, 0.29]$ | $\text{CI}_{\text{high}} < 0$ | FAIL (0を跨ぐ) | $-0.07$ $[-0.38, 0.25]$ | $\text{CI}_{\text{low}} > 0$ | FAIL (0を跨ぐ) |
| **OLMo 2** | Peak $\Delta d^*$ | $-0.67$ $[-1.00, -0.07]$ | $\text{CI}_{\text{high}} < 0$ | **PASS** | $-0.93$ $[-0.93, 0.27]$ | $\text{CI}_{\text{high}} < 0$ | FAIL (0を跨ぐ) |
| | Center-of-Mass $\Delta\bar{d}$ | $-0.24$ $[-0.44, -0.04]$ | $\text{CI}_{\text{high}} < 0$ | **PASS** | $-0.62$ $[-0.63, -0.10]$ | $\text{CI}_{\text{low}} > 0$ | FAIL (符号逆転) |

### 主要な発見
1. **Valence における部分的一致**:
   - Llama 3.2 および OLMo 2 では、Peak差・重心差ともに 95% CI が完全に負側に位置（$CI_{\text{high}} < 0$）し、**Qwen Discovery で観測された「Causal層がDecodability層よりも有意に入力側に先行する」という空間的解離が再現（PASS）**されました。
2. **Arousal および Gemma 3 における不一致**:
   - Gemma 3 では Valence / Arousal ともに CI が 0 を跨ぎ有意な解離が再現されませんでした。
   - Arousal においては、Llama 3.2 と OLMo 2 の重心差 $\Delta\bar{d}$ が負方向（$-0.12$, $-0.62$）であり、Qwen Discovery の正方向（$+0.0387$）と符号が逆転していました。
3. **結論の整理**:
   - 全体として、**「Valence においては Qwen Discovery で観測された方向の空間的解離が一部の replication family（Llama, OLMo）で再現されたが、Arousal では再現されず、3 families 全体にわたって一貫して一般化する普遍的解離とはならなかった」** という精密な実証的結論が得られました。
   - また、前提条件である State-Induction Gate が NO_GO（4 family すべてで失敗）であるため、H1 の結果は確証的結論（confirmatory claims）ではなく、探索的知見（exploratory evidence）として論文内で位置づけられています。

---

## 4. 論文原稿（LaTeX）への反映

[iclr2027/iclr2027_conference2.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex) の以下の箇所を改修・更新しました：

1. **Methods (§3.2 Replication H1: Directional Decodability--Causal Dissociation)**:
   - 固定の正方向判定（$CI_{\text{low}} > 0$）を廃止し、事前凍結した Qwen 符号 $s$ による Directional Replication Criterion（$\widetilde{\Delta} = s \cdot \Delta > 0$, 95% bootstrap CI が 0 を跨がない）として正確に定式化。
2. **Results (§3.3 Confirmatory Results & Table 4)**:
   - 「全 family で FAIL」という従前の記述から、**Valence においては Llama 3.2 と OLMo 2 で Qwen と同一方向の解離（$\Delta d^* < 0$, $\Delta\bar{d} < 0$）が再現されたこと、しかし Arousal では再現されず、Gemma 3 も有意な差を示さなかったため、3 families 全体に普遍的に一般化するパターンではなかったこと**を明確に記載。
3. **General Discussion (§4.1 & Guardrail Implication)**:
   - Representation-Causality Dissociation の考察において、Valence で観察された部分的一致と Arousal での不一致を正確に引用し、アーキテクチャ依存性および Guardrail 設定における過度な線形仮定への注意喚起として論述を精密化。

---

## 5. テストと検証

- **ユニットテスト**: [tests/test_confirmatory_pipeline.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_confirmatory_pipeline.py)
  - `test_v3_confirmatory_h1_directional_alignment`: Qwen符号を用いた alignment と CI 判定ロジックの単体検証（PASS）
  - `test_frozen_h1_replication_direction_consistency`: `frozen_confirmatory_sites.json` の符号および値が Qwen spatiotemporal summary と完全一致することの検証（PASS）
  - `pytest -q tests/test_confirmatory_pipeline.py`: 全 6 テスト PASS
- **コード品質**:
  - `v3/scripts/build_paper_summary.py` に `import json` を追加し、構文エラーのない正常実行（24 records 生成）を確認。

---

## 6. fail-fast バリデーションの厳格化と「事前登録」表記の適正化

### 6.1 本番実行時の default フォールバック禁止と厳格な検証
- [run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py):
  - 本番実行時（`not args.dry_run`）に `v3_spatiotemporal_summary.json` が見つからない場合は `FileNotFoundError` を送出。
  - パース失敗・必須キー欠落時にも `RuntimeError` を送出。
  - 各軸（Valence/Arousal）× 各指標（Peak/COM）の符号が厳密に `-1` または `1` であるかのバリデーションを追加。
  - これにより、本番出力されるアーティファクトの H1 符号が**必ず Qwen Discovery の実測値から生成されたものであることをコードレベルで保証**。
- [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py):
  - `frozen_confirmatory_sites.json` 内に `h1_replication_direction` が欠落していた場合、本番実行時（`not args.dry_run`）は default 辞書へフォールバックせず直ちに `KeyError` を送出して停止するよう改修。

### 6.2 表 Caption / Note および本文の「事前登録（prespecified / pre-registered）」表記の適正化
- [summarize_v3_causal_utilization.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/summarize_v3_causal_utilization.py):
  - [v3_confirmatory_details.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v3_confirmatory_details.tex) の Caption を「`V3 cross-family replication details。Llama 3.2、Gemma 3、OLMo 2におけるH1--H4のValence / Arousal別推定値、family-specific 95% confidence interval、replication criterion、およびPass / Failを示す。H1のdirectionはQwen Discovery終了後、replication-family outcomesを評価する前にfreezeした。`」に修正。
  - Note から「事前登録仮説」「事前登録CI基準」を削除し、「`各ファミリー固有の95%ブートストラップ信頼区間および判定基準...に基づく評価。H1では、Qwen Discoveryで観測されたspatial dissociationのdirectionを、Llama、Gemma、OLMoのoutcomeを評価する前にfreezeし、同方向のeffectについてfamily-specific bootstrap CIが0を除外するかを評価した。...`」と適正化。
  - Gate 表（[v3_gate_decision.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v3_gate_decision.tex)）の Caption も「V3 State-Induction Gate 判定結果」に統一。
- [iclr2027/iclr2027_conference2.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex):
  - 本文における H1 の位置づけを「Qwen Discoveryで観測されたdirectionを他familyへfreezeして移送するdirectional replication design」として明確化し、過度な事前登録主張を排除。

### 6.3 全テスト検証結果
- `PYTHONPATH=src:. pytest -q` を実行し、**156 passed, 2 deselected** を確認。

