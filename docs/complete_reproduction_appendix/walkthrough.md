# Walkthrough: Appendix の正本データ・実在コード完全準拠リファクタリング（完了）

## 1. 監査と是正の総括
ユーザーからの詳細なコード監査指摘に基づき、`v3/docs/paper2.md` の Appendix（A〜G）を最新の正本データおよびGitリポジトリ実態と100%照合し、不整合や誤解釈を完全に一掃しました。

---

## 2. 実施した最新是正項目（全件確認済み）

1. **E.10 Focused 39-Pair 中央値の正本一致**:
   - `focused_causal_sweep_39pairs.csv` に厳密に準拠：
     - Layer 18 Residual: Mean **42.12%**, Median **49.31%**
     - Layer 20 Residual: Mean **50.22%**, Median **60.16%**
     - Layer 24 Residual: Mean **53.24%**, Median **61.57%**, 95% Bootstrap CI **[45.74%, 60.45%]**
     - Layer 15 MLP: Mean **-0.06%**, Median **+0.50%**, 95% Bootstrap CI **[-2.02%, +1.83%]**
2. **E.12 多層パッチング（Section 15.6）の完全分離**:
   - 独立パイプラインの実測値（`generation_multilayer_residual_results.csv`）を適用：
     - L18 単層: Mean **43.51%**, Median **51.52%**
     - L20 単層: Mean **51.85%**, Median **61.12%**
     - L24 単層: Mean **55.08%**, Median **62.77%**
     - L20 + L24: Mean **55.67%**, Median **61.88%**
     - L18 + L20 + L24: Mean **55.74%**, Median **61.88%**
     - L18–L24: Mean **55.35%**, Median **62.77%**
   - focused sweep 単層値（53.24%）との独立性注記を明記。
3. **E.13 & E.14 介入位置（Token Position）の実装完全準拠**:
   - `run_probe_aligned_necessity_sweep.py` および `run_focused_necessity_n100.py` の実装（`target_pos = inputs.input_ids.shape[1] - 1`）通り、`Token Position: Prompt final token (target_pos = prompt_len - 1), not generation-prefix position.` と明記。
   - Prompt-time accessibility / necessity と Generation-time causal leverage の時間的分離を論文全体で死守。
4. **E.13 統計手法記述の刷新**:
   - 誤記だった「Paired t-test on log-odds shift」を排し、実装通りの `2D Joint OT displacement`、等方的・直交ランダム帰無分布に対する `standardized Z-scores`、`pseudo-count empirical p-values`、`Benjamini–Hochberg FDR correction` の正確な数式とプロトコルを記述。
5. **E.14 N=100 特異性検定 p値範囲の是正**:
   - `focused_necessity_sweep_n100.csv` の `p_value_perp` 正本値：
     - 全15サイト範囲: **`0.2970 〜 1.000`**（L20 Resid: 0.2970, L15 MLP: 0.8614, L24 MLP: 0.9802, L24 Attn: 1.000）
     - 全15サイトの FDR 補正後 $q$ 値: **$q = 1.000$**
6. **E.17 結論表現の緩和とLlama Residual 負値範囲の是正**:
   - 結論表現を「モデルファミリーや学習履歴に依存する可能性と整合する」へ緩和。
   - `generation_time_causal_sweep_llama.csv` の全16層 Residual 正本値：
     - **`-36.18% 〜 -4.23%`**（最大正回復は L4 Attention の **0.73%**）
7. **E.15 スクリプト名**:
   - **`v3/scripts/run_dual_outcome_behavior.py`** に是正。
8. **E.16 Mood Congruency 正本値と位置づけ**:
   - 保存先: `v3/results/derived/mood_congruency/Qwen_Qwen2.5-1.5B-Instruct_mood_congruency_ambiguous_summary.csv`
   - 本文確定値:
     - L14 Valence: $\beta = -0.0323$ ($p = 1.18 \times 10^{-70}$) vs Random: $\beta = -0.0042$ ($p = 9.63 \times 10^{-4}$)
     - L16 Valence: $\beta = -0.0169$ ($p = 6.20 \times 10^{-40}$) vs Random: $\beta = +0.0444$ ($p = 6.40 \times 10^{-138}$)
     - L20 Valence: $\beta = +0.0244$ ($p = 3.23 \times 10^{-61}$) vs Random: $\beta = +0.0091$ ($p = 1.45 \times 10^{-23}$)
   - L16 でランダム効果が上回るため、純粋な気分一致回路の確立とは断定せず、探索的分析として客観的に記述。
9. **Appendix B / G 公開リモート追跡確認**:
   - GitHub `main`（`origin/main`）上に `run_generation_multilayer_residual.py`、`plot_main_figure1.py`、`focused_causal_sweep_39pairs_pair_level.csv`、`generation_multilayer_residual_results.csv`、`figure1_four_panel_dissociation.png/pdf` が確実に push 済みであることをコミットハッシュ（`3e65370`, `bce4cbf`, `9f772bc`, `84e9562`, `bbc5fa4`）で確認。
10. **Appendix D (v2) のトーン抑制と事実限定**:
    - 推論や過度な断定を排し、Phase 1〜9 の実測成果物から直接確認できる事実のみに整理。
11. **Appendix G の実在ファイル厳選**:
    - 実在が確認されたスクリプトおよび結果CSVのみを網羅掲載。

---

## 3. 自動検証結果
全検証項目の自動テストスクリプトを実行し、すべての正本値および修正記述が合致（`ALL CHECKS PASSED (100%)`）していることを確認しました。
これで査読者からの指摘リスクを極限まで排除した、正本・実装・Gitリモートに完全準拠した Appendix が完成しました。
