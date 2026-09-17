# タスク進捗: V1 Phase C Causal Patching 正規化・高速化および V3 Confirmatory 抜本修正

## 完了した項目

- [x] **POT (`ot`) 未インストール環境でのテスト収集・実行の安全化**
  - `v3/src/ot_utils.py`: `import ot` の安全な try-except ガードと明示的エラーメッセージの設置。
  - `v3/tests/test_causal_extensions.py`: `pytest.importorskip("ot")` を追加し、テスト収集のクラッシュを防止。
- [x] **V1 Phase C Causal Patching の正規化と最適化**
  - `v1/scripts/run_v1_phase_c_causal_patching.py`:
    - `add_special_tokens=False` と `prompt_end = len(prompt_ids) - 1` に完全統一（1-tokenズレの根絶）。
    - 共通 Sequence Likelihood (`affective_empathy_eval.likelihood`) の 729 候補と完全一致。
    - $\alpha=0.0$ の forward スキップ（0 shift 直接保存による計算量削減）。
    - Clean baseline および hidden states のディスク事前キャッシュ機能 (`--cache-dir`)。
    - Group/Pair-level での Discovery / Confirmation split (50:50) の導入。
    - E3（全層因果マップ探索）と E4（Discovery同定の候補層3〜5層のみConfirmationで評価）の分離実行モード (`--mode all`, `--mode e3`, `--mode e4`)。
    - 出力先を `v1/results/derived/v1_phase_c_prompt_end` に設定し、旧結果ディレクトリと完全分離。
    - 設定・環境・Gitコミット情報を追跡可能な `metadata.json` の自動出力。
- [x] **V1 Phase C Targeted Ablation (E6) の整合性確認**
  - `v1/scripts/run_v1_phase_c_targeted_ablation.py`:
    - `prompt_end = len(prompt_ids) - 1` および Discovery / Confirmation split との整合性を確認。
- [x] **V3 Confirmatory Replication の 7 大バグ・不整合の修正**
  - `v3/scripts/run_v3_confirmatory_replication.py`:
    1. デコード能 $D(l)$ を 5-fold cross-validation による held-out $R^2$ に改修。
    2. 情動方向 $d_V, d_A$ を最終層ではなく `mid_layer = int(num_layers * 0.65)` の hidden states $H_{mid}$ から正しく推定。
    3. Causal profile における clean baseline の 5.0 fallback を撤廃し、実測 `clean_ev_list` を確実に参照。
    4. matched-neutral の実測ベースライン参照および centered 2D 投影除去 ($h' = h - Q Q^\top (h - \mu_{neu})$) への修正。
    5. $d_V, d_A$ を個別に引くのではなく、QR 分解による 2D 正規直交基底 $Q$ で一括除去（二重除去を排除）。
    6. Temporal anchors において、prompt 単体ではなく candidate 文字列と結合した joint sequence に対し `get_generation_stage_tokens` を用いて `candidate_start`, `pre_V`, `V_value`, `pre_A`, `A_value`, `response_end` の位置を厳密に同定。
- [x] **`v3/scripts/plot_paper_figures.py` の人工乱数データ除去**
  - Figure 4(b) の `np.random` 合成コードを撤廃し、`qwen_emobank_recognition_post.csv` の実測アノテーション・尤度期待値から散布図を正当に描画するよう改修。
  - `RESULTS_DIR` をスクリプト相対パスで堅牢化。
- [x] **新規ユニットテストの作成**
  - `tests/test_phase_c_tokenization_and_anchors.py`:
    - Tokenization 境界の prefix 一致テスト
    - 2D QR 正規直交化および centered 投影除去の直交性不変量テスト
    - Generation stage tokens の単調順序関係テスト
    - $\alpha=0.0$ のゼロシフト不変量テスト
    - Exact derangement の自己一致ゼロテスト
- [x] **論文導線の整理・案内文書の配置**
  - `v3/docs/README.md`: greedy collapse / neutralization の論文導線からの除外・補助指標化、および Behavioral 4本柱への統一を明記。
