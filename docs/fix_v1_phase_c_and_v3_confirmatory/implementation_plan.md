# V1 Phase C Causal Patching 正規化・高速化および V3 Confirmatory 抜本修正 実装計画

本計画は、ユーザーからの詳細なフィードバックに基づき、V1 Phase C における介入位置トークナイズの不整合を根治し、計算効率を大幅に向上させるとともに、V3 Confirmatory Replication における 7 つの実装上の問題点（held-out 化、推定層の修正、実測ベースライン参照、QR 直交化 + centered 投影除去、temporal candidate アンカーの厳密同定等）を包括的に修正するための詳細設計である。

---

## ユーザー確認が必要な事項 (User Review Required)

> [!IMPORTANT]
> **1. 出力ディレクトリの完全分離**
> - 旧結果（response-onset版）は `v1/results/derived/v1_phase_c_response_onset_legacy/`（または既存のまま）に保存・維持し、ロバストネス解析として活用可能にします。
> - 今回修正する prompt-end 版の Primary 結果は、新規ディレクトリ `v1/results/derived/v1_phase_c_prompt_end/` に出力します。
>
> **2. E3 と E4 の分離実行および候補層選定（Discovery / Confirmation）**
> - E3: 全層（全28層等）・全192ペア（Discovery split でピーク特定、Confirmation split で追試）で実行し、因果マップと CoM（Center of Mass）を記録。
> - E4: 全層ループをやめ、E3 の Discovery split で同定された代表層（Reader peak, Self peak, その近傍, late layer の計3〜5層）のみを Confirmation split で実行。これにより計算量を約 80% 削減します。
>
> **3. V3 実測ベースラインの厳格化**
> - `5.0` へのデフォルト fallback を廃止し、実測の `clean_ev_list` および matched-neutral の実測レポートを完全に紐付けて使用します。

---

## 変更内容 (Proposed Changes)

### 1. V1 Phase C Causal Patching の正規化・高速化

#### [MODIFY] [`v1/scripts/run_v1_phase_c_causal_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/scripts/run_v1_phase_c_causal_patching.py)
- **介入位置の正規化**:
  - `encoded_s = tokenizer(prompts_s_neu[p_idx], return_tensors="pt")` の BOS 差分を根絶。
  - `prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)`
  - `prompt_end = len(prompt_ids) - 1` に統一。
  - 共通 `compute_sequence_likelihoods_for_candidates` における candidate 開始位置 `cand_start_indices[0] - 1 == prompt_end` をアサート。
- **$\alpha=0.0$ の forward スキップ**:
  - $\alpha=0.0$ のときはパッチング介入を行わず、差分 0.0（shift=0.0）を直接代入して余計な forward を完全削減。
- **キャッシュ機能の追加 (`--cache-dir`)**:
  - Clean baseline 分布と、各層の hidden states ($h_{R,aff}, h_{R,neu}, h_{S,aff}, h_{S,neu}$) および $\Delta h$ を事前キャッシュ。
  - 再実行・resume 時に baseline / hidden state 計算をスキップ可能に。
- **Discovery / Confirmation Split の導入**:
  - AIPsy 192 ペアをペア単位（Group/Pair-level）で Discovery (50%) と Confirmation (50%) に分割。
  - E3 の Discovery 側でピーク層を決定し、E4 は Confirmation 側で評価。
- **E3 / E4 の分離実行フラグ (`--mode e3`, `--mode e4`, `--mode all`)**:
  - E3 を全層実行した後、上位候補層（3〜5層）のみに対して E4 を実行できる設計。
- **安全な層単位 Resume & メタデータ保存**:
  - `metadata.json` に tokenizer 設定、介入位置 (`prompt_end`)、729候補、Git commit を記録し、条件不一致の checkpoint の誤混入を防止。

#### [MODIFY] [`v1/scripts/run_v1_phase_c_targeted_ablation.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/scripts/run_v1_phase_c_targeted_ablation.py)
- E6 についても `prompt_end = len(prompt_ids) - 1` の一貫性を再点検し、Discovery / Confirmation split との整合性を完全に担保。

---

### 2. V3 Confirmatory Replication & 関連スクリプトの修正

#### [MODIFY] [`v3/scripts/run_v3_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/run_v3_confirmatory_replication.py)
ユーザーから指摘された 7 つの重大な実装不整合を修正：
1. **$D(l)$ の held-out 化**:
   - in-sample の `ridge.fit(H, y_v); ridge.predict(H)` を廃止。
   - 5-fold cross-validation (`sklearn.model_selection.KFold`) による out-of-fold 予測から held-out $R^2$ を算出。
2. **$d_V, d_A$ の推定層の修正**:
   - ループ末尾の最終層 $H$ ではなく、`mid_layer = int(num_layers * 0.65)` の hidden states $H_{mid}$ を明示的に抽出し、そこから Ridge 回帰で $d_V, d_A$ を推定。
3. **実測 clean baseline の参照**:
   - `row.get("ev_clean", 5.0)` による 5.0 fallback を撤廃し、事前算出した `clean_ev_list[idx]` を確実に使用。
4. **実測 matched-neutral baseline & centered projection removal**:
   - matched-neutral の実測 report をベースラインとして取得・使用。
   - 介入 hook を centered 形式へ修正：
     $$\mu_{neu} = \frac{1}{N}\sum h_{neu}, \quad h' = h - Q Q^\top (h - \mu_{neu})$$
5. **$d_V, d_A$ の QR 分解による 2D 直交基底化**:
   - $d_V, d_A$ を個別に引き算するのではなく、QR 分解（または `compute_orthonormal_subspace`）により直交基底 $Q = \mathrm{orth}(d_V, d_A) \in \mathbb{R}^{d \times 2}$ を構成し、$Q Q^\top$ で 2D 部分空間を一括除去。
6. **Temporal anchors における candidate トークン位置の厳密同定**:
   - prompt 単体エンコードから candidate 位置を引く誤りを解消。
   - prompt と candidate 文字列（代表例）を結合した joint sequence に対し、`get_generation_stage_tokens` を適用して `candidate_start`, `pre_V`, `V_value`, `pre_A`, `A_value`, `response_end` の正確なトークン位置を特定して介入。

#### [MODIFY] [`v3/scripts/plot_paper_figures.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/plot_paper_figures.py)
- Figure 4(b) に残る `np.random.uniform`, `np.random.normal` による人工データ生成を完全撤去。
- 実測データ CSV / JSON から散布図を直接描画するよう改修。

#### [MODIFY] [`v3/src/ot_utils.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/src/ot_utils.py) & [`v3/tests/test_causal_extensions.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/tests/test_causal_extensions.py)
- `import ot` の未インストール時に親切なエラーまたはスキップを行うガードを設置（`pytest.importorskip("ot")`）。

---

### 3. 新規ユニットテストの作成

#### [NEW] [`tests/test_phase_c_tokenization_and_anchors.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_phase_c_tokenization_and_anchors.py)
- `add_special_tokens=False` での prompt 長と joint sequence の境界位置が完全に一致することを検証。
- $\alpha=0.0$ の shift が厳密に 0 となることの確認。
- 2D QR 直交化基底の性質（$Q^\top Q = I_2$）および centered projection removal の不変量検証。
- Temporal candidate anchor の位置同定の正当性検証。

---

## 検証計画 (Verification Plan)

### 自動テスト
- `tests/test_phase_c_tokenization_and_anchors.py` の実行（tokenization, QR 直交化, temporal anchors, $\alpha=0$ 最適化の単体テスト）
- 既存の共通テスト `tests/` の回帰確認

### スモークテスト（Dry-run / 小規模実行）
- `v3/scripts/run_v3_confirmatory_replication.py --dry-run` の実行確認
- V1 Phase C の 1 pair × 2 layers × 2 alphas のテスト実行による出力フォーマットとメタデータ確認
