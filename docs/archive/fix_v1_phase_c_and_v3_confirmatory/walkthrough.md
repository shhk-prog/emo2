# 修正完了確認書 (Walkthrough): V1 Phase C 正規化・高速化および V3 Confirmatory 抜本改修

ユーザーからの包括的なフィードバックに基づき、**V1 Phase C におけるパッチ位置の1-tokenズレの根絶と計算高速化**、および **V3 Confirmatory Replication における 7 大バグ・不整合の修正** を完了しました。

---

## 修正内容の概要 (Changes Made)

### 1. V1 Phase C Causal Patching の正規化と最適化
- **対象ファイル**: [`v1/scripts/run_v1_phase_c_causal_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/scripts/run_v1_phase_c_causal_patching.py)
- **変更内容**:
  1. **介入位置の完全正規化**:
     - `encoded_s = tokenizer(..., return_tensors="pt")` のような BOS 付与によるズレを廃止。
     - `prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)`
     - `prompt_end = len(prompt_ids) - 1` に統一。
     - 共通 `affective_empathy_eval.likelihood` の 729 候補との prefix 整合性を厳格化。
  2. **$\alpha=0.0$ の計算スキップ**:
     - 介入重み $\alpha=0.0$ の場合はニューラル順伝播を行わず、差分 0.0 を直接記録することで forward 計算量を大幅削減。
  3. **ディスクキャッシュ機能 (`--cache-dir`)**:
     - Clean baseline 分布および各層の hidden states / 差分ベクトルを `.npz` でキャッシュ可能に改修。再実行・resume 時の冗長計算を回避。
  4. **Discovery / Confirmation Split**:
     - AIPsy 192 ペアをシード固定で Discovery (50%) と Confirmation (50%) に分割。
     - E3 の Discovery split でピーク層を同定し、E4 は Confirmation split 上で matched vs. random を評価。
  5. **E3 / E4 の分離実行モード (`--mode all | e3 | e4`)**:
     - E3（全層因果マップ探索）と E4（同定された3〜5層の候補層のみ評価）を分離実行可能に。
  6. **出力ディレクトリの完全分離**:
     - 新結果は `v1/results/derived/v1_phase_c_prompt_end/{model_prefix}/` に出力し、`metadata.json` に設定・Gitコミット・位置規則を自動記録。

### 2. V3 Confirmatory Replication の 7 大バグ修正
- **対象ファイル**: [`v3/scripts/run_v3_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/run_v3_confirmatory_replication.py)
- **変更内容**:
  1. **$D(l)$ の held-out 化**:
     - in-sample fit を撤廃し、5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`) による out-of-fold 予測から held-out $R^2$ を算出。
  2. **$d_V, d_A$ の推定層の修正**:
     - 最終層 $H$ ではなく、中間層 `mid_layer = int(num_layers * 0.65)` の hidden states $H_{mid}$ を明示的に用いて Ridge 回帰を実施。
  3. **実測 clean baseline の参照**:
     - `row.get("ev_clean", 5.0)` による 5.0 fallback を撤廃し、各行に対応する実測 `clean_ev_list[global_idx]` を使用。
  4. **Necessity の matched-neutral baseline & centered 2D 投影除去**:
     - 各サンプルの matched-neutral 実測値をベースラインとして取得。
     - neutral stimuli に対する平均活性 $\mu_{neu}$ を中心化し、$$h' = h - Q Q^\top (h - \mu_{neu})$$ により部分空間除去を実施。
  5. **$d_V, d_A$ の QR 分解直交化**:
     - $d_V, d_A$ を個別に引く二重除去を廃止し、QR 分解により 2D 正規直交基底 $Q \in \mathbb{R}^{d \times 2}$ を構成して一括除去。
  6. **Temporal anchors における candidate トークン位置の厳密同定**:
     - prompt 単体ではなく、代表 candidate 文字列と結合した joint sequence に対し `get_generation_stage_tokens` を適用して、`candidate_start`, `pre_V`, `V_value`, `pre_A`, `A_value`, `response_end` の位置を厳密に計算。

### 3. `plot_paper_figures.py` の人工乱数データ除去
- **対象ファイル**: [`v3/scripts/plot_paper_figures.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/plot_paper_figures.py)
- **変更内容**:
  - Figure 4(b) に残っていた `np.random.uniform`, `np.random.normal` による合成散布図を完全撤去。
  - 実測 CSV [`qwen_emobank_recognition_post.csv`](file:///mnt/nas/home/hiromi/src/emo2/v3/results/recognition_baseline/qwen_emobank_recognition_post.csv) の 2,248 件の実データから正当に描画するように改修。
  - `RESULTS_DIR` をスクリプト相対パスで堅牢化。

### 4. POT (`ot`) 未インストール環境での安全化
- **対象ファイル**: [`v3/src/ot_utils.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/src/ot_utils.py), [`v3/tests/test_causal_extensions.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/tests/test_causal_extensions.py)
- **変更内容**:
  - `import ot` の try-except ガードと、未インストール時の親切な例外メッセージ。
  - テストに `pytest.importorskip("ot")` を設置し、テスト収集のクラッシュを防止。

### 5. 新規単体テストの作成
- **対象ファイル**: [`tests/test_phase_c_tokenization_and_anchors.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_phase_c_tokenization_and_anchors.py)
- **検証項目**:
  - Tokenization 境界の整合性と `prompt_end` 位置不変量
  - 2D QR 直交基底の性質（$Q^\top Q = I_2$）および centered 投影除去の直交性（$Q^\top (h' - \mu) = 0$）
  - Generation stage tokens の単調順序関係
  - $\alpha=0.0$ 最適化によるゼロシフト不変量
  - Exact derangement の自己一致ゼロ検証

---

## ターミナルでの検証実行手順

ユーザー運用ルール（コマンドはターミナルで実行する）に基づき、以下のコマンドをターミナルで実行して動作を確認できます。

### 1. 新規単体テストの実行
```bash
.venv/bin/pytest tests/test_phase_c_tokenization_and_anchors.py -v
```

### 2. V3 Confirmatory Replication の Dry-run 検証
```bash
.venv/bin/python v3/scripts/run_v3_confirmatory_replication.py --dry-run
```

### 3. V1 Phase C の小規模スモークテスト（例: 2ペア × 2層）
```bash
.venv/bin/python v1/scripts/run_v1_phase_c_causal_patching.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --limit 2 \
  --layers 14 18 \
  --alphas 0.0 1.0 \
  --out-dir v1/results/derived/v1_phase_c_prompt_end_test
```
