# 全再実行・リポジトリ公開に向けた最終リファインメント完了報告 (Walkthrough)

## 概要
全再実行およびリポジトリ公開に向けた、計13項目にわたる厳密な数学的・実験的修正、コード正本化、ドキュメント整合、キャッシュ削除、およびテスト検証がすべて完了しました。

---

## 主な変更点と成果物

### 1. V3 RQ1 Centered Projection Removal の中心点修正
- **問題**: 従来は `h_mean = np.mean(H_train, axis=0)` による全体平均が減算されており、情動条件と中立条件が混在していました。
- **対応**: train 側の matched-neutral 条件から中立中心 $\mu_{\mathrm{neu}} = E[h_{\mathrm{neutral}}]$ を個別に算出し、射影除去を $h' = h - Q Q^\top (h - \mu_{\mathrm{neu}})$ に修正しました。
- **対象ファイル**:
  - `v3/primary/run_rq1_state_induction.py`
  - `v3/scripts/run_v3_state_induction.py`
  - （確証的スクリプト `v3/primary/run_confirmatory_replication.py` は先行して適合済み）

### 2. V3 RQ1 Orthogonal Control ($d_{\perp}$) 実測と 3 条件 Specificity 評価
- **問題**: 直交統制空間 $Q_{\perp}$ を生成していたものの、実際の介入評価で実行されていませんでした。
- **対応**: $Q_{\perp}$ による $d_{\perp}$ 介入条件を実測実行し、`affective`, `orthogonal`, `neutral` の 3 条件について $\Delta \text{Valence}$, $\Delta \text{Arousal}$ を測定・記録。特異性比（Specificity Ratio）を実測値から算出するように実装しました。
- **対象ファイル**:
  - `v3/primary/run_rq1_state_induction.py`

### 3. V3 RQ1 Topic Control 指標の正規化・比較可能性改善
- **問題**: トピック統制指標がヒストグラムビン数依存の Jensen-Shannon 距離で記述されており、比較直観性に課題がありました。
- **対応**: $[0, 1]$ 基準の Total Variation Distance (TVD) を主指標として算出し、閾値 `topic_tvd_threshold: 0.15` と比較・評価するよう明記・改修しました。

### 4. V3 RQ1 Go/No-Go ゲートの Valence / Arousal 独立判定化
- **問題**: 単一の `decision: GO/NO-GO` 判定では、どちらか一方のみ有意・特異的であった場合の追跡が困難でした。
- **対応**: `decision_valence` および `decision_arousal` を独立に判定・記録し、総合判定とあわせて出力するように改修しました。

### 5. V3 Path Mediation の `pair_id` Group Split 化
- **問題**: Discovery と Confirmation のデータ分割で、同一テキストペア（affective / matched-neutral）が train/test に漏洩する懸念がありました。
- **対応**: `pair_id` をキーとした `GroupKFold` / `GroupShuffleSplit` による厳密な漏洩防止分割に移行しました。
- **対象ファイル**:
  - `v3/primary/run_rq3_path_mediation.py`
  - `v3/scripts/run_v3_path_mediation.py`

### 6. V3 Path Mediation 因果媒介用語の緩和
- **問題**: 自然言語処理モデルの内部表現に対する介入実験において、強い因果媒介推論を前提とする `NDE` (Natural Direct Effect) / `NIE` (Natural Indirect Effect) 表記は主張が強すぎる懸念がありました。
- **対応**: 測定実態に即した安全で厳密な用語へ移行しました：
  - `total_affective_shift`: 総自己報告変位 ($\Delta Y_{\mathrm{total}}$)
  - `residual_shift_mediator_blocked`: 媒介候補層の変位を中立射影でブロックした後の残差変位 ($\Delta Y_{\mathrm{blocked}}$)
  - `mediated_attenuation`: 媒介による減衰量 ($\Delta Y_{\mathrm{total}} - \Delta Y_{\mathrm{blocked}}$)
  - `attenuation_ratio`: 減衰比率 (Mediation Ratio)

### 7. V3 Path Mediation Discovery 内の探索的 site selection の明記
- **対応**: Discovery 段階における全層・全Token位置マップ探索は多重検定・データ依存を含む探索的（Hypothesis-generating）なサイト選定であることをコードの docstring および README に明記し、Confirmation 段階の独立 `pair_id` ホールドアウトで確証的検証を行う二段階構造を明確化しました。
- **対象ファイル**:
  - `v3/primary/run_rq3_path_mediation.py`
  - `v3/README.md`
  - `v3/primary/README.md`

### 8. Mistral Instruct チェックポイントの全 Stage 統一
- **問題**: `configs/models.yaml` で定義されている正本チェックポイント `mistralai/Mistral-7B-Instruct-v0.2` に対し、`configs/v3_experiments.yaml` で `v0.3` が混在していました。
- **対応**: `v3_experiments.yaml` の Instruct モデルを `v0.2` に統一しました。さらに、`v1/primary/phase_c/summarize_phase_c.py` のモデル指定をハードコードから `configs/models.yaml` の動的読み込みに改修しました。
- **対象ファイル**:
  - `configs/v3_experiments.yaml`
  - `v1/primary/phase_c/summarize_phase_c.py`

### 9. V1 Phase C Summary の表記修正
- **問題**: 解析スクリプトおよび出力ディレクトリ名に `response_onset` という語が含まれていましたが、実態はプロンプト最終トークン（Prompt-End）でした。
- **対応**: `v1/primary/phase_c/summarize_phase_c.py` 内の表記、出力パス (`v1/results/derived/v1_phase_c_prompt_end`) をすべて `prompt_end` / `Prompt-End` に整合させました。

### 10. V1 Phase B / E5 README と Primary 実装の整合
- **問題**: `v1/primary/run_phase_b.py` のコメントに S-BERT / VADER / PPL の記述が残存し、`v1/README.md` の E5 表記と食い違っていました。
- **対応**: 実際に実装されている Jaccard 類似度、Levenshtein 距離、および感情・トピック摂動統制（Perturbation Controls）の記述に完全統一しました。
- **対象ファイル**:
  - `v1/primary/run_phase_b.py`
  - `v1/README.md`

### 11. V2 Primary 構造の正本化
- **問題**: `v2/primary/run_rq1_rq2_cross_decoding.py` が `v2/scripts/` からのラッパーになっており、`sys.path.insert()` に依存していました。
- **対応**: 実装本体を `v2/primary/run_rq1_rq2_cross_decoding.py` へ移管し、共通パッケージ `affective_empathy_eval` を直接インポートする形へ正本化しました。`v2/scripts/run_v2_2x2_cross_decoding.py` は後方互換用ラッパーとしました。
- **対象ファイル**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`
  - `v2/scripts/run_v2_2x2_cross_decoding.py`

### 12. `v2/README.md` の UTF-8 化と旧 Neutralization ストーリー撤去
- **問題**: `v2/README.md` に不正な非 UTF-8 バイトが含まれており、また内容も旧 Neutralization 前提の記述となっていました。
- **対応**: Post-training Reorganization（Base vs. Instruct、4大 RQ1〜RQ4）を中心とする最新の設計に刷新し、クリーンな UTF-8 ファイルとして再配置しました。
- **対象ファイル**:
  - `v2/README.md`

### 13. キャッシュ削除、テスト堅牢化、README 数値固定の撤廃
- **対応**:
  - リポジトリ全体の `__pycache__` および `*.pyc` を完全に削除。
  - `tests/test_phase_c_tokenization_and_anchors.py` の先頭にあった不要な `from transformers import AutoTokenizer` を削除し、transformers 未導入環境でもテストコレクションが失敗しないよう堅牢化。
  - ルート `README.md` にあった「40/40 tests pass」という固定数値を「All tests should pass」に修正。

---

## 検証結果

### 1. 全テストスイートの実行 (`.venv/bin/pytest -q`)
```text
........................................                                                   [100%]
40 passed, 1 warning in 7.31s
```
すべてのテスト（40件）が問題なく PASS することを確認しました。

### 2. 主要スクリプトの dry-run スモークテスト
- **V3 RQ1 State Induction**:
  - `python v3/primary/run_rq1_state_induction.py --dry-run` $\rightarrow$ **正常終了 (Code 0)**
  - `decision_valence: GO`, `decision_arousal: GO` を独立して正しく判定・出力確認。
- **V3 RQ3 Path Mediation**:
  - `python v3/primary/run_rq3_path_mediation.py --dry-run` $\rightarrow$ **正常終了 (Code 0)**
  - Group split によるデータ漏洩防止、緩和された因果媒介指標でのサマリー出力を確認。
- **V2 RQ1 & RQ2 Cross Decoding**:
  - `python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run --family Qwen` $\rightarrow$ **正常終了 (Code 0)**
  - `v2/results/raw/v2_geometry_Qwen.json` および `v2/results/derived/v2_cross_family_summary.json` への出力を確認。
- **V1 Phase C Summary**:
  - `python v1/primary/phase_c/summarize_phase_c.py` $\rightarrow$ 例外なくモデル設定を動的ロードし、正常に待機状態を報告。

---

## 結論
以上の修正および検証により、コードベースは完全にクリーンかつ理論的に堅牢な状態となり、全再実行（GPU本実行）およびリポジトリ公開に万全の体制が整いました。
