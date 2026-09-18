# 実装計画: 全再実行・公開に向けた重要修正・堅牢化 (Pre-execution Refinements)

ユーザーから指摘された11点の必須修正事項および追加の改善点を反映し、リポジトリの全再実行可能性、測定妥当性、因果推論の厳密性、コードベースの美しさを完成形に引き上げます。

---

## 修正項目と技術方針

### 1. V3 RQ1 Centered Projection Removal の中心点修正
- **現状の問題**: `h_mean = np.mean(H_train, axis=0)` で中心化しており、情動刺激と中立刺激を区別しない全体の平均になっている。
- **改訂仕様**:
  - train 分割の matched-neutral 刺激から抽出した表現 $H_{\text{train, neutral}}$ の平均ベクトル $\mu_{\text{neu}} = \mathbb{E}[h_{\text{neutral}}]$ を算出。
  - テスト時の射影除去を以下のように厳格化：
    $$h' = h - Q Q^\top (h - \mu_{\text{neu}})$$
  - `v3/primary/run_rq1_state_induction.py`, `v3/scripts/run_v3_state_induction.py`, `v3/primary/run_confirmatory_replication.py` に適用。

### 2. V3 RQ1 Orthogonal Control の実測と Specificity 評価
- **現状の問題**: `d_perp` (orthogonal norm-matched direction) を生成しているが、実際の Specificity 評価は $d_V$ vs $d_{\text{rand}}$ のみとなっている。
- **改訂仕様**:
  - affect ($d_V$), random ($d_{\text{rand}}$), orthogonal ($d_{\perp}$) の3条件を実測。
  - Specificity を $\text{Effect}_{\text{affect}} > \max(\text{Effect}_{\text{random}}, \text{Effect}_{\perp})$ および個別差分 $\Delta_{\text{rand}} = \text{Effect}_{\text{affect}} - \text{Effect}_{\text{random}}$、$\Delta_{\perp} = \text{Effect}_{\text{affect}} - \text{Effect}_{\perp}$ として算出・報告。

### 3. V3 RQ1 Topic Control 指標の正規化・比較可能性
- **現状の問題**: Self-report は $|\Delta \mathbb{E}[V]|$（$[-1, 1]$ 基準）、Topic control は $\|\Delta \text{logits}\|$ であり、尺度が異なるため差分を直接取るのが科学的に不適切。
- **改訂仕様**:
  - Topic control 側も分類確率分布の変化（Total Variation Distance または Top-1 topic probability shift $|\Delta P(\text{topic})| \in [0, 1]$）を測定。
  - 確率・正規化スケールで統一し、科学的に整合した Task Selectivity を算出。

### 4. V3 RQ1 Go/No-Go ゲートの Valence / Arousal 独立判定
- **改訂仕様**:
  - 現在の $V_{\text{slope}} > 0.1 \land A_{\text{slope}} > 0.1$ の一律制約を改め、`decision_v: GO / NO_GO`、`decision_a: GO / NO_GO` を独立判定し、総合結果として `GO (both)`, `GO (valence-only)`, `GO (arousal-only)`, `NO_GO` を出力。

### 5. V3 Path Mediation の Discovery/Confirmation を pair_id Group Split 化
- **現状の問題**: `indices = np.arange(len(df)); rng.shuffle(indices)` により行単位で半々に分割しているため、AIPsy の同一 matched pair が Discovery と Confirmation に跨るリスクがある。
- **改訂仕様**:
  - データセットに `pair_id` が存在する場合、`pair_id` のユニーク値単位で 50/50 Group Split を実行。同一ペアが両分割に混入する循環性を完全に排除。

### 6. V3 Path Mediation の因果媒介用語 (NDE/NIE) の緩和
- **改訂仕様**:
  - Pearl流の厳密な Natural Direct/Indirect Effect との混同を避け、安全かつ正確な用語に変更：
    - Total effect $\rightarrow$ **Total affective shift**
    - Natural direct effect $\rightarrow$ **Residual shift after mediator blocking**
    - Natural indirect effect $\rightarrow$ **Mediated attenuation**
    - Mediation ratio $\rightarrow$ **Attenuation ratio**

### 7. V3 Path Mediation Discovery 内の探索的 site selection の明記
- **改訂仕様**:
  - Discovery 内での方向推定・介入評価について「探索的 site selection」である旨を明記。Confirmation 側が完全独立であることをドキュメントとログで強調。

### 8. Mistral Instruct checkpoint の全 Stage 統一 & 動的ロード化
- **改訂仕様**:
  - `configs/models.yaml`: `mistralai/Mistral-7B-Instruct-v0.2` を正本とする。
  - `configs/v3_experiments.yaml`: `v0.3` を `v0.2` に統一。
  - `v1/primary/phase_c/summarize_phase_c.py`: ハードコードされた `MODELS = [...]` を撤去し、`configs/models.yaml` をパースして動的構築するように改修。

### 9. V1 Phase C Summary の表記修正
- **改訂仕様**:
  - `summarize_phase_c.py` 内の旧「Response-Onset」表記を最新の「Prompt-End」に修正。

### 10. V1 Phase B / E5 README と Primary 実装の整合
- **改訂仕様**:
  - `v1/README.md` の E5 記述から実装されていない S-BERT, VADER, PPL を削除。
  - 実装されている語彙交絡監査（Jaccard, Levenshtein）および統制実験（Original, Paraphrase, Word shuffle, Polarity reversal）に完全整合させる。

### 11. V2 Primary 構造の正本化と sys.path.insert の排除
- **改訂仕様**:
  - `v2/scripts/run_v2_2x2_cross_decoding.py` の実装本体を `v2/primary/run_rq1_rq2_cross_decoding.py` へ移動し正本化。
  - `v2/scripts/run_v2_2x2_cross_decoding.py` は後方互換ラッパーとし、Primary 側の不要な `sys.path.insert` を解消。

### 12. V2 README の UTF-8 再保存 & 旧 Neutralization ストーリーの撤去
- **改訂仕様**:
  - `v2/README.md` の破損バイトを修復し、クリーンな UTF-8 で再保存。
  - 旧「自己報告の抑制・中立化」ストーリーを撤去し、現行の Post-training reorganization (RQ1〜RQ4) を中心とした構成へ全面書き直し。
  - 旧 scripts は「Legacy / Exploratory experiments」として末尾に簡潔にリンク。

### 13. リポジトリ全体のクリーンアップ & ドキュメント微修正
- **改訂仕様**:
  - `find . -type d -name '__pycache__' -prune -exec rm -rf {} +` および `.pyc` ファイルを完全削除。
  - root `README.md` の「40/40 100% pass」固定数を「All tests should pass」等に修正。
  - `v3/README.md` 等に Spatiotemporal 全探索が Discovery 用途である旨を明記。
  - `tests/test_phase_c_tokenization_and_anchors.py` で `pytest.importorskip("transformers")` を適用。

---

## 変更対象ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `v3/primary/run_rq1_state_induction.py` | [MODIFY] | neutral mean 中心化, orthogonal control 実測, topic prob shift, V/A 独立 gate |
| `v3/scripts/run_v3_state_induction.py` | [MODIFY] | 上記と同等の修正 |
| `v3/primary/run_confirmatory_replication.py` | [MODIFY] | neutral mean 中心化の反映 |
| `v3/scripts/run_v3_confirmatory_replication.py` | [MODIFY] | neutral mean 中心化の反映 |
| `v3/primary/run_rq3_path_mediation.py` | [MODIFY] | pair_id Group split, NDE/NIE用語緩和, neutral mean, 探索的記述 |
| `v3/scripts/run_v3_path_mediation.py` | [MODIFY] | 上記と同等の修正 |
| `configs/v3_experiments.yaml` | [MODIFY] | Mistral Instruct v0.2 統一, orthogonal control設定 |
| `v1/primary/phase_c/summarize_phase_c.py` | [MODIFY] | Prompt-End表記, configs/models.yaml 動的ロード |
| `v1/README.md` | [MODIFY] | E5 語彙交絡記述の実コード整合 |
| `v2/primary/run_rq1_rq2_cross_decoding.py` | [MODIFY] | 実装本体を統合し正本化, sys.path.insert 排除 |
| `v2/scripts/run_v2_2x2_cross_decoding.py` | [MODIFY] | Primary への後方互換呼び出し |
| `v2/README.md` | [MODIFY] | UTF-8再保存, Post-training reorganizationへの書き換え |
| `v3/README.md` | [MODIFY] | Spatiotemporal 全マップ探索の Discovery 用途明記 |
| `README.md` | [MODIFY] | 40/40 固定テスト数記述の削除 |
| `tests/test_phase_c_tokenization_and_anchors.py` | [MODIFY] | transformers 非依存環境での堅牢化 |

---

## 検証手順
1. **pytest スイート実行**:
   `.venv/bin/pytest -q`
2. **各ステージ dry-run 実行**:
   - `python v1/primary/run_phase_b.py --dry-run` (または軽量確認)
   - `python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run`
   - `python v3/primary/run_rq1_state_induction.py --dry-run`
   - `python v3/primary/run_rq3_path_mediation.py --dry-run`
3. **文字コード・不要ファイル検査**:
   - `file -i v2/README.md` で UTF-8 認識確認
   - `find . -name '__pycache__' -o -name '*.pyc'` で不要ファイルゼロ確認
