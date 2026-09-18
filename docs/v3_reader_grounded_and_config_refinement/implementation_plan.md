# 実装計画書: V3 情動方向の Reader-Grounded 化・因果介入サンプル数拡張・設定整理

## 1. 概要と研究上の位置づけ
本改修では、論文全体のストーリーと測定妥当性を飛躍的に強化するため、V3（Mechanistic State-to-Report）の主要分析における情動方向 $d_V, d_A$ を、モデル自身の自己報告（Self-report）からではなく、同じ刺激に対する**感情認知タスク（Reader Prediction）**から同定する「Reader-Grounded Affect Direction」へと刷新します。

これにより、4つの実験ステージが以下の通り完全に連結します：
$$\boxed{\text{Behavioral coupling}} \longrightarrow \boxed{\text{V1 shared representation}} \longrightarrow \boxed{\text{V2 reorganization}} \longrightarrow \boxed{\text{V3 Reader-grounded affect state} \longrightarrow \text{Self-report}}$$

V1で「他者感情認識（Reader）と自己報告（Self）が内部情動表現を部分共有している」ことを示した上で、V3では「独立に感情認知と同定された内部状態 $d^R$ が、生成過程のどこで自己報告に対する因果的効力を獲得するか」を直接実証する構造となります。従来の Self-derived 方向は Secondary（補助分析）として保持します。

あわせて、RQ2 Discovery における因果介入サンプル数を 5 件から 15 件へ増量し、設定ファイルの紛らわしい記述を整理します。

---

## 2. 変更対象と実装内容

### 2.1 `configs/v3_experiments.yaml` の整理
- `neutral_text_column: "neutral_text"`: loader-generated フィールドであることをコメントで明記。
- `spatiotemporal.n_causal_samples: 15`: 因果介入サンプル数を明示的に設定（推奨範囲 10〜20件）。

### 2.2 `v3/primary/run_rq1_state_induction.py`
- **Reader Prediction の取得**:
  train split の各刺激に対して `TaskType.READER` のプロンプトを提示し、モデルの感情認知期待値 $y_{V,\text{train}}^R, y_{A,\text{train}}^R$ を算出。
- **Primary 方向の刷新**:
  内部表現 $H_{\text{train}}$ から $y_{V,\text{train}}^R, y_{A,\text{train}}^R$ を予測するリッジ回帰により $d_V^R, d_A^R$ を同定。
- **介入と検証**:
  同定された $d_V^R, d_A^R$ を Self-report 処理に注入し、自己報告への十分性・必要性・特異性を検証。
- **Secondary 分析の保持**:
  従来の Self-derived 方向（$d_V^S, d_A^S$）およびその介入メトリクスも `secondary_self_derived` として結果辞書に保存。

### 2.3 `v3/primary/run_rq2_spatiotemporal_maps.py`
- **デコード・方向ターゲット**:
  各刺激に対する Reader Prediction $y_V^R, y_A^R$ を Primary デコード能 $D_V^R, D_A^R$ および局所介入方向 $d_V^R(l, s), d_A^R(l, s)$ のターゲットとする。
- **因果介入サンプル数の増量**:
  `sub_eval_idx = list(range(min(n_causal_samples, N)))`（既定 15 件）に変更。
- **Secondary 分析の保持**:
  従来の Self-report をターゲットとする $D_V^S, D_A^S$ 等も保持。

### 2.4 `v3/primary/run_rq3_path_mediation.py`
- **Discovery 候補層スクリーニング**:
  Discovery セットにおける情動ターゲットを Reader Prediction $y_V^R$ とし、Reader-grounded な情動状態 $d_{\text{stim}}^R$ を同定。
- **Confirmation 媒介推論**:
  $d_{\text{stim}}^R \rightarrow M \rightarrow Y$（Self-report）の自然間接効果（NIE）および媒介割合を検定。

### 2.5 `v3/primary/run_confirmatory_replication.py`
- **Confirmatory モデル検証**:
  他 3 ファミリー（Llama, Gemma, OLMo）においても、各モデルの Reader Prediction を取得して各層固有の $d_V^{R,(l)}, d_A^{R,(l)}$ を推定し、Self-report への介入・層解離・時間的出現を検証。
- **シミュレーション・実モデルのキー整合性**:
  `reader_grounded` を明記したメタデータと結果構造を統一。

---

## 3. 検証計画

### 自動テスト
- `python -m pytest tests/test_production_entrypoints.py -v`:
  エントリポイントの実体存在およびドライランディスパッチが正常に動作することを確認。
- `python -m pytest -q`:
  リポジトリ全体の 60 件のテストが通過することを確認。
- 各スクリプトのドライラン検証:
  `python -m v3.primary.run_rq1_state_induction --dry-run` 等の実行。

### 破壊的変更・互換性確認
- 従来の `y_v`, `y_a` を用いていた出力キーは上位互換を保ち、Primary に Reader-grounded メトリクス、Secondary に Self-derived メトリクスを配置して論文執筆時に両方を参照可能にします。
