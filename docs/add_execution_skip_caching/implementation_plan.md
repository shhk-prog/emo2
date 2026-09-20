# 全ステージにおける実行結果自動スキップ機能の実装計画

Behavioral, V1, V3 RQ1 の各評価スクリプトにおいて、正常終了済みの成果物およびマニフェストが存在する場合に重みロードと推論を即座にスキップ（既存結果を再利用）する機能と、強制再実行のための `--force` オプションを追加します。

## 背景と目的
- **背景**: V2 および V3 (RQ2, RQ3, Confirmatory) はすでにマニフェスト照合による完全自動スキップが実装されていますが、Behavioral と V1、V3 RQ1 は既存成果物があってもモデルロードと推論が再実行される実装になっていました。
- **目的**:
  1. 既に完了したステージ・モデルの無駄な再計算時間を防ぐ。
  2. プロセス中断時に同一コマンドを再実行するだけで、未完了のモデルのみを安全に続行（冪等実行）できるようにする。
  3. 明示的に再計算したい場合のために `--force` 引数を提供する。

---

## 提案される変更点

### 1. Behavioral ステージ
#### [MODIFY] [`run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- `--force` 引数を追加。
- モデルロード前に、出力先 `out_csv` (`{args.tag}_3way_vad.csv`) の存在チェックと検証を実施。
  - ファイルが存在し、行数が期待値以上（破損していない）場合、モデルロードおよび評価ループをスキップして完了ログを出力し、要約ステップへ進む。

#### [MODIFY] [`run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
- `--force` 引数を追加。
- モデルロード前に、出力先 `out_csv` (`{args.tag}_aipsy_4split.csv`) の存在・有効性をチェックし、存在時はモデルロードをスキップ。

---

### 2. V1 ステージ
#### [MODIFY] [`run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- `--force` 引数を追加。
- モデルロード前に、`manifest.json` および出力ファイル（`e1_emobank_regression.csv`, `e2_cross_decoding.csv` 等）の存在を検証。有効な場合はスキップ。

#### [MODIFY] [`run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- `--force` 引数を追加。
- 成果物（`e5_semantic_controls.csv`, `manifest.json`）の存在・有効性チェックによる早期スキップ。

#### [MODIFY] [`run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- `--force` 引数を追加。
- 成果物（`e3_causal_map.csv`, `e4_interchangeability.csv`, `manifest.json`）の存在・有効性チェックによる早期スキップ。

#### [MODIFY] [`run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- `--force` 引数を追加。
- 成果物（`e6_specialization.csv`, `manifest.json`）の存在・有効性チェックによる早期スキップ。

---

### 3. V3 ステージ
#### [MODIFY] [`run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)
- `--force` 引数を追加。
- `out_raw` (`v3_rq1_results.json`) と `out_gate` (`v3_gate_decision.json`)、および `manifest_rq1_{fam_key}.json` が存在し、マニフェストが一致している場合、モデルロードおよび介入実験をスキップして既存 Gate 判定を読み込み。

---

### 4. 統合ランナー & シェルスクリプト
#### [MODIFY] [`affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `--force` オプションを追加。
- 各ステージ呼び出し関数（`run_behavioral`, `run_v1`, `run_v2`, `run_v3`）において、`--force` が渡された場合は子スクリプトに伝達。

---

## 検証計画
### 自動テスト
- `python -m py_compile ...`: 対象全スクリプトの構文検証。
- `python -m pytest -q -m "not slow"`: 既存テスト（83件）の通過確認。

### 動作検証（スモークテスト）
- `--dry-run` を用いて、
  1. 1回目の実行: 成果物生成。
  2. 2回目の実行: `Loaded existing results ... Skipping computation.` がログされ、計算がスキップされることを確認。
  3. `--force` を付けた3回目の実行: スキップされずに再計算が行われることを確認。
