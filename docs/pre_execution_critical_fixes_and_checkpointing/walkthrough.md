# 再実行前クリティカル修正および逐次保存・レジリエンス強化 完了報告 (Walkthrough)

## 概要
ユーザーから指摘された全再実行前の実質的問題（V3 ModelRegistry KeyError、欠落した V1 Phase B 入力、Transformers 下限）を完全に解決し、5大必須修正、精度・整合性向上、および「途中で落ちても問題ない逐次保存・レジリエンス（チェックポイント＆レジューム）」を全ステージに実装・検証しました。

---

## 1. 実施した修正・強化一覧

### 1.1 V3 ModelRegistry 逆引き・アダプタ解決の修正
- **[registry.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/registry.py)**:
  - `get_family_by_model_id(model_id: str)` を追加。
  - `get_family(family_id: str)` にも model_id による自動フォールバックを実装。
- **[adapters.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/adapters.py)**:
  - `get_model_adapter(model, adapter_name)` に `ModelFamilyConfig` インスタンスが直接渡された場合でも属性（`.adapter` や `.family_id`）から自動解決できるよう堅牢化。
- **V3 スクリプト群**:
  - `run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py` で `fam_cfg = registry.get_family_by_model_id(model_id)` を呼び出すよう修正。

### 1.2 欠落していた V1 Phase B 入力データの生成自動化
- **[prepare_v1_phase_b_controls.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/prepare_v1_phase_b_controls.py)**（新規作成）:
  - AIPsy (`aipsy_4split_all.csv`) から同一 `pair_id` を持つ 192 ペアの clinical と neutral を抽出し、決定論的変換（Paraphrase, Word Shuffle, Outcome Reversal）を実行。
  - [v1_e5_semantic_controls.csv](file:///mnt/nas/home/hiromi/src/emo2/v1/data/processed/v1_e5_semantic_controls.csv) を生成・配置完了。
- **自動前処理組み込み**:
  - `scripts/run_production_v1.sh` および `src/affective_empathy_eval/run.py` の V1 実行冒頭で本 CSV の存在を確認し、存在しなければ自動生成するよう連携。

### 1.3 Transformers 依存下限の引き上げ
- **[pyproject.toml](file:///mnt/nas/home/hiromi/src/emo2/pyproject.toml)** & **[requirements.txt](file:///mnt/nas/home/hiromi/src/emo2/requirements.txt)**:
  - OLMo 2 (4.48+ / 4.50.3) および Gemma 3 (4.50+) の公式要件を満たすため、`transformers>=4.40.0` から `transformers>=4.50.3` へ引き上げ。

### 1.4 V1 Phase B の pair-aware CV 化 & 理論的解釈の明記
- **[run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)**:
  - `StratifiedKFold` を `StratifiedGroupKFold(groups=pair_id)` に変更し、対照ペア間のデータリーケージを完全に遮断。
  - Original probe を全データ fit して Paraphrase/Reversal へ適用する部分は「未知 pair への一般化ではなく、Original で定義した決定境界に対する変換感度（transformation sensitivity）を見る解析である」旨をコメント・docstring に明記。

### 1.5 V3 実モデル経路 Registry Integration Test の追加
- **[test_model_registry_and_adapters.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_model_registry_and_adapters.py)**:
  - `test_registry_get_family_by_model_id`: 4 family (Base/Instruct 計8モデル) + Mistral 7B の全 HF model ID から family config が正確に逆引きできることを検証。
  - `test_v3_entrypoints_with_mock_model`: Mock model を用いて V3 の活性化フックおよびアダプタ解決が正常に動作することを検証。

### 1.6 精度・整合性向上
- **[run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)**:
  - Held-out $D(l)$ 算出時に `GroupKFold(groups=pair_id)` を使用するよう統一。
- **[run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)**:
  - 出力ファイル名を `v3_discovery_spatiotemporal_maps_{fam_key}.json` に変更。
  - 全サンプル数 $N_{\mathrm{total}}$ と介入サンプル数 $N_{\mathrm{intervene}}$ を結果辞書に分離保存。
- **V1 Phase C wrapper**:
  - `run_e3_causal_map.py` および `run_e4_interchangeability.py` の `sys.path.insert` を撤廃し、クリーンなサブプロセス実行ラッパーに改修。

### 1.7 逐次保存・レジリエンス (Checkpointing / Fault Tolerance)
- **Behavioral ステージ**:
  - `run_behavioral_emobank.py` / `run_behavioral_aipsy.py`: 10サンプルごとに中間 CSV（`*_checkpoint.csv`）を自動フラッシュ保存。再実行時は処理済みサンプルを自動スキップして続きから再開（Resume）。
- **V1 Phase A**:
  - レイヤー完了ごとに `e1_emobank_decodability.csv` および `e2_emobank_geometry.csv` を即時フラッシュ。
- **V2 (RQ1〜RQ4)**:
  - ファミリー完了ごとに即時 JSON 保存。再実行時は完了済みファミリーを自動認識してスキップ。
- **V3 (RQ2, RQ3, Confirmatory)**:
  - 出力ファイル存在時の自動スキップ・レジュームを実装。

---

## 2. 検証結果

1. **単体テスト**:
   - `pytest -q` を実行し、**52 passed**（全テスト完全合格）を確認。
2. **結合ドライランテスト**:
   - `bash scripts/run_production_all.sh cpu --dry-run` を実行し、Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3 の全ステージ・全モデルがエラーなく完走（Exit Code 0）することを確認。

---

## 3. 本番実行コマンド

GPU 環境で以下のコマンドを実行してください：

```bash
cd /mnt/nas/home/hiromi/src/emo2

# tmux セッションを開始
tmux new -s emo_prod

# 全ステージ一括実行（デバイス指定: 例 cuda:0）
bash scripts/run_production_all.sh cuda:0
```
※ 途中で万一中断された場合でも、実装されたチェックポイントとレジューム機能により、再実行時に完了済みモデル・サンプルが安全にスキップされ、続きから自動再開されます。
