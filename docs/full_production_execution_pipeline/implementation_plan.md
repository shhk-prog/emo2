# 本番データ全件・4小型family一括実行パイプライン実装計画

## 概要
Behavioral, V1, V2, V3 の全4ステージにおいて、Primary 4小型family（Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B）を、論文発表に耐えうる本番データ規模（サブサンプル制限のないフルデータ）で確実に一括実行できるコードおよび実行スクリプト・コマンドを整備します。

---

## 1. 本番データ仕様・サンプル数定義

| ステージ | 対象データセット | サンプル数・条件 | 対象モデル |
|---|---|---|---|
| **Behavioral** | EmoBank (3-Way VAD)<br>AIPsy-Affect (4-Split) | 1,000 刺激 (全件)<br>480 刺激 (全件: 120 pairs $\times$ 4 split) | 4 family $\times$ 2条件 (Base, Instruct)<br>合計 8モデル |
| **V1** | Phase A: EmoBank + AIPsy<br>Phase B: Semantic Controls<br>Phase C: E3 Causal Map & E4 Patching<br>Phase C E6: Double Dissociation | 1,000 + 480 刺激 (全層)<br>400 刺激 (マッチド統制全件)<br>480 刺激 (全件, prompt-end)<br>480 刺激 (全件, 2x2 ablation) | 4 family $\times$ 2条件 (Base, Instruct)<br>合計 8モデル |
| **V2** | RQ1/RQ2: Cross-decoding & Geometry<br>RQ3: Causal Map & Dissociation<br>RQ4: Distribution Recovery Patching | 1,000 刺激 (700 train / 300 test)<br>1,000 刺激 (全層因果マッピング)<br>1,000 刺激 (全層・Wasserstein/EMD) | 4 family (Base vs Instruct 内部対比)<br>合計 4 family |
| **V3** | RQ1: State Induction & Gate<br>RQ2: Spatiotemporal 4-Maps<br>RQ3: Path Mediation (Discovery / Conf)<br>Step 7: Confirmatory Replication | 1,000 刺激 (全件 Gate 判定)<br>1,000 刺激 (全層探索)<br>1,000 刺激 (500 disc / 500 conf 全件)<br>1,000 刺激 (4 family 全モデル追試) | Target (Qwen) + Confirmatory 3 family (Llama, Gemma, OLMo) |

---

## 2. 修正・整備方針

### 2.1 スクリプト内のサブサンプル・リミット撤廃
1. **`v1/primary/run_phase_c.py`**:
   - 現状 `--limit` default=20 になっているのを、default=0（0 = 全件実行）に変更。
   - 明示的に `--limit` が指定された場合のみサブサンプルを行う。
2. **`v1/primary/phase_c/run_e6_specialization.py`**:
   - 現状 `--limit` default=20 になっているのを、default=0（0 = 全件実行）に変更。
3. **`v3/primary/run_rq3_path_mediation.py`**:
   - 現状 `--subsample` default=40 になっているのを、default=0（0 = 全件実行）に変更。
   - `if subsample > 0:` の場合のみ head(subsample) し、0 または指定なしの場合は全件を使用。
4. **`v3/primary/run_confirmatory_replication.py`**:
   - 現状 `--subsample` default=20 になっているのを、default=0（0 = 全件実行）に変更。
   - 同様に 0 の場合は全件評価データを使用。
5. **`src/affective_empathy_eval/run.py`**:
   - `--max-samples` が指定されない限り、各スクリプトに制限（limit）を一切渡さず、本番データ全件で実行することを保証。

### 2.2 本番一括実行スクリプトの整備
本番実行は GPU リソースを数時間〜十数時間消費するため、以下のスクリプトを `scripts/` に作成します：
- **`scripts/run_production_all.sh`**:
  - 引数として GPU デバイス（例: `cuda:0` や `0`）を受け取り、Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3 を順次一括実行。
  - 各ステージの開始・終了時刻、ログファイル (`results/logs/production_*.log`) への Tee 出力、エラー時の安全な中断処理。
- **ステージ別実行スクリプト**:
  - `scripts/run_production_behavioral.sh`
  - `scripts/run_production_v1.sh`
  - `scripts/run_production_v2.sh`
  - `scripts/run_production_v3.sh`

---

## 3. 検証計画

### 3.1 コード検証
- 各スクリプトの引数・デフォルト値の修正確認。
- `.venv/bin/pytest -q` で単体テスト 46 件が全て通過することを確認。

### 3.2 ドライラン検証
- `python -m affective_empathy_eval.run --stage all --model-set primary_small --dry-run` を実行し、4 family 全体がエラーなく一括走査されることを確認。

---

## 4. ユーザー確認事項
- GPU の利用（CUDA デバイス番号、例: `cuda:0`）について確認します。
- スクリプト準備後、ターミナルで即座に実行できるコマンド形式（`nohup` や `tmux` でのバックグラウンド実行を推奨）を案内します。
