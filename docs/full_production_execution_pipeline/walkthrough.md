# 本番データ全件・4小型family一括実行パイプライン 完了報告 (Walkthrough)

## 概要
Behavioral, V1, V2, V3 の全4ステージにおいて、Primary 4小型family（Qwen 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B: Base + Instruct 計8モデル）を、本論文に使える本番データ規模（全件・サブサンプル制限なし）で確実に一括実行できるコードおよび実行スクリプト群を整備しました。

---

## 1. 実施した改修内容

### 1.1 スクリプト内のサブサンプル・リミット撤廃（フルデータ化）
テスト用の固定制限を撤廃し、引数なしまたは `--max-samples` 未指定時は常にデータセット全件を使用するよう改修しました：
- [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py): `--limit` default=0（0 = 全件実行）
- [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py): `--limit` default=0（0 = 全件実行）
- [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py): `--subsample` default=0（0 = 全件実行、全層×全ステージ走査）
- [run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py): `--subsample` default=0（0 = 全件実行、Discovery 500 / Confirmation 500 全件）
- [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py): `--subsample` default=0（0 = 全件実行、4 family 全モデル追試）
- [run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py): `--max-samples` が指定された時のみ下位スクリプトへ制限フラグを渡し、未指定時は常にフルデータで実行。

### 1.2 本番一括実行スクリプトの整備 (`scripts/`)
仮想環境の自動ロード、ログの自動保存（`results/logs/`）、実行時間の計測、エラー時の安全な停止を備えた実行スクリプトを作成し、実行権限（`chmod +x`）を付与しました：
- [run_production_all.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_all.sh): 全4ステージ（Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3）を順次実行するマスタースクリプト
- [run_production_behavioral.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_behavioral.sh): Behavioral ステージ（EmoBank 1,000 + AIPsy 480）
- [run_production_v1.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_v1.sh): V1 ステージ（Phase A $\rightarrow$ Phase B $\rightarrow$ Phase C E3/E4 $\rightarrow$ E6 $\rightarrow$ summarize）
- [run_production_v2.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_v2.sh): V2 ステージ（RQ1/RQ2 $\rightarrow$ RQ3 $\rightarrow$ RQ4）
- [run_production_v3.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_v3.sh): V3 ステージ（RQ1 Gate $\rightarrow$ RQ2 4-Maps $\rightarrow$ RQ3 Mediation $\rightarrow$ Confirmatory）

---

## 2. 本番データ規模の仕様

| ステージ | 対象データセット | サンプル数・条件 | 対象モデル |
|---|---|---|---|
| **Behavioral** | EmoBank (3-Way VAD)<br>AIPsy-Affect (4-Split) | 1,000 刺激 (全件)<br>480 刺激 (全件: 120 pairs $\times$ 4 split) | 4 family $\times$ 2条件 (Base, Instruct)<br>合計 8モデル |
| **V1** | Phase A: EmoBank + AIPsy<br>Phase B: Semantic Controls<br>Phase C: E3 Causal Map & E4 Patching<br>Phase C E6: Double Dissociation | 1,000 + 480 刺激 (全層)<br>400 刺激 (マッチド統制全件)<br>480 刺激 (全件, prompt-end)<br>480 刺激 (全件, 2x2 ablation) | 4 family $\times$ 2条件 (Base, Instruct)<br>合計 8モデル |
| **V2** | RQ1/RQ2: Cross-decoding & Geometry<br>RQ3: Causal Map & Dissociation<br>RQ4: Distribution Recovery Patching | 1,000 刺激 (700 train / 300 test)<br>1,000 刺激 (全層因果マッピング)<br>1,000 刺激 (全層・Wasserstein/EMD) | 4 family (Base vs Instruct 内部対比)<br>合計 4 family |
| **V3** | RQ1: State Induction & Gate<br>RQ2: Spatiotemporal 4-Maps<br>RQ3: Path Mediation (Discovery / Conf)<br>Step 7: Confirmatory Replication | 1,000 刺激 (全件 Gate 判定)<br>1,000 刺激 (全層探索)<br>1,000 刺激 (500 disc / 500 conf 全件)<br>1,000 刺激 (4 family 全モデル追試) | Target (Qwen) + Confirmatory 3 family (Llama, Gemma, OLMo) |

---

## 3. 動作検証結果

1. **単体テスト**:
   - `pytest -q` を実行し、**46 passed**（全件合格）を確認。
2. **ドライラン結合テスト**:
   - `bash scripts/run_production_all.sh cpu --dry-run` を実行。
   - Behavioral (8モデル) $\rightarrow$ V1 (8モデル) $\rightarrow$ V2 (4 family) $\rightarrow$ V3 (Target + 3 Confirmatory) の全パイプラインが正常に完走（Exit Code 0）することを確認。

---

## 4. 本番実行コマンド

本番データ（フルスケール）での実行は、GPU（H100/H200/A100等）を用いてターミナルで実行してください。
プロセスが途中で切断されないよう、`tmux` または `nohup` の使用を推奨します。

### 4.1 全ステージ一括実行（最も推奨）
```bash
# プロジェクトルートに移動
cd /mnt/nas/home/hiromi/src/emo2

# tmux セッションを開始する場合
tmux new -s emo_prod

# 全ステージ一括実行 (デバイス例: cuda:0)
bash scripts/run_production_all.sh cuda:0
```
※ バックグラウンド（nohup）で実行する場合：
```bash
nohup bash scripts/run_production_all.sh cuda:0 > results/logs/production_all_master.log 2>&1 &
# 進行状況確認:
tail -f results/logs/production_all_master.log
```

### 4.2 ステージ別実行
ステージごとに分けて実行することも可能です：

```bash
# 1. Behavioral のみ (所要時間目安: 約30分〜1時間)
bash scripts/run_production_behavioral.sh cuda:0

# 2. V1 のみ (所要時間目安: 約2〜4時間)
bash scripts/run_production_v1.sh cuda:0

# 3. V2 のみ (所要時間目安: 約2〜3時間)
bash scripts/run_production_v2.sh cuda:0

# 4. V3 のみ (所要時間目安: 約3〜5時間)
bash scripts/run_production_v3.sh cuda:0
```

### 4.3 Python 統合 CLI による直接実行
統合 CLI から直接実行することも可能です：
```bash
# 仮想環境の有効化
source .venv/bin/activate

# 例: 全ステージ実行
python -m affective_empathy_eval.run --stage all --model-set primary_small --device cuda:0

# 例: V3 ステージのみ実行
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0
```
