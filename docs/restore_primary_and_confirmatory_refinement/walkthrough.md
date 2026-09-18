# 変更内容の確認 (Walkthrough): Primary 完全復元と本番検証環境の確立

## 1. 概要
整理アーカイブ処理に伴い一時的に消失していた V1 および V2 の Primary ソースコード（8ファイル）を Git 履歴（コミット `277e362`）から完全に復元し、`.gitignore` の再設定とキャッシュ削除を行いました。
さらに、本番実行時の安全性と再現性を担保するため、全 21 箇所のエントリポイント実体存在確認テストとドライランディスパッチテストを追加し、V3 Confirmatory Replication の科学的統制（layer-specific 注入および Arousal 時間的出現検証）を実装しました。

## 2. 実施内容と検証結果

### 2.1 Primary ソースコードの完全復元
以下のファイルがワークスペースに完全復元されました：
- [prepare_v1_phase_b_controls.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/prepare_v1_phase_b_controls.py)
- [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- [run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- [run_rq1_rq2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
- [run_rq3_causal_map.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py)
- [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)

> [!NOTE]
> `v1/primary/phase_c/run_e6_specialization.py` についても、直前のセッションで確立された「Task-Specific Causal Specialization / タスク選択性差コントラスト選定 / No-Go (Negative Result) 判定」の厳密版コードが完全に復元されていることを確認しました。

### 2.2 本番エントリポイント検証テストの追加
[tests/test_production_entrypoints.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_production_entrypoints.py) を新規作成し、以下の自動テストを実装：
- `test_all_primary_entrypoints_exist()`: 
  Behavioral, V1, V2, V3 の全 14 スクリプトおよび本番シェルスクリプト 7 本（計 21 ファイル）が実体として存在することを検証。
- `test_production_dry_run_dispatch()`:
  統合ランナー `affective_empathy_eval.run` が全ステージ（Behavioral, V1, V2, V3）を正常に認識し、ドライランで欠落なくサブスクリプトを呼び出せることを検証。

```bash
$ .venv/bin/pytest tests/test_production_entrypoints.py -v
tests/test_production_entrypoints.py::test_all_primary_entrypoints_exist PASSED [ 50%]
tests/test_production_entrypoints.py::test_production_dry_run_dispatch PASSED   [100%]
============================== 2 passed in 6.81s ==============================
```

### 2.3 V3 Confirmatory Replication の科学的統制精緻化
[v3/primary/run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py) において：
1. **Causal Profile $C(l)$ の layer-specific 化**:
   各層 $l$ の hidden states $H_l$ から局所情動ベクトル $d_V^{(l)} = \beta_V^{(l)} / \|\beta_V^{(l)}\|$ と局所スケール $\operatorname{std}(H_l d_V^{(l)})$ を推定して注入するよう改修。中間層ベクトルの他層不適合による効果減衰という交絡を完全に排除。
2. **Temporal Emergence (H4) の Arousal 拡張**:
   Valence だけでなく Arousal ($d_A$) についても生成トークン段階別の効果推移を測定し、`stage_causal_v`, `stage_causal_a`, `passed_valence`, `passed_arousal` を独立して出力・記録。

### 2.4 テストスイート全体検証
```bash
$ .venv/bin/pytest -q
............................................................                            [100%]
60 passed in 14.23s
```
全 60 件のテストが一切の回帰なく通過しました。

## 3. 本番全再実行コマンド一覧

本番のフルデータ（4小型ファミリー: Qwen, Llama, Gemma, OLMo）に対する実行コマンドは以下の通りです：

```bash
# 1. Behavioral Stage（EmoBank & AIPsy）
bash scripts/run_production_behavioral.sh

# 2. V1 Stage（Phase A / B / C E3, E4, E6 / Summarize）
bash scripts/run_production_v1.sh

# 3. V2 Stage（RQ1/RQ2, RQ3 Causal Map, RQ4 Recovery Patching, Confirmatory）
bash scripts/run_production_v2.sh

# 4. V3 Stage（RQ1 State Induction, RQ2 Spatiotemporal, RQ3 Mediation, Confirmatory Replication）
bash scripts/run_production_v3.sh
```

または統合 CLI によるステージ個別・一括実行：
```bash
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small
python -m affective_empathy_eval.run --stage v1 --model-set primary_small
python -m affective_empathy_eval.run --stage v2 --model-set primary_small
python -m affective_empathy_eval.run --stage v3 --model-set primary_small
```
