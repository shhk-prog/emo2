# 実装計画: behavioral/primary/README.md の実行例整理

## 概要
`behavioral/primary/README.md` の実行例について、モデル ID を直接指定する個別スクリプト呼び出しに先立ち、root `configs/models.yaml` を正本として自動解決する統合 CLI（`affective_empathy_eval.run`）および本番 Bash ランナー（`run_production_behavioral.sh`）を Primary な実行方法として提示するよう整理します。

---

## 変更内容

### [MODIFY] [`behavioral/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/README.md)
- 「## 実行」セクションを体系化：
  1. **推奨: 統合 CLI（root configs/models.yaml から一括実行）**
     - 全 Primary コホート実行: `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --device cuda:0`
     - 単一ファミリー実行: `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --family qwen --device cuda:0`
  2. **本番 Bash ランナー**
     - `bash scripts/run_production_behavioral.sh cuda:0`
  3. **個別スクリプト実行例（低レイヤ確認・手動検証用）**
     - 既存の `run_behavioral_emobank.py` / `run_behavioral_aipsy.py` の明示呼び出し例を補助として配置。

---

## 検証計画
- ドキュメント内の markdown フォーマットおよびリンクの整合性を確認。
