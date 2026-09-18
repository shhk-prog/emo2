# 作業完了報告書 (Walkthrough): behavioral/primary/README.md の実行例整理

## 実施概要
ユーザーからの助言に従い、モデル ID の正本管理方針（`configs/models.yaml` を唯一の正本とする）を徹底するため、[`behavioral/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/README.md) の「## 実行」セクションを体系化しました。

---

## 修正内容

### [`behavioral/primary/README.md`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/README.md)
実行例の優先度と役割分担を明確化：
1. **推奨: 統合 CLI（root configs/models.yaml から一括実行）**
   - Primary 1-1.5B コホート全体の一括実行例：
     `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --device cuda:0`
   - 特定ファミリー（Qwen等）のみの実行例：
     `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --family qwen --device cuda:0`
2. **本番 Bash ランナー**
   - ログの自動記録・環境 activate を伴う本番実行例：
     `bash scripts/run_production_behavioral.sh cuda:0`
3. **個別スクリプト実行例（低レイヤ確認・手動検証用）**
   - 各スクリプト（EmoBank / AIPsy）に対する明示引数指定例をデバッグ・低レイヤ検証用として整理。

---

## 成果物ドキュメント
- タスク管理: [`docs/behavioral_primary_readme_refinement/task.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/behavioral_primary_readme_refinement/task.md)
- 実装計画: [`docs/behavioral_primary_readme_refinement/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/behavioral_primary_readme_refinement/implementation_plan.md)
- 作業完了報告書: [`docs/behavioral_primary_readme_refinement/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/behavioral_primary_readme_refinement/walkthrough.md)
