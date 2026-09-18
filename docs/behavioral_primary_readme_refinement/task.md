# タスク: behavioral/primary/README.md の実行例整理と最終確認

## 背景・目的
ユーザーからのアドバイスに基づき、`behavioral/primary/README.md` における実行例を整理し、「モデルIDは root `configs/models.yaml` が正本である」という設計方針を反映して、統合 CLI（`affective_empathy_eval.run`）および本番 Bash ランナーを Primary な実行例として前面に出す。個別スクリプト呼び出しは低レイヤ確認用として位置づける。

## タスクリスト
- [x] `docs/behavioral_primary_readme_refinement/task.md` および `implementation_plan.md` の作成
- [x] `behavioral/primary/README.md` の修正（統合 CLI による registry 駆動実行を最上位に配置）
- [x] `docs/behavioral_primary_readme_refinement/walkthrough.md` の作成
