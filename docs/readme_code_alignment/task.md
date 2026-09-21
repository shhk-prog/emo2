# タスク：現行コードに合わせて各 README を詳細修正

## 目的

現行 Primary 実装（`src/affective_empathy_eval/`、`configs/*.yaml`、`scripts/run_production_*.sh`、各 `*/primary/`）に合わせて、ルートおよび Behavioral / V1 / V2 / V3 の README を詳細化する。論文用の階層説明は維持し、コードと食い違う実行フラグ・キャッシュ・dtype・revision・dry-run 隔離・自動後処理を正本に合わせる。

## 対象 README

- `README.md`
- `behavioral/README.md`, `behavioral/primary/README.md`
- `v1/README.md`, `v1/primary/README.md`
- `v2/README.md`, `v2/primary/README.md`
- `v3/README.md`, `v3/primary/README.md`
- `scripts/README.md`

## 変更しないもの

- 実験結果の捏造
- GPU 本番実行
- `conference2.tex` の本文改稿
- 既存の `docs/readme_*` 履歴フォルダの上書き
