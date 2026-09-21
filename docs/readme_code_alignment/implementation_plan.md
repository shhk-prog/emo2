# 実装計画：README を現行コードに合わせる

## 方針

科学的な階層（Central RQ、4-stage、IUT、729 vs 81）は維持する。今回は **運用正本** をコードに揃える。

## 現行コードで README に必須の差分

1. `configs/models.yaml` は HF ID だけでなく **pinned `*_revision` SHA** と `inference_dtype: bfloat16` を持つ。
2. 統合 CLI は `--force` / `--all-layers` / `--batch-size` を受け付ける。
3. `scripts/run_production_v1.sh` は常に `--all-layers` を付与し、全 production スクリプトは `EXTRA_ARGS` を転送する。
4. V1 Phase B は `--task-type reader` の後に `--task-type self` を別実行する。
5. V2 は RQ4 後、`--family` 未指定なら confirmatory を自動実行する。
6. Behavioral は dry-run でも実スクリプトを呼び、完了後に `summarize_behavioral_outputs.py` を自動実行する。
7. V3 dry-run の成果物と gate は `v3/results/derived/dry_run/` に隔離する。
8. `sequence_likelihood.normalize_length: true` が V1/V2/V3 設定の正本である。
9. V3 confirmatory は YAML の frozen Qwen ブロックを読む。
10. `resolve_joint_stage_index` は `response_start = cand_start - 1`（= prompt_end）、他段階は `cand_start + offset`。

## 手順

1. ルート README の CLI / モデルレジストリ / 設定正本を更新する。
2. 各 Stage README と `*/primary/README.md` に同じ運用差分を入れる。
3. `scripts/README.md` に `EXTRA_ARGS` と V1 `--all-layers` を明記する。
4. walkthrough に照合表を残す。
