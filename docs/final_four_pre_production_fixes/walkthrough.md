# 本番前最終4点ピンポイント修正 完了報告 (Walkthrough)

本番全実行を安全に開始するため、指摘された4点の問題（Chat template fallback、EmoBank JSON上書き、EmoBank CVテキスト重複リーク、V2 cross-family両軸集約）をすべて修正しました。

---

## 1. 実施した修正の詳細

### ① V1 Phase B / Phase C の chat-template fallback 修正 & 単体テスト追加
- **問題点**: `except Exception:` 内で元の `messages` (system+user) を再度渡しており、system role を非受容のテンプレート（Gemma系など）で本番停止するリスクがあった。
- **実施内容**:
  - `v1/primary/run_phase_b.py`: `except Exception:` 内で `[{"role": "user", "content": user_content}]` のみにフォールバックするよう修正。
  - `v1/primary/run_phase_c.py`: 同様に `[{"role": "user", "content": user_content}]` のみにフォールバックするよう修正。
  - `tests/test_pre_production_fixes.py`: `test_chat_template_system_role_fallback()` を追加し、system非受容tokenizerにおいて例外を捕捉してuser-onlyでテンプレート適用が成功することを自動検証。

---

### ② V1 Phase A の modular JSON が EmoBank 結果を上書きする問題の修正
- **問題点**: `--dataset both` 実行時に、最初に保存された `v1_e1_decodability_{model_prefix}.json` の EmoBank 結果が、後続の AIPsy 保存によって丸ごと上書きされて消失していた。
- **実施内容**:
  - `v1/primary/run_phase_a.py`:
    - `e1_payload = {}` を初期化。
    - EmoBank 完了時に `e1_payload["emobank"] = e1_emobank_records` を格納。
    - AIPsy 完了時に `e1_payload["aipsy_primary"] = ...` 等を格納。
    - Phase A の末尾で1回だけ `save_experiment_result()` を実行。
    - dry-run においても同一のスキーマ（`e1_dry_payload`）で1回だけ保存するように統一。

---

### ③ V1 EmoBank の CV における同一テキスト leakage 修正
- **問題点**: 実データ1,000件中に存在する同一テキスト（"Dear Name:", "Sincerely,"）が異なるIDを振られていたため、`group_ids = df_emobank["id"].values` では別foldに分配される微小なリーク（4/1000件）が存在した。
- **実施内容**:
  - `v1/primary/run_phase_a.py`:
    ```python
    content_groups = (
        df_emobank["text"]
        .astype(str)
        .str.normalize("NFKC")
        .str.strip()
    )
    group_ids = content_groups.values
    ```
    に修正し、同一テキスト内容が必ず同一foldに割り当てられるように厳格化。

---

### ④ V2 cross-family summary の Valence / Arousal 両軸集約化
- **問題点**: 各層データでは両軸を計算しているにもかかわらず、`summary_metrics` および `v2_cross_family_summary.json` が実質 Valence のみであり、キー名にも `valence` が明示されていなかった。
- **実施内容**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`:
    - 各ファミリーの `results["summary_metrics"]` に `by_axis: {"valence": {...}, "arousal": {...}}` を新設し、Valence と Arousal の両軸で重心・ピーク・delta_sharing を個別に算出。
    - 4ファミリー統合の `v2_cross_family_summary.json` においても、`by_axis: {"valence": {...}, "arousal": {...}}` 配下で両軸を別々に cross-family bootstrap 集約。
    - 既存のルートキー（`peak_depth_base_cross`, `peak_depth_inst_cross`, `mean_delta_sharing` 等）は Valence alias として維持し、後方互換性を100%保持。

---

## 2. 検証と本番実行手順

```bash
source .venv/bin/activate
python --version  # 3.12.x
pytest -q
ruff check src behavioral v1/primary v2/primary v3/primary tests

bash scripts/run_production_behavioral.sh cpu --dry-run
bash scripts/run_production_v1.sh cpu --dry-run
bash scripts/run_production_v2.sh cpu --dry-run
bash scripts/run_production_v3.sh cpu --dry-run
```

上記が完走した後、GPU環境（`cuda:0`）で本番全実行を開始してください。
