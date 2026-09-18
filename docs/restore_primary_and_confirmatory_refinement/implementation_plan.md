# 実装計画書: Primary ソース復元・本番エントリポイント検証・V3 Confirmatory 統制精緻化

## 1. 目的と概要
本作業では、前回の整理アーカイブ作業時に誤って除外・欠落していた V1 および V2 の Primary ソースコード（8ファイル）を完全復元し、リポジトリ全体の整合性を再構築する。
さらに、本番実行時にファイル欠落や呼び出し不整合による中断が二度と起きないよう、全 21 entrypoint の存在保証テストおよび統合 CLI のファンアウト検証テストを新設する。
また、V3 Confirmatory Replication において、層解離（H1）および時間的出現（H4）の測定妥当性を高めるための統制（layer-specific な局所ベクトル推定、Valence/Arousal 独立時間検証）を実施する。

## 2. 変更対象と実装内容

### 2.1 消失ソースコードの Git 復元
コミット `277e362` より以下のスクリプトを復元：
- `v1/primary/prepare_v1_phase_b_controls.py`
- `v1/primary/run_phase_a.py`
- `v1/primary/run_phase_b.py`
- `v1/primary/run_phase_c.py`
- `v1/primary/phase_c/run_e6_specialization.py`
- `v2/primary/run_rq1_rq2_cross_decoding.py`
- `v2/primary/run_rq3_causal_map.py`
- `v2/primary/run_rq4_recovery_patching.py`

### 2.2 `.gitignore` の再整備
- `__pycache__/`, `*.py[cod]`, `.pytest_cache/`
- `results/raw/*`, `results/derived/*`, `results/figures/*` (ただし `.gitkeep` は維持)
- 仮想環境 `.venv/`, `.env`

### 2.3 本番エントリポイント検証テスト (`tests/test_production_entrypoints.py`)
- `test_all_primary_entrypoints_exist()`:
  Behavioral (2), V1 (5), V2 (3), V3 (4), Production Shell Scripts (7) の計 21 ファイルの実体存在をアサート。
- `test_production_dry_run_dispatch()`:
  `python -m affective_empathy_eval.run` の各 stage (`--stage v1`, `--stage v2`, `--stage v3`, `--stage behavioral`) について、dry-run モードで全サブコマンドが正常にファンアウト・ディスパッチされることを検証。

### 2.4 V3 Confirmatory Replication の統制精緻化 (`v3/primary/run_confirmatory_replication.py`)
- **H1 (Causal Profile) の layer-specific 化**:
  中間層 $d_V$ を他層に注入する旧実装から、各層 $l$ の hidden states $H_l$ より求めた局所情動ベクトル $d_V^{(l)} = \beta_V^{(l)} / \|\beta_V^{(l)}\|$ と局所スケール $\operatorname{std}(H_l d_V^{(l)})$ を用いた注入へ改修。他層不適合による効果減衰という交絡を排除。
- **H4 (Temporal Emergence) の Arousal 拡張**:
  Valence ($d_V$) のみならず Arousal ($d_A$) についても生成ステージ（`candidate_start`, `pre_V`, `pre_A`, `response_end` 等）ごとの効果を独立追跡し、`passed_valence` と `passed_arousal` を両方判定・記録。

## 3. 検証手順
1. `tests/test_production_entrypoints.py` の実行（実体存在とファンアウトの確認）。
2. プロジェクト全体の回帰テスト（`pytest -q`）の実行。
3. `v3/primary/run_confirmatory_replication.py` の構文・動作確認。
