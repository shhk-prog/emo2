# Implementation Plan - Directional Decodability--Causal Dissociation (H1) Replication Criterion

## 背景と目的
V3 Stageでは、Qwen Discoveryで得られた知見（Affective DecodabilityとCausal Displacementの空間的・時間的配置）を、独立した3つのモデルファミリー（Llama 3.2、Gemma 3、OLMo 2）において事前登録基準（Pre-registered Replication Criteria）に基づき追試（Confirmatory Replication）する設計となっている。

現在の定義：
$$
\Delta d^* = d_C^* - d_D^*, \qquad \Delta \bar{d} = \bar{d}_C - \bar{d}_D
$$
Qwen Discoveryの実測値（`v3_spatiotemporal_summary.json`）：
- **Valence**:
  - $\Delta d^* = -0.1111 < 0$（Peak: 因果がデコーディングより浅い層に位置）
  - $\Delta \bar{d} = -0.1277 < 0$（COM: 因果重心が浅い層に位置）
- **Arousal**:
  - $\Delta d^* = -0.0741 < 0$（Peak: 因果がデコーディングより浅い層に位置）
  - $\Delta \bar{d} = +0.0387 > 0$（COM: 因果重心が深い層に位置）

しかし、従来のH1判定コードおよび論文の記述では、一律に $CI_{\text{low}}(\Delta) > 0$ を要求していた。これはQwenで発見されたパターンのreplication criterionになっておらず、特にArousalのCOMのように符号が反転しているケースを評価できない。

本改修では、H1を**「Qwen Discoveryで観測された方向のspatial dissociationが他familyでも再現するか（Directional Decodability--Causal Dissociation）」**として再定式化する。

## 修正計画

### 1. 符号の明示的 Freeze
- `v3/results/derived/frozen_confirmatory_sites.json` に `h1_replication_direction` を追加：
  ```json
  "h1_replication_direction": {
    "valence": {
      "peak": -1,
      "center_of_mass": -1
    },
    "arousal": {
      "peak": -1,
      "center_of_mass": 1
    }
  }
  ```
- `v3/primary/run_rq3_path_mediation.py` にて、Qwen Discovery summary から符号を抽出し、`frozen_confirmatory_sites.json` に自動出力するロジックを担保。

### 2. `v3/primary/run_confirmatory_replication.py` の修正
- `frozen_confirmatory_sites.json` から `h1_replication_direction` を取得（ない場合は Qwen Discovery summary またはデフォルトから取得）。
- 各モデルの H1 評価において：
  - 各軸（valence, arousal）について、Qwen sign を取得：
    - $s_{\text{peak}} = \text{sign}_{\text{peak}}$
    - $s_{\text{com}} = \text{sign}_{\text{com}}$
  - 生の指標（raw）を保持：
    - `delta_d_peak_raw` / `delta_d_peak`: $\Delta d^*$
    - `delta_d_center_raw` / `delta_d_center`: $\Delta \bar{d}$
    - `delta_d_peak_ci`: $[l_{\text{peak}}, u_{\text{peak}}]$
    - `delta_d_center_ci`: $[l_{\text{com}}, u_{\text{com}}]$
  - 整列指標（aligned）を算出・保持：
    - `qwen_sign_peak`: $s_{\text{peak}}$
    - `qwen_sign_center`: $s_{\text{com}}$
    - `delta_d_peak_aligned`: $s_{\text{peak}} \cdot \Delta d^*$
    - `delta_d_center_aligned`: $s_{\text{com}} \cdot \Delta \bar{d}$
    - `aligned_peak_ci`: bootstrap標本を $s_{\text{peak}}$ 倍した95% CI（または $[s \cdot u, s \cdot l]$ if $s < 0$ else $[l, u]$）
    - `aligned_center_ci`: bootstrap標本を $s_{\text{com}}$ 倍した95% CI
    - `peak_replication_pass`: `aligned_peak_ci[0] > 0`
    - `center_replication_pass`: `aligned_center_ci[0] > 0`
    - `passed`: `peak_replication_pass and center_replication_pass`
- `summary["primary_effect_estimates"]["H1_peak_dissociation"]` および各family結果にrawとalignedを併記。

### 3. 集計スクリプト・サマリーテーブルの更新
- `v3/scripts/build_paper_summary.py`:
  - `table_v3_4_confirmatory.csv` の H1 行において、aligned CI または Qwen sign を考慮した判定を行い、pass列に反映。
  - threshold表示を `CI_low(aligned) > 0` または対応する方向を明記。
- `scripts/summarize_v3_causal_utilization.py`:
  - LaTeX テーブルの注記やヘッダーを Directional Replication に整合。
- 再集計の実行：
  - 保存された raw bootstrap / 統計値に基づき、全familyのH1判定を再集計。
  - `v3_cross_model_replication_summary.json`、`table_v3_4_confirmatory.csv`、`table_v3_confirmatory_matrix.csv`、`iclr2027/tables/` 以下のTeXファイルを更新。

### 4. 論文 `iclr2027/iclr2027_conference2.tex` の更新
- **Methods**: `\subsection{Replication H1：Directional Decodability--Causal Dissociation}` の数式と定義を、Qwen-sign-aligned criterionに整合。
- **Results & Discussion**:
  - 再集計結果に基づき記述を更新：
    - Valenceにおいては、Llama 3.2 および OLMo 2 で Qwen と同一方向（因果ピーク・重心がデコーディングより浅い層に位置する）の空間的解離が有意に再現された。
    - しかし、Arousalにおいては信頼区間が0を跨ぐか重心の方向が一致せず、再現されなかった。
    - したがって、「Qwen Discoveryで観測されたdirectional spatial dissociationは一部のreplication families（Valence）で再現されたが、3 families全体および双方の軸で一貫して一般化するパターンではなかった」という精密な学術的報告に更新。
    - また、Gate failureによるexploratory statusであることも堅持。

### 5. テストと検証
- `tests/test_confirmatory_pipeline.py` に directional H1 alignment のテストを追加。
- pytest, ruff を実行。
