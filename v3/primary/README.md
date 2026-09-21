# V3 Primary
## 論文対応: Section 6. From Decodability to Causal Leverage: Sufficiency, Specificity, and Spatiotemporal Dynamics

V3 の正式実行面。設計・指標・解釈の本文は親の [`v3/README.md`](../README.md) を正本とする。

- **科学的問い (RQ)**: *Where does affect-relevant information exert measurable causal leverage over self-report in instruct models?*（指示追従モデルにおいて、情動関連の内部情報は計算過程のどこで自己報告に対する測定可能な因果的影響力（causal leverage）を行使するのか？）
- **介入オペレータ**: 
  - **十分性（Sufficiency）の検証**: 中立文への方向加算注入（Additive Direction Injection）$h' = h + \alpha \cdot \sigma_h \cdot \hat{d}$（$\alpha \in [-2.0, 2.0]$）
  - **内生的関連性（Endogenous Relevance）の検証**: 情動文に対する中心化2D直交部分空間除去（Centered Orthogonal Subspace Removal）$h' = h - Q Q^T (h - \mu_{\text{neu}})$
  - 両オペレータは対象（中立文 vs 情動文）および検証目的が明確に分離されており、同一操作として扱わない。
- **Primary Metric**:
  - **RQ1 State Induction & Dose-Response**: 中立ベースラインに対する用量反応スロープ $\beta_{\text{dose}}$、直交方向 $\hat{d}_{\perp}$ およびランダム方向 $\hat{d}_{\text{rand}}$ での特異性（Null effect 検証）。
  - **RQ2 Spatiotemporal Maps**: 層 $l \times$ 生成段階 $t$ における時空間因果影響スロープ $\beta_{l,t}$、ピーク層 $l^*$、ピークトークン段階 $t^*$。因果的影響力が特定の層、および teacher-forced candidate sequence 上の特定の計算段階に集中（*concentrated at particular layers and stages along the teacher-forced candidate sequence*）することを検証。
  - **RQ3 Path Mediation**: 情動部分空間の除去（Centered Subspace Removal）による自己報告変位の減衰量および減衰率（Mediated Attenuation / Attenuation Ratio。内生的関連性の検証）。
  - **Confirmatory Replication**: 独立したサンプル単位交差フィッティング（**Sample-Level Holdout Cross-Fitting / GroupKFold on `pair_id`**）による他ファミリーでの厳格な再現性検証（プローブ推定と介入評価のデータ重複リークを完全排除）。
- **統計単位**: Matched pair ($N=192$ clinical-neutral pairs) / Sample-level holdout test folds
- **統制条件**:
  - ゼロ介入ベースライン ($\alpha=0$)
  - 直交方向統制 ($\hat{d}_{\perp}$) およびランダム方向統制 ($\hat{d}_{\text{rand}}$)
  - 非情動的トピック統制 (Topic Control)
  - 刺激ペア単位の厳格なサンプル分割 (Sample-level holdout)
- **出力成果物**: `v3/results/derived/` (`v3_rq1_state_induction.csv`, `v3_gate_decision.json`, `v3_rq2_spatiotemporal_maps.csv`, `v3_rq3_path_mediation.csv`, `v3_confirmatory_replication.csv`)

中心の問い: **Where does affect-relevant information exert measurable causal leverage over self-report?**  
データ: AIPsy clinical–neutral 192 pair（`load_v3_matched_pair_table`）。EmoBank 3-way は使わない。  
候補空間: 81 VA。Teacher-forced joint sequence evaluation により自己回帰的な生成ドリフトを統制し、Stage 間では絶対値を直接比較せず、各 Stage 内の contrast と relative pattern を主たる推論対象とする。

## 実行順とゲート

統合 CLI / `run_production_v3.sh` は次の順である。

1. `run_rq1_state_induction.py` → 本番は `v3/results/derived/v3_gate_decision.json`、`--dry-run` は `v3/results/derived/dry_run/v3_gate_decision.json`
2. `decision == "GO"` のときだけ RQ2 → RQ3 → Confirmatory
3. `NO_GO` / 軸片方の GO は終了コード 2
4. `--force-after-no-go` のときだけ 2 を強制する
5. `--force` は各 RQ のキャッシュ再計算であり、ゲート継続ではない

単独で RQ2 を呼ぶとゲートは見ない。本番経路では見ないといけない。設定正本は `configs/v3_experiments.yaml`（`min_sufficiency_slope: 0.1`、`n_causal_samples: 15`、`confirmatory` frozen ブロック、`normalize_length: true`）。

## スクリプト

| ファイル | 問い | 実装上の固定点 |
|---|---|---|
| `run_rq1_state_induction.py` | 方向注入は特異的に自己報告を動かすか | pair Group split。Primary 方向は人間 reader またはモデル Reader Prediction。$d_V$ / $d_A$ 別 sweep。matched-neutral 必須。層は $d=0.5$ |
| `run_rq2_spatiotemporal_maps.py` | $D,\beta,\gamma,C$ のピークはどこか | joint sequence patch。YAML `response_start` を実行キー `candidate_start` に正規化。`n_map_samples` と `n_intervene_samples`（既定 15）を分離。`analysis_role: discovery`。符号付き $\beta$ と `abs_beta_*` |
| `run_rq3_path_mediation.py` | 部分空間遮断で変位は減衰するか | Discovery / Confirmation 50:50。NDE/NIE とは呼ばない |
| `run_confirmatory_replication.py` | 他 family でも同じか | Llama / Gemma 3 / OLMo 2。Sufficiency も V/A 別 sweep |

未知 family は `KeyError`。`--pilot` は RQ1 の 50 行。`--subsample` は RQ2/RQ3/Confirmatory の確認用。

## 実行

```bash
bash scripts/run_production_v3.sh cuda:0
# bash scripts/run_production_v3.sh cuda:0 --force-after-no-go

python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0

python v3/primary/run_rq1_state_induction.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v3/primary/run_rq1_state_induction.py --dry-run --family qwen
```

`v3/scripts/legacy/` は主解析に使わない。
