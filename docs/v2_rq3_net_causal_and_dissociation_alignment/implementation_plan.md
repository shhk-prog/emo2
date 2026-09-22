# V2 RQ3 設計不整合 修正実装計画（最終確定版）

## 設計方針（不変）

$$
C_{\mathrm{raw}} \text{ を保存} \rightarrow C_{\mathrm{net,rand}} \text{ を Primary} \rightarrow C_{\mathrm{net,\perp}} \text{ を robustness}
$$
$$
\text{LMM は signed } C_{\mathrm{net,rand}},\quad \text{localization は positive component}
$$
$$
\text{meaningful decodability peak } (R^2 > 0) \quad\text{vs.}\quad \text{control-adjusted positive causal peak } (C_{\mathrm{net,rand}} > 0)
$$
$$
c_v \equiv c_{v,\mathrm{raw}} \text{ の意味は永久に変えない}
$$

---

## 修正 1: RQ3 Net Causal Effect を Primary に昇格

### 1-A. 全メトリクスを明示的キーで保持（c_v 上書き禁止）

```python
PRIMARY_CAUSAL_METRIC_V = "c_v_net_rand"
PRIMARY_CAUSAL_METRIC_A = "c_a_net_rand"

causal_entry = {
    "c_v_raw":      res["c_v"],          # Secondary
    "c_a_raw":      res["c_a"],
    "c_v_rand":     res["c_v_rand"],     # Control profile
    "c_a_rand":     res["c_a_rand"],
    "c_v_perp":     res["c_v_perp"],     # Control profile
    "c_a_perp":     res["c_a_perp"],
    "c_v_net_rand": res["c_v_net_rand"], # Primary
    "c_a_net_rand": res["c_a_net_rand"],
    "c_v_net_perp": res["c_v_net_perp"], # Secondary robustness
    "c_a_net_perp": res["c_a_net_perp"],
    "c_v_zero":     res["c_v_zero"],
    "c_a_zero":     res["c_a_zero"],
    "c_v": res["c_v"],  # Legacy exact alias: ALWAYS raw, NEVER net
    "c_a": res["c_a"],
}
```

---

### 1-B. 負値を考慮した peak / center-of-mass の定義（Causal & Decodability 双対ガード）

$$C_{\mathrm{net}}^+(l) = \max(C_{\mathrm{net}}(l),\, 0), \quad D^+(l) = \max(D(l),\, 0)$$

| 指標 | プロファイル | 全層 $\le 0$ の場合 | 理由 |
|---|---|---|---|
| $d_C^*$ | $C_{\mathrm{net}}^+$ | `NaN` | random control を上回る positive causal peak が存在しない |
| $\bar{d}_C$ | $C_{\mathrm{net}}^+$ | `NaN` | 正の因果効果質量が存在しない ($\sum C^+ = 0$) |
| $d_D^*$ | $D^+$ | `NaN` | 「全層失敗の中で最も悪くない層」の誤採択を防止 |
| $\bar{d}_D$ | $D^+$ | `NaN` | 正のデコード質量が存在しない ($\sum D^+ = 0$) |
| $\Delta d^* = d_C^* - d_D^*$ | 両者有限 | `NaN` | 有意な decodability peak と positive causal peak が揃った時のみ定義 |
| $\Delta \bar{d} = \bar{d}_C - \bar{d}_D$ | 両者有限 | `NaN` | 両者の質量分布が存在する時のみ定義 |
| LMM / 統計検定 | signed $C_{\mathrm{net,rand}}$ | そのまま | 統計モデルは方向性・符号を保持したまま推定 |

---

### 1-C. `run_causal_patching_for_model()` 内部の変更

#### 新規配列（全部追加）

```python
layer_shifts_v_rand     = [[] for _ in range(actual_layers)]
layer_shifts_a_rand     = [[] for _ in range(actual_layers)]
layer_shifts_v_perp     = [[] for _ in range(actual_layers)]
layer_shifts_a_perp     = [[] for _ in range(actual_layers)]
layer_shifts_v_net_rand = [[] for _ in range(actual_layers)]
layer_shifts_a_net_rand = [[] for _ in range(actual_layers)]
layer_shifts_v_net_perp = [[] for _ in range(actual_layers)]
layer_shifts_a_net_perp = [[] for _ in range(actual_layers)]
```

#### サンプルループ追記

```python
layer_shifts_v[l].append(cv)
layer_shifts_a[l].append(ca)
layer_shifts_v_rand[l].append(cv_rand)
layer_shifts_a_rand[l].append(ca_rand)
layer_shifts_v_perp[l].append(cv_perp)
layer_shifts_a_perp[l].append(ca_perp)
layer_shifts_v_net_rand[l].append(cv - cv_rand)
layer_shifts_a_net_rand[l].append(ca - ca_rand)
layer_shifts_v_net_perp[l].append(cv - cv_perp)
layer_shifts_a_net_perp[l].append(ca - ca_perp)
```

#### dry-run mock も同じキーを全部返す

```python
c_v_rand_mean = [...]   # mock: random control profile
c_v_perp_mean = [...]   # mock: perp control profile
# 戻り値
"c_v_rand":     c_v_rand_mean,
"c_a_rand":     c_a_rand_mean,
"c_v_perp":     c_v_perp_mean,
"c_a_perp":     c_a_perp_mean,
"c_v_net_rand": [v - r for v, r in zip(c_v_mean, c_v_rand_mean)],
"c_a_net_rand": [a - r for a, r in zip(c_a_mean, c_a_rand_mean)],
"c_v_net_perp": [v - p for v, p in zip(c_v_mean, c_v_perp_mean)],
"c_a_net_perp": [a - p for a, p in zip(c_a_mean, c_a_perp_mean)],
```

---

### 1-D. Checkpoint Schema Version 2（layer + condition 両方に適用）

**Schema 定義**:
```python
CAUSAL_CKPT_SCHEMA_VERSION = 2
```

**Layer checkpoint 保存時**:
```python
_ckpt_payload = {
    "schema_version": CAUSAL_CKPT_SCHEMA_VERSION,
    "causal_primary_metric": "net_rand",
    "layer_shifts_v_net_rand": ...,
    "layer_shifts_v_perp": ...,
    ...
}
```

**Layer checkpoint 読み込み時（自動 invalidate）**:
```python
_schema = _ckpt.get("schema_version", 1)
_has_net = "layer_shifts_v_net_rand" in _ckpt
_has_perp = "layer_shifts_v_perp" in _ckpt
if _schema < CAUSAL_CKPT_SCHEMA_VERSION or not _has_net or not _has_perp:
    logger.warning("[layer-ckpt] Old schema → invalidating.")
    completed_layers = set(); pair_records = []
```

**Condition checkpoint 保存時（`v2_causal_cond_*.json`）**:
```python
cond_payload = {
    "schema_version": CAUSAL_CKPT_SCHEMA_VERSION,
    "causal_primary_metric": "net_rand",
    "cond_key": cond_key,
    "fam_id": fam_id,
    "causal_entry": causal_entry,  # c_v_net_rand 等を含む
    "aliases": aliases,
    "pair_level": cond_pair_records,
}
```

**Condition checkpoint 読み込み時**:
```python
_schema = saved.get("schema_version", 1)
_entry = saved.get("causal_entry", {})
_has_net_v = "c_v_net_rand" in _entry
_has_net_a = "c_a_net_rand" in _entry
if _schema < CAUSAL_CKPT_SCHEMA_VERSION or not _has_net_v or not _has_net_a:
    logger.warning(f"[cond-ckpt] Old schema for {cond_key} → recomputing.")
    # continue ではなく通常実行へ落ちる
else:
    # キャッシュヒット
    ...
    continue
```

---

### 1-E. pair-level CSV への列追加

```python
row = {
    "c_v":          prec["c_v"],
    "c_a":          prec["c_a"],
    "c_v_rand":     prec.get("c_v_rand",     float("nan")),
    "c_a_rand":     prec.get("c_a_rand",     float("nan")),
    "c_v_perp":     prec.get("c_v_perp",     float("nan")),
    "c_a_perp":     prec.get("c_a_perp",     float("nan")),
    "c_v_net_rand": prec.get("c_v_net_rand", float("nan")),  # Primary
    "c_a_net_rand": prec.get("c_a_net_rand", float("nan")),
    "c_v_net_perp": prec.get("c_v_net_perp", float("nan")),
    "c_a_net_perp": prec.get("c_a_net_perp", float("nan")),
    "c_v_zero":     prec.get("c_v_zero",     0.0),
    "c_a_zero":     prec.get("c_a_zero",     0.0),
    ...
}
```

**Combined pair-level CSV は per-family CSV の再構築で生成**（append 禁止）:

```python
# ファミリー完了後に per-family CSV を保存
fam_pair_df.to_csv(pair_dir / f"v2_causal_pair_level_{fam_id}.csv", index=False)

# 全ファミリーループ後に再構築
all_fam_csvs = sorted(pair_dir.glob("v2_causal_pair_level_*.csv"))
df_combined = pd.concat([pd.read_csv(p) for p in all_fam_csvs], ignore_index=True)
df_combined.to_csv(derived_dir / "v2_causal_pair_level.csv", index=False)
```

---

### 1-F. LMM formula の更新

```python
formula_v = f"{PRIMARY_CAUSAL_METRIC_V} ~ C(alignment) * C(task) * relative_depth"
formula_a = f"{PRIMARY_CAUSAL_METRIC_A} ~ C(alignment) * C(task) * relative_depth"
```

---

### 1-G. RQ3 dissociation で axis 別 Primary metric を使う

```python
for axis in ["valence", "arousal"]:
    primary_metric = (
        PRIMARY_CAUSAL_METRIC_V if axis == "valence"
        else PRIMARY_CAUSAL_METRIC_A
    )

    for cond_name, d_canonical, d_legacy, c_key in conditions_map:
        d_prof = _get_geometry_profile(axis_sharing, d_canonical, d_legacy)
        if d_prof is not None and c_key in fam_causal:
            net_prof = fam_causal[c_key][primary_metric]  # ← axis 別
            d_metrics = compute_net_causal_dissociation_metrics(
                d_prof, net_prof, depths
            )
            dissoc_results[axis][cond_name] = d_metrics
```

---

### 1-H. RQ3 native geometry legacy fallback

```python
def _get_geometry_profile(axis_data, canonical_key, legacy_key=None):
    val = axis_data.get(canonical_key)
    if val is not None:
        return val
    if legacy_key is not None:
        return axis_data.get(legacy_key)
    return None

conditions_map = [
    ("base_reader",         "base_r2_reader",        None,             "base_reader"),
    ("base_self",           "base_r2_self",           None,             "base_self"),
    ("inst_matched_reader", "inst_matched_r2_reader", None,             "inst_matched_reader"),
    ("inst_matched_self",   "inst_matched_r2_self",   None,             "inst_matched_self"),
    ("inst_native_reader",  "inst_native_r2_reader",  "inst_r2_reader", "inst_native_reader"),
    ("inst_native_self",    "inst_native_r2_self",    "inst_r2_self",   "inst_native_self"),
]
```

---

## 修正 2: キー統一と Consumer 防御的 Fallback 確認

### Producer（`run_rq1_rq2_cross_decoding.py`）

```python
# Canonical（新）
"inst_native_r2_reader":    value,
"inst_native_r2_self":      value,
"inst_native_cross_r_to_s": value,
"inst_native_cross_s_to_r": value,
"inst_native_sharing":      value,
"delta_sharing_native":     value,
"delta_delta_cross_native": value,
# Legacy exact alias（旧 consumer 保護）
"inst_r2_reader":    value,
"inst_r2_self":      value,
"inst_cross_r_to_s": value,
"inst_cross_s_to_r": value,
"inst_sharing":      value,
"delta_sharing":     value,
"delta_delta_cross": value,
```

### Consumer 防御的確認マトリクス

既存 JSON（RQ1/RQ2 未再実行）を安全に読み込めるよう、全 Consumer で fallback が担保されていることを確認：

| Consumer | 参照箇所 | Canonical Key | Legacy Fallback Key | 確認状態 |
|---|---|---|---|---|
| `run_confirmatory_analysis.py` | H1 Secondary native shift | `inst_native_r2_reader` | `inst_r2_reader` | **本修正で追加** |
| `run_confirmatory_analysis.py` | H1 Secondary native shift | `inst_native_r2_self` | `inst_r2_self` | **本修正で追加** |
| `run_confirmatory_analysis.py` | H2 Secondary native sharing | `delta_sharing_native` | `delta_sharing` | 既存実装済 (`.get(...) or .get(...)`) |
| `run_rq3_causal_map.py` | RQ3 native reader profile | `inst_native_r2_reader` | `inst_r2_reader` | **本修正で追加** (`_get_geometry_profile`) |
| `run_rq3_causal_map.py` | RQ3 native self profile | `inst_native_r2_self` | `inst_r2_self` | **本修正で追加** (`_get_geometry_profile`) |

```python
# run_confirmatory_analysis.py の実装
inst_reader_native = axis_data.get(
    "inst_native_r2_reader",
    axis_data.get("inst_r2_reader", [])
)
inst_self_native = axis_data.get(
    "inst_native_r2_self",
    axis_data.get("inst_r2_self", [])
)
```

---

## 修正 3: Confirmatory H3 を `c_v_net_rand` / `c_a_net_rand` に変更

```python
PRIMARY_CAUSAL_COLUMNS = {
    "valence": "c_v_net_rand",
    "arousal": "c_a_net_rand",
}

for axis_name, col_name in PRIMARY_CAUSAL_COLUMNS.items():
    formula = (
        f"{col_name} ~ "
        "C(family) + C(alignment) * C(task) * relative_depth"
    )
```

### Confirmatory dry-run mock の更新

mock pair-level records に `c_v_net_rand`, `c_v_rand`, `c_v_perp` 等を追加。

---

## 新規関数の追加（`src/affective_empathy_eval/geometry.py`）

### 1. `compute_decodability_peak()` — NaN-safe & Positive-guard

```python
def compute_decodability_peak(profile, depths) -> float:
    vals   = np.asarray(profile, dtype=np.float64)
    depths = np.asarray(depths,  dtype=np.float64)
    valid  = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")
    vals_v   = vals[valid]
    depths_v = depths[valid]
    if np.max(vals_v) <= 0.0:
        return float("nan")
    return float(depths_v[np.argmax(vals_v)])
```

### 2. `compute_decodability_center_of_mass()` — NaN-safe & Positive-guard

```python
def compute_decodability_center_of_mass(profile, depths) -> float:
    vals   = np.asarray(profile, dtype=np.float64)
    depths = np.asarray(depths,  dtype=np.float64)
    valid  = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")
    vals_v   = vals[valid]
    depths_v = depths[valid]
    positive = np.maximum(vals_v, 0.0)
    total    = np.sum(positive)
    if total <= 0.0:
        return float("nan")
    return float(np.sum(depths_v * positive) / total)
```

### 3. `compute_causal_peak_from_net()` — NaN-safe & Positive-guard

```python
def compute_causal_peak_from_net(c_net, depths) -> float:
    vals   = np.asarray(c_net,   dtype=np.float64)
    depths = np.asarray(depths,  dtype=np.float64)
    valid  = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")
    vals_v   = vals[valid]
    depths_v = depths[valid]
    positive = np.maximum(vals_v, 0.0)
    if np.max(positive) <= 0.0:
        return float("nan")
    return float(depths_v[np.argmax(positive)])
```

### 4. `compute_causal_center_of_mass_from_net()` — NaN-safe & Positive-guard

```python
def compute_causal_center_of_mass_from_net(c_net, depths) -> float:
    vals   = np.asarray(c_net,  dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    valid  = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")
    vals_v   = vals[valid]
    depths_v = depths[valid]
    positive = np.maximum(vals_v, 0.0)
    total    = np.sum(positive)
    if total <= 0.0:
        return float("nan")
    return float(np.sum(depths_v * positive) / total)
```

### 5. `compute_net_causal_dissociation_metrics()` — 完全双対ガード

```python
def compute_net_causal_dissociation_metrics(
    decodability_profile, net_causal_profile, relative_depths
) -> dict:
    d_d    = compute_decodability_peak(decodability_profile, relative_depths)
    d_c    = compute_causal_peak_from_net(net_causal_profile, relative_depths)
    bar_dd = compute_decodability_center_of_mass(decodability_profile, relative_depths)
    bar_dc = compute_causal_center_of_mass_from_net(net_causal_profile, relative_depths)

    fin = lambda x: np.isfinite(x) if x is not None else False
    delta_d_star = float(d_c - d_d)       if fin(d_c) and fin(d_d)       else float("nan")
    delta_bar_d  = float(bar_dc - bar_dd) if fin(bar_dc) and fin(bar_dd) else float("nan")

    return {
        "d_d_star":                      d_d,
        "d_c_star":                      d_c,
        "delta_d_star":                  delta_d_star,
        "bar_d_d":                       bar_dd,
        "bar_d_c":                       bar_dc,
        "delta_bar_d":                   delta_bar_d,
        "no_positive_net_causal_peak":   not fin(d_c),
        "no_positive_decodability_peak": not fin(d_d),
    }
```

---

## 影響ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `src/affective_empathy_eval/geometry.py` | MODIFY | 5関数追加・更新（decodability / causal 双対 positive guard） |
| `v2/primary/run_rq3_causal_map.py` | MODIFY | net/perp 配列追加・schema v2 両 ckpt・conditions_map fallback・axis 別 primary metric・LMM 列名・CSV 再構築 |
| `v2/primary/run_rq1_rq2_cross_decoding.py` | MODIFY | canonical + legacy exact alias |
| `v2/primary/run_confirmatory_analysis.py` | MODIFY | H3 列名・native fallback・dry-run mock 更新 |
| `tests/test_rq3_net_causal_effect.py` | NEW | 14件の unit test |

---

## 削除対象（RQ3 再実行前）

```bash
# RQ3 チェックポイント（両種類）
rm -f v2/results/raw/v2_causal_cond_*.json
rm -f v2/results/raw/v2_causal_layer_ckpt_*.json
# RQ3 manifest（キャッシュ判定を確実に無効化）
rm -f v2/results/raw/manifest_causal_map_*.json
# RQ3 成果物
rm -f v2/results/raw/v2_causal_map_*.json
rm -f v2/results/raw/v2_rq3_causal_relocation_*.json
rm -f v2/results/derived/v2_causal_pair_level.csv
rm -f v2/results/derived/pair_level/v2_causal_pair_level_*.csv
rm -f v2/results/derived/v2_causal_dissociation_summary.json
# Confirmatory
rm -f v2/results/derived/v2_lmm_confirmatory.json
# RQ1/RQ2・RQ4 の成果物は削除しない
```

---

## 推奨実行順序

```bash
# 1. テスト（14件）
python -m pytest tests/test_rq3_net_causal_effect.py -q --tb=short

# 2. dry-run smoke test
python v2/primary/run_rq3_causal_map.py --family olmo --dry-run --force
python v2/primary/run_confirmatory_analysis.py --dry-run

# 3. 削除（上記参照）

# 4–7. RQ3 全 4 family（--force 必須）
python v2/primary/run_rq3_causal_map.py --family qwen  --device cuda:0 --force
python v2/primary/run_rq3_causal_map.py --family llama --device cuda:0 --force
python v2/primary/run_rq3_causal_map.py --family gemma --device cuda:0 --force
python v2/primary/run_rq3_causal_map.py --family olmo  --device cuda:0 --force

# 8. 統合確認
python -c "
import pandas as pd
df = pd.read_csv('v2/results/derived/v2_causal_pair_level.csv')
print(df['family'].value_counts())
required = ['c_v_net_rand','c_a_net_rand','c_v_perp','c_a_perp','c_v_raw','c_a_raw']
missing = [c for c in required if c not in df.columns]
print('Missing columns:', missing)
"

# 9. Confirmatory 再計算
python v2/primary/run_confirmatory_analysis.py
```

---

## 検証計画（14件の unit test）

### `tests/test_rq3_net_causal_effect.py`

| # | テスト内容 |
|---|---|
| 1 | `C_affect=0.8, C_rand=0.3` → `C_net=0.5`（数値正確性） |
| 2 | raw peak ≠ net peak（`C_raw=[0.9,1.0,0.8]`, `C_rand=[0.8,0.95,0.1]` → net peak は layer 3） |
| 3 | Causal 全層 ≤ 0 → $d_C^* = \mathrm{NaN}$（`C_net=[-0.8,-0.4,-0.1]`） |
| 4 | Causal 全層 ≤ 0 → $\bar{d}_C = \mathrm{NaN}$ |
| 5 | Causal 全層 ≤ 0 → `delta_d_star = NaN`, `delta_bar_d = NaN` |
| 6 | dry-run mock が `c_v_net_rand`, `c_v_perp` 等すべてのキーを含む |
| 7 | layer ckpt schema_version=1 が自動 invalidate される |
| 8 | cond ckpt schema_version=1（または `c_v_net_rand` 欠落）が自動 invalidate される |
| 9 | aggregation consistency: sample-level mean と layer profile 一致 |
| 10 | Confirmatory H3 が `c_v_net_rand`（net 側）を使うことを検証（raw と net で depth pattern が異なる mock data） |
| 11 | Combined CSV が per-family CSV から**再構築**されること（qwen → qwen+llama → 4 family の順で family 数が正しいことを確認） |
| 12 | **介入等価性テスト**: `\|\|Δh_affect\|\| ≈ \|\|Δh_rand\|\| ≈ \|\|Δh_perp\|\|`（同一 `alpha` と `hidden_std` で注入された場合に成立することを確認） |
| 13 | **Decodability 全層 ≤ 0 ガード**: `D = [-0.8, -0.4, -0.1]` → $d_D^* = \mathrm{NaN}, \bar{d}_D = \mathrm{NaN}$ |
| 14 | **Decodability 全層 ≤ 0 による解離ガード**: `D 全層 ≤ 0` → $\Delta d^* = \mathrm{NaN}, \Delta \bar{d} = \mathrm{NaN}$, `no_positive_decodability_peak = True` |

### Manual Verification

```bash
# native dissociation エントリの存在と no_positive_*_peak フラグの整合確認
python -c "
import json, math
d = json.load(open('v2/results/derived/v2_causal_dissociation_summary.json'))
for fam, fd in d.get('families', {}).items():
    entry = fd.get('valence', {}).get('inst_native_reader')
    assert entry is not None, f'{fam}: inst_native_reader missing (fallback failed)'
    if entry.get('no_positive_net_causal_peak', False):
        assert not math.isfinite(entry['d_c_star']), f'{fam}: expected NaN causal peak'
        print(f'{fam}: native reader → no positive causal peak (NaN, valid negative result)')
    else:
        assert math.isfinite(entry['d_c_star']), f'{fam}: expected finite causal peak'
        print(f'{fam}: native reader → d_c_star={entry[\"d_c_star\"]:.3f}')
    
    if entry.get('no_positive_decodability_peak', False):
        assert not math.isfinite(entry['d_d_star']), f'{fam}: expected NaN decodability peak'
        print(f'{fam}: native reader → no positive decodability peak (NaN)')
    else:
        assert math.isfinite(entry['d_d_star']), f'{fam}: expected finite decodability peak'
        print(f'{fam}: native reader → d_d_star={entry[\"d_d_star\"]:.3f}')
"
```
