# V2 コードベース詳細監査レポート: 実装状況と未実装・要修正箇所の特定

## 1. 監査サマリー

提示された懸念事項および批判的検討項目（12.1〜12.8, 13）について、`/mnt/nas/home/hiromi/src/emo` 内の実際のソースコードを1行ずつ精査しました。

その結果、**「12.1（Model Adapterの不在）」は現行リポジトリで既に実装・解決済み**である一方、**「12.2（RQ4実パッチング未実装）」および「12.3（RQ3/RQ4の先頭81ロジット切り出しバグ）」は現行コードにそのまま残存している極めて重大な未実装・欠陥である**ことが確認されました。

| 項目 | 指摘内容 | 現行コードでの判定 | 影響度 | 現状のコード箇所・詳細 |
|:---|:---|:---:|:---:|:---|
| **12.1** | Model Adapter が存在しない | **解決済み** | なし | `src/affective_empathy_eval/models/` に `adapters.py`, `hooks.py`, `registry.py` が存在し実装済み。 |
| **12.2** | V2-RQ4の実モデルパッチングが未実装 | **未実装** | **重大** | `v2/scripts/run_v2_recovery_patching.py` (L136-141) でパッチング処理がコメントアウトされており、無介入のInstruct出力をそのまま追加。全層回復率 0% となる。 |
| **12.3** | 81状態評価がSequence-Likelihoodになっていない | **未実装 / バグ** | **極めて重大** | `run_v2_2x2_causal_map.py` (L125, L142) および `run_v2_recovery_patching.py` (L116, L123) で、JSON候補列ではなく `logits[:, -1, :81]`（語彙の先頭81トークン）を切り出して期待値・EMDを算出している。 |
| **12.4** | データ分割が `pair_id` 単位ではない | **要修正** | 中 | `run_v2_2x2_cross_decoding.py` (L55-63) の `split_dataset()` が行単位のランダムシャッフル。同一文脈の刺激がtrain/testに跨る可能性がある。 |
| **12.5** | `matched_plain` フォーマットが未使用 | **未実装** | 中 | `configs/v2_experiments.yaml` に定義があるが、`run_v2_2x2_cross_decoding.py` (L274-277) では Base=`plain`, Instruct=`chat` に固定。 |
| **12.6** | RQ1/RQ2のターゲットがValenceのみ | **未実装** | 中 | `run_v2_2x2_cross_decoding.py` (L141-142) で `reader_V` のみ取り出しており、`reader_A`（Arousal）は計算されていない。 |
| **12.7** | 「差の差」の計算式が計画書と不一致 | **独自定義** | 小 | `dd = (inst_cross - base_cross) - (inst_within - base_within)` となっており、一般的な2×2 DiDではない。 |
| **12.8** | Bootstrap統計が未接続 | **未接続** | 中 | 設定ファイルに `n_boot: 1000` の記述はあるが、点推定値のリスト保存のみで信頼区間が算出されていない。 |
| **13** | 旧スクリプトのモック・疑似実装 | **事実** | 注意 | `run_strict_cross_decoding.py` は固定値辞書を保存する完全なモック。`run_introspective_accessibility_test.py` も活性化パッチングを行わない疑似スクリプト。 |

---

## 2. 各項目の詳細検証結果

### 2.1 【解決済み】12.1: Model Adapter の実装状況
- **ファイル**: `src/affective_empathy_eval/models/`
  - `adapters.py` (4,326 bytes): `BaseModelAdapter`, `QwenAdapter`, `LlamaAdapter`, `GemmaAdapter`, `MistralAdapter` が完備。
  - `hooks.py` (11,481 bytes): `ActivationHookManager`, `HookPoint`（`POST_MLP_RESID`, `ATTN_OUTPUT` 等）が実装済み。
  - `registry.py` (3,431 bytes): `ModelRegistry`, `FamilyConfig`, `ModelSpec` が実装済み。
- **結論**: `from affective_empathy_eval.models.adapters import get_model_adapter` 等の import は正常に通る状態になっており、現行コードでは解決しています。

---

### 2.2 【未実装・重大】12.2: V2-RQ4 の実パッチング未実装
- **ファイル**: `v2/scripts/run_v2_recovery_patching.py` (L136-151)
```python
    for l in range(num_layers):
        patched_log_probs_list = []
        for i, row in df.iterrows():
            # Base の活性化を抽出して Instruct の層 l に注入
            # ... (実モデルパッチング処理)
            patched_log_probs_list.append(inst_log_probs_list[i])

        patched_metrics = compute_distribution_metrics(
            np.mean(patched_log_probs_list, axis=0),
            np.mean(base_log_probs_list, axis=0),
        )
        emd = patched_metrics["emd_va"]
        ratio = (initial_emd - emd) / (initial_emd + 1e-12)
        recovery_emd.append(emd)
        recovery_ratios.append(ratio)
```
- **検証**:
  Base の活性化を抽出し Instruct へ介入するフック処理が一切書かれておらず、無介入の `inst_log_probs_list[i]` をそのまま入れています。
  したがって、`emd == initial_emd` となり、回復率 `ratio` は全層で `0.0` になります。
- **結論**: **未実装**。このスクリプトで出力された `v2_distribution_recovery_*.json` は実験結果として全く意味をなしません。

---

### 2.3 【未実装・極めて重大】12.3: 先頭81ロジット切り出しバグ（Sequence-Likelihood未実装）
- **ファイル 1**: `v2/scripts/run_v2_2x2_causal_map.py` (L122-143)
```python
    out_clean = model(input_ids)
    logits_clean = out_clean.logits[:, -1, :].float().cpu().numpy()[0]
    # 81候補のトークン確率から期待値を概算（簡易版）
    ev_clean, ea_clean = compute_expected_va(logits_clean[:81], candidates)
```
- **ファイル 2**: `v2/scripts/run_v2_recovery_patching.py` (L115-116, L122-123)
```python
    out_b = model_base(**enc_b)
    probs_b = torch.softmax(out_b.logits[0, -1, :81].float(), dim=-1).cpu().numpy()
```
- **検証**:
  `logits[0, -1, :81]` は、語彙空間（vocab）のインデックス 0〜80 番目のトークンのロジットです。これは JSON 文字列 `{"valence": 1, "arousal": 1}` 〜 `{"valence": 9, "arousal": 9}` の完全なシーケンス対数尤度 $\sum_t \log P(token_t \mid context, token_{<t})$ では**絶対にありません**。
  本来であれば `v2/src/likelihood.py` や `v3/src/batch_likelihood.py` に実装されている `compute_likelihoods_for_candidates()` を呼び出すべきところ、手抜き実装のまま放置されています。
- **結論**: **未実装かつ重大な計算誤り**。これによって得られた期待値 $E[V], E[A]$ や因果マップ、分布間距離（$EMD_{VA}$）は、トークンID 0〜80（句読点や特殊記号、無関係な単語等）の生起確率に基づいた無意味な数値になってしまっています。

---

### 2.4 【要修正】12.4: データ分割が pair_id 単位ではない
- **ファイル**: `v2/scripts/run_v2_2x2_cross_decoding.py` (L55-63)
```python
def split_dataset(df: pd.DataFrame, train_ratio: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return df.iloc[train_indices].copy().reset_index(drop=True), df.iloc[test_indices].copy().reset_index(drop=True)
```
- **検証**: 行インデックスの単純シャッフルであり、`pair_id` 列によるグループ化が行われていません。同一のヴィネットや最小対に属する文が train と test の両方に含まれる交絡（Data Leakage）の恐れがあります。

---

### 2.5 【未実装】12.5: `matched_plain` フォーマットの未評価
- **ファイル**: `v2/scripts/run_v2_2x2_cross_decoding.py` (L274-277)
```python
        conditions = [
            ("base", "reader", TaskType.READER, "plain"),
            ("base", "self", TaskType.SELF, "plain"),
            ("inst", "reader", TaskType.READER, "chat"),
            ("inst", "self", TaskType.SELF, "chat"),
        ]
```
- **検証**: Instruct モデルのプロンプト形式が `"chat"` に固定されており、`matched_plain`（Chat Templateを使わずにplainテキストで揃える条件）を走らせる分岐・ループが実装されていません。

---

### 2.6 【未実装】12.6: Arousal（覚醒度）のデコーディングが未実装
- **ファイル**: `v2/scripts/run_v2_2x2_cross_decoding.py` (L141-142)
```python
    y_train_v = train_df["reader_V"].values
    y_test_v = test_df["reader_V"].values
```
- **検証**: `reader_V`（Valence）しか取り出しておらず、`reader_A`（Arousal）に対するプローブ学習やクロスデコーディングは一切実行されていません。

---

### 2.7 【事実・注意】13: 旧スクリプトのモック・疑似実装
- **ファイル 1**: `v2/scripts/run_strict_cross_decoding.py` (L11-18)
  ```python
  results = {
      "Valence": 0.58,
      "Arousal": 0.42,
      "Token Count": 0.05,
      "Surface VAD": 0.12,
      "Narrative Richness": 0.08
  }
  ```
  実行するとこのハードコードされた辞書をJSON保存するだけの完全なモックです。
- **ファイル 2**: `v2/scripts/run_introspective_accessibility_test.py` (L22-23)
  ```python
  # 本来はここで Activation Patching を行い、状態を強制的に変化させる。
  # 今回は簡略化のため、Patching前後の生成テキストと尤度を比較する疑似スクリプトとする。
  ```
  活性化パッチングを行わずにプロンプトを投げるだけの疑似スクリプトです。

---

## 3. 結論と今すぐ必要なアクション

ユーザーのご指摘通り、**現行のV2コードには「実験結果として論文に書いてはならない未実装・誤り」が明確に存在します。**

特に今すぐ論文の実験として使う前に修正が必須なのは以下の2点です：
1. **RQ3 および RQ4 における Sequence-Likelihood の再実装**
   - `logits[:, -1, :81]` を廃止し、`v2/src/likelihood.py` の `compute_likelihoods_for_candidates()` または `v3/src/batch_likelihood.py` を呼び出して正準な81状態対数尤度を計算するように修正する。
2. **RQ4（`run_v2_recovery_patching.py`）の実モデルパッチングの実装**
   - `ActivationHookManager` を用いて、Baseモデルから抽出した各層の活性化をInstructモデルの対応層に注入するパッチングループを実装する。
