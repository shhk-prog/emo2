# Walkthrough: Olmo2Adapter hidden_size 属性欠落の修正

## 日時
2026-09-22

## 発端
`production_v2_20260921_191441.log` の RQ4 実行時に以下のエラーが発生：

```
AttributeError: 'Olmo2Adapter' object has no attribute 'hidden_size'
  File "v2/primary/run_rq4_recovery_patching.py", line 255
    assert adapter_base.hidden_size == adapter_inst.hidden_size
```

## 原因

`run_rq4_recovery_patching.py` の 255 行目で `adapter.hidden_size` を参照していたが、
`ModelAdapter` 基底クラスおよびサブクラス（`Olmo2Adapter` など）に `hidden_size` プロパティが実装されていなかった。

## 修正内容

### [MODIFY] `src/affective_empathy_eval/models/adapters.py`

`ModelAdapter` 基底クラスに `hidden_size` プロパティを追加：

- `model.config.hidden_size`（Transformers 標準属性）を最優先で参照
- `d_model`（BERT 系）、`n_embd`（GPT 系）へのフォールバックも実装
- `config` 非保持モデルには最初の層の LayerNorm weight shape から推論
- 基底クラスへの追加のため、`Olmo2Adapter`・`LlamaFamilyAdapter`・`GemmaAdapter` すべてに自動適用

## 検証

```
python -m pytest tests/test_model_registry_and_adapters.py -q
12 passed in 5.37s
```

既存テスト 12 件すべてパス。スキップ・エラーなし。

---

# Walkthrough: RQ3 cond_key 単位の逐次保存追加

## 日時
2026-09-22（同日追加）

## 背景
`base_plain_reader` → `inst_native_chat_self` など各条件が約3.8時間かかるにもかかわらず、
ファミリー全体が完了するまでファイル保存されない設計だった。
途中クラッシュ時に全条件をやり直す必要があった。

## 修正内容

### [MODIFY] `v2/primary/run_rq3_causal_map.py`

各 `cond_key` 完了直後に `v2/results/raw/v2_causal_cond_{fam_id}_{cond_key}.json` へ逐次保存。
再実行時はファイルが存在する cond_key をスキップ（モデルのロードも不要）。

保存される cond_key（olmo の例）:
- `v2_causal_cond_olmo_base_plain_reader.json`
- `v2_causal_cond_olmo_base_plain_self.json`
- `v2_causal_cond_olmo_inst_matched_plain_reader.json`
- `v2_causal_cond_olmo_inst_matched_plain_self.json`
- `v2_causal_cond_olmo_inst_native_chat_reader.json`
- `v2_causal_cond_olmo_inst_native_chat_self.json`

スキップの優先順位:
1. モデルグループ内の全 cond_key がキャッシュ済み → モデルロード自体をスキップ
2. 一部の cond_key がキャッシュ済み → そのキーだけスキップして残りを実行
3. `--force` が指定されている場合はキャッシュを無視して再計算

## 検証

```
python -m pytest tests/ -q --tb=short
127 passed, 2 deselected, 5 warnings in 27.60s
```
