# V2 コード修正・統計パイプライン完全化 実装計画書

## 概要

提示された批判的レビューに基づき、V2研究パイプライン（RQ1〜RQ4）における実行ブロッカー、トークナイズ・尤度計算の厳密化、対照実験・統計検定の未接続部分の全12項目を網羅的に修正・実装します。

---

## 修正対象・課題の整理と解決策

### 1. 【最優先】`affective_empathy_eval.models` の追跡漏れと editable install の統一
- **原因特定**: `.gitignore` の 12 行目に `models/` と記述されていたため、Git が `src/affective_empathy_eval/models/` 以下のすべてのファイルを自動除外（ignore）していました。また、`.venv` の editable install が `v1/src` を指しており、モジュールが分散していました。
- **解決策**:
  1. `.gitignore` を修正し、ソースコードディレクトリである `models/` が除外されないよう `/models/` 等に限定、`!src/**/models/` `!v1/**/models/` を明示。
  2. `src/affective_empathy_eval/` 以下のモジュールを `v1/src/affective_empathy_eval/` にも同期（シンボリックリンクまたはコピー）、どちらから import されても `affective_empathy_eval.models` が 100% 解決するように設定。

### 2. 【科学的妥当性】共通トークナイズ関数 `encode_prompt_canonical` の導入（指摘2）
- **課題**: activation extraction, semantic anchor 検出, sequence likelihood, patching でトークナイズ手法や special tokens の有無が異なると、パッチ位置と生成境界がずれるリスク。
- **解決策**:
  - `src/affective_empathy_eval/prompts.py` に `encode_prompt_canonical(tokenizer, prompt)` を追加。
  - `add_special_tokens=False`（chat template 適用済みテキストにはすでに特殊トークンが含まれる）で一貫してエンコードし、全パイプラインでこの共通関数を介して `input_ids` を生成。

### 3. 【科学的妥当性】Sequence Likelihood での joint encoding（指摘3）
- **課題**: `tokenize(prompt) + tokenize(candidate)` と `tokenize(prompt + candidate)` が BPE 境界で不一致になる現象の防止。
- **解決策**:
  - `src/affective_empathy_eval/likelihood.py` の `compute_sequence_likelihoods_for_candidates`:
  - `full_text = prompt + candidate` として joint tokenize を実施。
  - `prompt_ids` の境界を特定し、プレフィックス以降の candidate トークン部分のスライシングと評価ステップを厳密に対応づける。

### 4. 【対照実験】`matched_plain` 条件の解析への完全接続（指摘4）
- **課題**: `run_v2_2x2_cross_decoding.py` で `inst_matched` を抽出しているが解析に使われていない。
- **解決策**:
  - `analyze_v2_geometry_and_sharing()` を拡張し、以下の3比較をすべて算出：
    1. Base plain $\leftrightarrow$ Instruct native-chat（主比較）
    2. Base plain $\leftrightarrow$ Instruct matched-plain（純粋な事後学習効果）
    3. Instruct native-chat $\leftrightarrow$ Instruct matched-plain（プロンプト形式・Chat template 効果）
  - Procrustes 歪み、RSA、クロスデコーディングで各比較を算出して保存。

### 5. 【概念定義】RQ2 Sharing score $\Delta Share(l)$ の Primary 化（指摘5）
- **課題**: `delta_delta_cross` が意図した「差の差」になっていない。
- **解決策**:
  - Primary 指標として以下を定義：
    $$Share_B(l) = \frac{R^2_{BR \to BS}(l) + R^2_{BS \to BR}(l)}{2}$$
    $$Share_I(l) = \frac{R^2_{IR \to IS}(l) + R^2_{IS \to IR}(l)}{2}$$
    $$\Delta Share(l) = Share_I(l) - Share_B(l)$$
    （$\Delta Share < 0$: Reader/Self 分化、$\Delta Share > 0$: Reader/Self 統合）
  - matched-plain に対しても同様に $Share_{I,\text{matched}}(l)$ を算出。
  - 既存の `delta_delta_cross` は secondary 指標として維持。

### 6. 【統計検定】RQ3 の pair-level 出力と LMM 検定（指摘6）
- **課題**: layer 平均の因果効果しか保存されておらず、サンプル単位の統計検定ができない。
- **解決策**:
  - `run_v2_2x2_causal_map.py` で各刺激 $i$、各層 $l$ の因果変位 $C_{i,l}$（Valence, Arousal）をすべて保持。
  - long-form DataFrame として `v2_causal_pair_level.csv` を出力（列: `pair_id`, `family`, `alignment`, `task`, `layer`, `relative_depth`, `c_v`, `c_a`）。
  - `fit_sample_level_lmm` を用いて $C \sim \text{Alignment} \times \text{Task} \times \text{Depth} + (1|\text{pair})$ を推定し、結果 JSON にパラメータと p 値を保存。

### 7. 【完全性】RQ3 の 4条件解離量（dissociation）の完全算出（指摘7）
- **課題**: Base Self の解離量しか計算されていない。
- **解決策**:
  - BR (`base_reader`), BS (`base_self`), IR (`inst_reader`), IS (`inst_self`) の 4条件すべてでデコーダビリティプロファイルと因果プロファイルを突合。
  - 各条件の $\Delta d^*$（ピーク解離）および $\Delta \bar{d}$（重心解離）を計算。
  - 事後学習による解離の変化量 $(\Delta d_{IS} - \Delta d_{BS})$ と $(\Delta d_{IR} - \Delta d_{BR})$ を算出。

### 8. 【実行安全性】RQ4 の Base activation 全層一括 capture（指摘8）
- **課題**: 存在しない `extract_activation` を呼んでおり、1層ずつ hook manager を作成していたため実行不能。
- **解決策**:
  - `with ActivationHookManager(adapter_base) as hook_mgr:` を開き、全層に対して `register_capture_hook` を登録。
  - 1回の `model_base(**enc_b)` forward で全層の活性化を取得してテンソルとして保存。

### 9. 【測定妥当性】RQ4 の sample-wise EMD と Recovery 算出（指摘9）
- **課題**: 全サンプルの平均分布同士で EMD を計算していたため、刺激ごとの相殺が起きる。
- **解決策**:
  - 刺激 $i$ ごとに Base vs Instruct の初期 EMD $EMD_i^{\text{baseline}} = EMD(P_{\text{Inst},i}, P_{\text{Base},i})$ を計算。
  - パッチング後も $EMD_{i,l}^{\text{patched}} = EMD(P_{\text{Patched},i,l}, P_{\text{Base},i})$ を計算。
  - サンプル単位の回復率 $Recovery_{i,l} = \frac{EMD_i^{\text{baseline}} - EMD_{i,l}^{\text{patched}}}{EMD_i^{\text{baseline}} + \epsilon}$ を算出。
  - 全体平均 $\overline{EMD}_l, \overline{Recovery}_l$、およびサンプル単位値に対する Bootstrap CI を算出。

### 10. 【研究の核心】RQ4 の Reader / Self 両タスク比較（指摘10）
- **課題**: Self タスクしかパッチングしていない。
- **解決策**:
  - `TaskType.READER` と `TaskType.SELF` の両方で Base $\to$ Instruct recovery を実施。
  - Reader 回復率プロファイルと Self 回復率プロファイルを比較し、Base 活性化による回復効果が Self で特異的に強いかを検証。

### 11. 【統計解析】Bootstrap CI のパイプライン完全接続（指摘11）
- **課題**: `compute_bootstrap_ci` がインポートされただけで呼ばれていない。
- **解決策**:
  - クロスデコーディングのピーク深度 CI、重心 CI、$\Delta Share$ CI、RQ4 回復率 CI を実際に bootstrap リサンプリング（$B=1000$）で計算し、`derived_dir` のサマリー JSON に保存。

### 12. 【用語整合性】アンカー名称の整理（指摘12）
- **課題**: `response_start` が実際にはプロンプトの末尾トークンを指しており誤解を招く。
- **解決策**:
  - `prompts.py` において `prompt_end`（または `pre_response`）を主名称とし、後方互換用エイリアスとして `response_start` も保持。
  - ドキュメントおよび各スクリプトで意図（生成直前のコンテキスト末尾状態）を明記。

---

## 修正対象ファイル一覧

| コンポーネント | 対象ファイル | 変更種別 | 主な変更内容 |
|:---|:---|:---:|:---|
| **Git / パッケージ** | [.gitignore](file:///mnt/nas/home/hiromi/src/emo/.gitignore) | [MODIFY] | `models/` 除外を解除、コード追跡保証 |
| **ライブラリ** | [src/affective_empathy_eval/prompts.py](file:///mnt/nas/home/hiromi/src/emo/src/affective_empathy_eval/prompts.py) | [MODIFY] | `encode_prompt_canonical` 追加、`prompt_end` 名称整理 |
| **ライブラリ** | [src/affective_empathy_eval/likelihood.py](file:///mnt/nas/home/hiromi/src/emo/src/affective_empathy_eval/likelihood.py) | [MODIFY] | joint encoding による候補尤度計算、境界処理 |
| **パッケージ同期** | `v1/src/affective_empathy_eval/` | [MODIFY] | 最新モジュール（models等）を同期して editable install を完全対応 |
| **RQ1 & RQ2** | [v2/scripts/run_v2_2x2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_2x2_cross_decoding.py) | [MODIFY] | `matched_plain` 接続、$\Delta Share$、Bootstrap CI |
| **RQ3** | [v2/scripts/run_v2_2x2_causal_map.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_2x2_causal_map.py) | [MODIFY] | pair-level 出力、LMM検定、4条件解離量、canonical tokenization |
| **RQ4** | [v2/scripts/run_v2_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_recovery_patching.py) | [MODIFY] | 全層一括 capture、sample-wise EMD/回復率、Reader/Self 両タスク、Bootstrap CI |
| **テスト** | [tests/test_likelihood.py](file:///mnt/nas/home/hiromi/src/emo/tests/test_likelihood.py) 他 | [MODIFY] | joint encoding や canonical tokenization のユニットテスト追加 |

---

## 検証手順

### 1. 自動ユニットテスト
```bash
.venv/bin/python -m pytest tests/ -v
```
- `test_likelihood.py`: joint tokenization の動作検証、81候補尤度の正規化
- `test_geometry.py`: $\Delta Share$ および重心・ピーク計算の検証
- `test_models_hooks.py`: 一括 capture フックおよびパッチングフックの動作検証
- `test_statistics.py`: Bootstrap CI および LMM の動作検証

### 2. 統合 Dry-run 実行
```bash
.venv/bin/python v2/scripts/run_v2_2x2_cross_decoding.py --dry-run
.venv/bin/python v2/scripts/run_v2_2x2_causal_map.py --dry-run
.venv/bin/python v2/scripts/run_v2_recovery_patching.py --dry-run
```
- 各スクリプトが例外なく完走し、JSON / CSV 生成物が正しいフォーマットで出力されることを確認。
- `v2_causal_pair_level.csv` が出力され、LMM 検定結果が含まれることを確認。
- `v2_distribution_recovery_summary.json` に Reader と Self 両方の sample-wise EMD および回復率と Bootstrap CI が含まれることを確認。
