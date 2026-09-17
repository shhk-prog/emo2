# Implementation Plan: Token-level Path Patching / Unembedding Readout Experiment

## Goal Description
査読者（または指導教員）からのフィードバックに基づき、新規性を高めるための最小追加実験である「Token-level path patching (Unembedding readout)」を実装します。この実験により、BaseとInstructの間の自己報告の差が、「中間表現（Component）自体の差」に起因するのか、「中間表現から最終トークン（1〜9の数値）への直接的な読み出し（Readout / Unembedding）経路の差」に起因するのかを因果的に分解します。

## User Review Required
> [!IMPORTANT]
> 以下の実験設計で進めますが、プロンプトの扱いや抽出するトークン位置について意図と合致しているかご確認ください。
> 
> **提案するアプローチ:**
> 1. **Prompt Forcing**: `1〜9`の数値トークンのlogitを直接比較するため、プロンプトの末尾に `{\n  "valence": ` を強制的に付加した状態（`prompt_val`）で推論を行います。これにより、最終トークン位置（`target_pos`）での次トークン予測が直接数値（1〜9）になります。
> 2. **Direct Path Patching**: 中間コンポーネント（例: $MLP_{10}$）から最終Residual（$Residual_L$）への直接パス**のみ**をパッチします。
>    - 数式: $h_L^{(patched)} = h_L^{(Inst)} - c^{(Inst)} + c^{(Base)}$
>    - その後、この $h_L^{(patched)}$ を最終RMSNormとUnembeddingにかけて新しいLogitを計算します。
>    - この計算はフォワードパスを再実行せずとも、保存したテンソル間の演算で厳密に算出可能です。
> 3. **評価指標**: 
>    - 数値トークン（'1', '2', ..., '9'）のLogit変化（Candidate-token logit shift）
>    - パッチ適用前後の分布（Softmax後）における $\Delta E[V]$ および $\Delta WD$
>    - Token '5' の相対的Logit変化（relative logit prior）
> 4. **対照実験 (Controls)**: 
>    - ランダムな別の刺激文からのBase活性化を用いたパッチング（Random source）を対照群として計算します。

## Open Questions
> [!WARNING]
> 1. **ターゲットコンポーネントの指定**: Table 3の上位コンポーネント（Layer 10 res, 15 res, 16 res, 10 mlp, 14 attn）をSourceとしてハードコードしてスクリプトを作成してよいでしょうか？
> 2. **Base活性化の取得**: 今回はプロンプトの末尾が `{"valence": ` まで延長されるため、既存の `base_states.npz`（元のプロンプト長で取得したもの）は使えません。Baseモデルも同時に（または事前に）実行して延長プロンプトでの活性化を取得する実装にします。これで問題ありませんか？

## Proposed Changes
### `v2/scripts/run_path_patching.py`
以下の処理を行う新規Pythonスクリプトを作成します。

#### [NEW] `v2/scripts/run_path_patching.py`
- BaseモデルとInstructモデルの両方をメモリにロード（または順番に実行）します。
- AIPsy-AffectのTest splitの各テキストに対して、`prompt_val = original_prompt + '{\n  "valence": '` を作成します。
- Baseモデルで `prompt_val` を入力とし、指定されたComponent（例: L10 MLP）の最終トークン位置の出力 $c^{(Base)}$ を取得します。
- Instructモデルで `prompt_val` を入力とし、$c^{(Inst)}$ と最終Residual $h_L^{(Inst)}$ を取得します。
- $h_L^{(patched)} = h_L^{(Inst)} - c^{(Inst)} + c^{(Base)}$ を計算し、Instructモデルの `model.model.norm` と `model.lm_head` に通してパッチ後のLogitsを得ます。
- トークン '1'〜'9' に対するLogitを抽出し、パッチ前後の $E[V]$ やLogitの変動をCSVに保存します。
- Random controlも並行して計算します。

## Verification Plan
### Automated Tests
- `python v2/scripts/run_path_patching.py --limit 5` を実行し、Logitの抽出と $h_L^{(patched)}$ の計算がエラーなく行われるか確認します。
- 結果のCSVを確認し、Logitや分布がNaNになっていないこと、またパッチングによってInstructの予測（通常は'5'が最大）がBaseの予測にシフトするかを定性的に確認します。

---

## Phase 8: Main-Conference Level Additions (New Proposed Plan)

### Goal
現在の論文は「頑健な機構的比較研究」として完成度が高いですが、さらにMain Conference (NeurIPS / ICLR等) の Findings / Main-track 水準へ引き上げるため、査読者からの指摘を見越した以下の追加検証を提案します。

現在、「複数モデルでの再現」はデータセットの再生成、フック位置の再定義（LLaMA-3等のアーキテクチャ依存の修正）など膨大なコストがかかるため、**既存の Qwen2.5-1.5B 環境で完結しつつ最大のインパクトをもたらす以下の2点（Option 2, Option 3）の実施を優先して提案します。**

### Proposed Additions

#### 1. 2D joint-distribution指標の追加 (Option 2)
現在の 1次元 Wasserstein Distance ($WD_V$) は Valence周辺分布のみを比較しており、Valence-Arousal の相関（Joint structure）を破棄しています。
- **実装内容**: `scipy` や `pot` (Python Optimal Transport) ライブラリを用いて、81状態の完全な結合確率分布に対する以下の指標を計算するPythonスクリプトを作成します。
  - **Jensen-Shannon Divergence (JSD)**
  - **2D Earth Mover's Distance (2D Wasserstein)**: 基底距離関数を $d((v,a), (v',a')) = \sqrt{(v-v')^2 + (a-a')^2}$ とした厳密な2D距離。
- **目的**: 8条件Swap分析やDirect-contributionテストの結果が、単なる周辺分布のシフトではなく、結合空間全体の構造シフトとして成立していることを証明します。

#### 2. 上流componentからfinal residualまでの厳密path patching (Option 3)
現在の "Direct final-residual contribution test" は、上流からの線形成分を最終残差から数学的に引き算・足し算して近似するものでした。
- **実装内容**: `TransformerLens` ライクな「真の Causal Scrubbing (Path Patching)」を実装します。
  1. Instructモデルでフォワードパスを実行し、全てのアクティベーションをキャッシュ。
  2. Baseモデルでフォワードパスを実行し、特定の上流コンポーネント（例: `res_15`）のアクティベーションをキャッシュ。
  3. 再度Instructモデルでフォワードパスを実行する際、最終層のAttention/MLPへの**入力時のみ**、その上流コンポーネントからの寄与をBase由来のものに置き換える（Frozen computational graph への局所的介入）。
- **目的**: 単なる線形近似ではなく、「上流から最終層への直接の因果エッジ（Direct Edge）」を選択的に検証し、単一ショートカット説を完全に棄却します。

### User Review Required
> [!IMPORTANT]
> 1. 上記の2つの追加実装（2D Joint指標、厳密なPath Patching）を進める方針でよろしいでしょうか？（もしどうしても「他モデルでの検証」を優先されたい場合は、計算コストはかかりますがLLaMA-3等への対応計画を作成します）。
> 2. 【システム上の制約事項】現在、ターミナル実行環境（Standard Sandbox）で障害が発生しており、自動でPythonスクリプトを実行できません。実験スクリプトを実行する際は、UIから毎回「Approve（Sandbox Bypass）」をクリックして承認いただく必要がございます。この点をご了承いただけますでしょうか。

