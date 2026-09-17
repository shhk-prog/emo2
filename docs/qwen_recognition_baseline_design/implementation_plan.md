# 実装計画書: Qwen2.5-1.5B-InstructにおけるRecognition Baseline測定および同一モデル論理閉包

## 1. 背景と問題意識（査読対策の必然性）
現在の研究ストーリーは以下の流れになっています：
1. **GPT-4o × EmoBank**: 感情認識（$r_V=0.921, r_A=0.590$）と自己報告の中立化（98.6%）を観測。
2. **モデル乗り換え**: メカニズム解析のために open-weight の `Qwen2.5-1.5B-Instruct` に移行。
3. **Qwen × AIPsy-Affect**: 線形プロービング、Ablation、Activation Patching を実施。

ここで査読者から必ず寄せられる致命的な批判は：
> **「内部解析対象のQwen自体が、そもそも刺激の感情を正しく認識（認知的理解）しているのか？ 認識能力が低いから自己報告が中立化しているだけではないのか？ なぜGPT-4oで観察した前提がQwenにも成り立つと見なせるのか？」**

モデルを途中で交換しているため、この批判に対して現行の草稿だけでは反論が困難です。

---

## 2. 解決策：Qwenでの同一モデル論理閉包（3段階アラインメント）

### (1) EmoBankにおけるQwen Recognition Baseline
GPT-4oと同一のEmoBank刺激（$N=322$ または $N=3,210$）をQwenに与え、人間注釈との相関（$r_V, r_A$）および自己報告中立化率を計測し、以下の比較表を完成させます。

| Model | Valence $r$ | Arousal $r$ | Self-Report Neutralization % |
|---|:---:|:---:|:---:|
| **GPT-4o (予備観察)** | 0.921 | 0.590 | 98.6% (モード `{"valence": 5, "arousal": 5}`) |
| **Qwen2.5-1.5B-Instruct** | **[測定]** | **[測定]** | **[測定]** |

### (2) AIPsy-AffectにおけるQwen Recognition vs. Self-Report
本実験データセットである AIPsy-Affect（統制ペア）において、Qwenに以下の2条件を実行させます：
1. **Recognition Task**: 「この文章が一般読者に喚起する感情（Valence/Arousal）を推測せよ」
2. **Self-Report Task**: 「この文章を読んだあなた自身の感情状態を報告せよ」

これにより、**「RecognitionではAffectiveとNeutralの差が明確に現れるのに、Self-reportではその差が消失する」** ことを実証します。

### (3) 同一モデル内で閉じる4段階の論理構成
$$\text{刺激} \;\longrightarrow\; \underbrace{\text{Recognition}}_{\text{Qwenは認識できる}} \;\longrightarrow\; \underbrace{\text{Internal Representation}}_{\text{Qwen内部に情報がある (Probe)}} \;\longrightarrow\; \underbrace{\text{Self-Report}}_{\text{しかし自己報告は中立}} \;\longrightarrow\; \underbrace{\text{Intervention}}_{\text{因果レバレッジはどこにあるか}}$$

---

## 3. 実装スクリプト設計案

### スクリプト構成: `scripts/run_qwen_recognition_baseline.py`
1. **対象データセット**:
   - `v1/data/processed/stimuli.csv` (EmoBank: 322件)
   - `v2/data/processed/aipsy/` (AIPsy-Affect Strict: 144件、または Expanded: 422件)
2. **評価プロトコル**:
   - **Sequence Likelihood Protocol**（81通りの候補列 $P(V, A \mid x)$ による期待値 $E_{\mathrm{rec}}[V], E_{\mathrm{rec}}[A]$ の算出）を採用し、離散生成崩壊を完全に排除。
   - 比較のため、標準的な Greedy Generation による JSON パースも同時記録。
3. **所要時間・GPUリソース見積もり**:
   - Qwen2.5-1.5B-Instruct, バッチ推論（Batch Size = 64〜128）
   - EmoBank (322件) + AIPsy-Affect (144件) = 計466件
   - 1GPU（H100/H200等）で **約1〜2分以内** で完了可能（極めて低コスト）。

---

## 4. 論文ストーリーの再編成案
- **GPT-4o × EmoBank**: 「第1章 / Preliminary Observation（動機付け）」へ位置づけ直し。
  「商用フロンティアモデルにおいて認知的理解と自己報告の解離が観測されるが、重みが非公開のため機構解析が不可能である」
- **Main Investigation (Qwen2.5-1.5B)**:
  「オープンモデルQwenにおいて、同一の認知的認識（Recognition Sanity Check）を確認した上で、内部表現プロービングおよび因果介入へと接続する」
