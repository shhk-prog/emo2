# 論文改訂および追加実験の実装計画 (Implementation Plan)

## Goal Description
本計画の目的は、`paper_draft.md` に対して査読者から予想される「増分的(incremental)」という批判を回避し、トップ会議（ACL/EMNLP/NeurIPS等）に採択される水準まで新規性を引き上げることです。
具体的には、以下の4つの方向性で論文を再構成し、必要な追加実験を実施します。
1. **次元別解離の検証**: VA空間内での次元別解離（ValenceとArousalの非対称性）の検証。
2. **SAEベースの特徴分解**: 線形プロービングからSAEによる解釈可能な特徴レベルの介入への格上げ。
3. **モデル横断的な頑健性**: Qwen2.5, Llama-3.2, Gemma-2等の複数モデル・スケールでの検証と混合効果モデルによる検定。
4. **Introspective Accessibility Gap**: Anthropicの機能的感情研究と対比し、「行動への因果的影響」と「内省的報告能力」の解離という新しい理論的枠組みの提示。

> [!IMPORTANT]
> ## User Review Required
> 本計画には大幅な実験の追加（SAEの学習/推論、複数モデルの評価）と論文の全面的な再構成が含まれます。計算リソースや実行時間に関する想定を確認の上、承認をお願いします。

> [!WARNING]
> ## Open Questions
> 1. **モデルの選定**: Qwen2.5 (0.5B/1.5B/7B), Llama-3.2 (1B/3B), Gemma-2 (2B/9B) のbase/instructペアを想定していますが、手元のH100(80GB)環境でこれらすべてのSAE処理と介入実験を実行するのに十分な時間とストレージリソースは確保されていますか？（必要に応じてモデルを絞ることも可能です）
> 2. **SAEの学習済み重み**: GemmaやLlamaについてはGemma Scope等で既存の学習済みSAEが利用可能かもしれませんが、Qwen2.5に関しては自前でSAEを学習する（または既存の公開SAEを探す）必要があります。SAEはゼロから学習させますか？それとも既存のものを利用しますか？

---

## Proposed Changes

### [MODIFY] 論文ドラフト (`v2/docs/post_training_readout_experiment/paper_draft.md`)

論文のトーンを「行動と内部表現の乖離の観測」から「Introspective Accessibility Gap (内省的アクセス可能性の欠如)」の証明へとリフレーミングします。

- **Title & Abstract**:
  - タイトル案: "Introspective Accessibility Gap: Why Language Models Cannot Accurately Report Their Own Functional Affective States"
  - Abstractを修正し、SAEを用いた精密な介入と複数モデルでの頑健性、そしてAnthropic研究との対比を強調。
- **1. Introduction**:
  - Anthropicの「Functional emotions (機能的感情)」パラダイムを紹介。
  - 問題提起：「モデルは感情を行動（出力生成）のステアリングに用いるが、それについて正しく内省（自己報告）できるのか？」
  - 新規性の主張として「Introspective Accessibility Gap」を定義。
- **2. Related Work**:
  - **対比の明確化**:
    - *Anthropic (2026)*: 感情の因果的機能は示したが、内省との解離は示していない。
    - *Keeman (2026) "Whether, Not Which"* / *Tak et al. (2025)*: receptionとcategorizationの解離や、appraisalの因果性は示したが、SAEレベルの精密な次元別解離や内省の限界までは踏み込んでいない。
- **3. Methodology**:
  - SAE (Sparse Autoencoder) を用いたVA特徴の抽出と分離（直交性）手法の追加。
  - SAE特徴単位でのActivation Patching / Ablation手法の記述。
  - モデルをランダム効果として扱う混合効果モデル（Mixed-effects modeling）の詳細を追加。
  - 内省精度テスト（モデル自身の状態変化推測）のプロトコル追加。
- **4. Results**:
  - **Result 1**: 5モデルファミリー×複数スケールにおける自己報告尤度差の中立収束（混合効果モデルによる検定結果）。
  - **Result 2**: SAEを用いたVA特徴の分解。ValenceとArousalが疎な特徴空間で独立して操作可能であることの証明。
  - **Result 3**: SAE特徴レベルのPatching。層全体（residual stream）の置換ではなく、解釈可能な特定SAE特徴のみの因果的寄与の特定。
  - **Result 4**: Introspective Accessibility テストの結果。Patchingによる内部状態の変化を、モデル自身が正しく内省できない（実際の行動/尤度変化と自己報告が一致しない）ことの実証。
- **5. Discussion & Conclusion**:
  - 査読者の懸念（単一モデル依存、指標の不安定性、既存研究との重複）を完全に払拭する論理構成への刷新。

---

## Verification Plan (追加実験の実施計画)

論文の改訂を裏付けるため、以下の追加実験を実装・実行します。これらは段階的に進めます。

### フェーズ1: 複数モデルでのスケーリングと頑健性検証
- Qwen2.5, Llama-3.2, Gemma-2 の各Base/Instructモデルに対して、既存のSequence Likelihood Protocolを実行。
- 結果を統合し、`statsmodels` を用いてモデルをRandom Effectとした混合効果モデルを推定。

### フェーズ2: SAEを用いたVA特徴抽出と介入
- 対象モデル（まずはQwen2.5-1.5B等の代表モデル）の中間層に対して、SAEを適用。
- VA空間に対応するTop-k SAE特徴を単離し、ValenceとArousalが独立して因果的に操作可能か（次元別解離）を検証。
- 単離したSAE特徴のみを用いたPatchingスクリプトの実装と評価。

### フェーズ3: 内省精度テスト (Introspective Accessibility)
- SAE Patching後のモデルに対して、「あなたの内部状態は今どのように変化しましたか？」という内省を促すプロンプトを与え、推測結果と実際の自己報告尤度変化の乖離を定量化。
- Greedyデコード頻度とSequence Likelihood分布の一致度検証。

### フェーズ4: 交絡排除とベースライン比較の主結果統合
- シャッフルラベル、文長、TF-IDFベースラインとの比較を整理し、本文の主結果として組み込む。

本計画が承認され次第、必要なタスクリスト (`task.md`) を作成し、フェーズ1の実験スクリプト実装から着手します。
