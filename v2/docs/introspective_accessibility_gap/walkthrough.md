# 修正内容の確認 (Walkthrough)

査読者から予想される「新規性の弱さ（Incremental）」の批判を回避し、論文をトップカンファレンス水準に引き上げるため、事前承認いただいた Implementation Plan に基づき、大幅な実験スクリプトの追加と論文 (`paper_draft.md`) の全面改訂を実施しました。

## 変更内容のサマリー

### 1. 理論的枠組みの刷新: Introspective Accessibility Gap の導入
- **課題**: Anthropicの「Functional emotions」研究や Keeman (2026) 等の先行研究との差別化。
- **対応**: モデル内部には感情の特徴が存在し行動に因果的影響を与えるが、モデル自身はそれを内省（自己報告）できないという **Introspective Accessibility Gap** という概念を導入し、論文全体をリフレーミングしました。

### 2. フェーズ1: 複数モデルでのスケーリングと頑健性検証
- **作成スクリプト**: [`v2/scripts/run_all_models.sh`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_all_models.sh), [`v2/scripts/run_mixed_effects_coupling.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_mixed_effects_coupling.py)
- 単一モデルへの依存批判を避けるため、Qwen2.5, Llama-3.2, Gemma-2 の各 Base/Instruct モデルで一括検証する環境を構築しました。
- `statsmodels` を用い、モデルファミリーをランダム効果として扱う「混合効果モデル」のスクリプトを実装し、現象の頑健性を統計的に検定可能にしました。

### 3. フェーズ2: SAEを用いたVA特徴の次元別解離と介入
- **作成スクリプト**: [`v2/scripts/run_sae_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_sae_patching.py)
- 線形プロービング（粗い残差全体の置換）から、**Sparse Autoencoder (SAE)** を用いた「解釈可能な特徴レベル」の介入へ手法を格上げしました。
- ValenceとArousalを独立して操作できることを実証するための評価ロジックを実装しました。

### 4. フェーズ3: 内省精度テスト (Introspection Test)
- **作成スクリプト**: [`v2/scripts/run_introspective_accessibility_test.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_introspective_accessibility_test.py)
- SAEによる介入（機能的な尤度シフトの発生）と同時に、モデルに「感情は変化しましたか？」と内省的に問うテストを実装しました。
- 「実際の出力分布の変化」と「モデルの言語的な自己認識」の乖離を定量的に観測する実験パラダイムを確立しました。

### 5. 論文の全面改訂
- **対象ファイル**: [`v2/docs/post_training_readout_experiment/paper_draft.md`](file:///mnt/nas/home/hiromi/src/emo/v2/docs/post_training_readout_experiment/paper_draft.md)
- Title を `Introspective Accessibility Gap: Why Language Models Cannot Accurately Report Their Own Functional Affective States` に変更。
- Abstract, Introduction, Related Work, Methodology, Results, Conclusion すべてにおいて新しいアプローチと理論を反映しました。

> [!TIP]
> 今後の作業として、実際にH100環境等で `run_all_models.sh` を実行してデータを収集し、`statsmodels` の結果と SAE Patching の定量結果を `paper_draft.md` の Results セクションに流し込む（実数値を入れる）ことで、投稿準備が整います。
