# 修正内容の確認 (Walkthrough)

ユーザーの皆様からの詳細なレビューとご要望に基づき、論文原稿（`docs/paper_draft.md`）の改訂および詳細化を完了しました。以下に主な変更点と、それらがどのように反映されたかを要約します。

## 1. 論文構成の再編と情報量の復元
第1章〜第9章と第10章（拡張実験計画）が混在していた構成を整理し、全体を一本の論理的な流れ（第1章〜第10章）に統合しました。また、省略されていた実験設定や結果の表を復元し、全体的な記述を詳細化しました。
- 旧第10章のデータセット（AIPsy-Affect, EmoBank等）の詳細な表を第4章（Data and Experimental Protocol）に復元しました。
- 各フェーズ（Baseline, Recognition, Reception）のプロンプト入力と期待される出力の具体例を表として追加しました。
- APIモデル（gpt-4o）による完全なニュートラル応答への収束（Phase 1-a）の定量的結果を表として詳細に記載しました。
- Probing、Patching、Ablation、Base vs Instructの各実験結果のテーブルをすべて復元し、具体的な数値データ（層別のRecovery/RemovalやSuppression Gap）を明記しました。

## 2. 関連研究と考察の拡充
- **関連研究（第2章）**: EmotionBenchやWang et al.、FACET等の行動評価研究と、Tak et al.やAnthropic等の内部表現解析（Mechanistic Interpretability）の研究を明確に対比させ、本研究の独自性（両者を同一タスク内で結びつける点）を詳細に説明しました。
- **考察（第9章）**: 初期に議論されていた「構造的解離（Structural Dissociation）」、「層別の機能的階層（Layer-wise Functional Hierarchy）」に関する詳細な論考を復元しました。ただし、断定を避け、これらが「候補位置」であり、「alignment suppression hypothesisと整合的である」という慎重な表現に留めています。

## 3. 因果的主張の緩和と正確な記述
全体を通して「〇〇確定」といった強い断定を避け、科学的に安全な解釈へとトーンダウンしました。
- **Patching / Ablation**: 単一位置の残差ストリームへの介入であることを明記し、「局在的な因果媒介」を確定させるものではなく、あくまで「候補位置の特定」にとどまること、および分布外介入（OOD）リスクについてDiscussionで言及しました。
- **Base vs Instruct**: 「Alignment Suppression確定」という表現を、「alignment suppression hypothesisと整合的な証拠」「post-training依存の因果的乖離」といった表現に修正しました。
- **Probing**: 高い線形復元精度（ROC-AUC > 97.5%）が直ちに因果的寄与を意味しないことを、第6章の結論部分に理論的制約として明示しました。

## 4. 評価指標・数式表記の統一と修正
- $E[V \mid x], E[A \mid x]$ を $\sum\sum$ を用いた数式に統一しました。
- Patching（十分性）の指標として $\mathrm{Recovery}_{VA}$、Ablation（必要性）の指標として $\mathrm{Removal}_{VA}$ をベクトル距離ベースで対称に定義しました。
- 分母が小さいペアの除外閾値（$\tau$）に関する記述を追加し、Steeringの結果については「意図通りの制御は未確認」としてExploratory（探索的）な位置づけに変更しました。

## 5. 実装要件・自動テストの明記
- `prompt_last_token` が単なる末尾インデックスではなく、attention maskに基づいて非パディングトークンから求められるべきである旨を第4章に記載し、検証事項であることを付記しました。
- 321刺激の抽出に関する「各セル50件」という推測の記述を削除し、データ選定スクリプトと集計表を最終稿で公開する形に変更しました。
- 論文の第9章（Discussion, Limitations, and Ethics）内に、「本研究の検証要件（自動テスト・実験的制約）」として、$\alpha=0$ 時の挙動や81候補の対数尤度の一致など、8項目のテスト・検証要件を明記しました。
