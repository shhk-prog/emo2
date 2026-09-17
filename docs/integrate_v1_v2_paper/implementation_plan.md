# v1・v2論文統合および再構成の実装計画

## 背景・目的
`/mnt/nas/home/hiromi/src/emo/v1` と `/mnt/nas/home/hiromi/src/emo/v2` の研究成果を統合し、一つの詳細な論文として `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` にまとめる。既存の「LLMは感情を持つか」というアプローチではなく、「LLMの情動的自己報告は因果的内部状態を追跡するか（introspective faithfulness）」という観点から、Post-training前後のrepresentation-to-report mappingの変化を解明する論文として構成する。

## 中心命題 (Central Story)
**Post-training後、LLMの情動的自己報告が内部表現を反映しなくなるとき、その乖離はどこで生じるのか？**
経路モデル: `Stimulus -> Affective Representation -> Readout / Routing -> Self-report`

## 推奨タイトル
**When Affective Self-Reports Do Not Trace Internal Representations: Post-Training-Associated Remapping from Affective Representations to Self-Reports**
(日本語仮題: LLMの情動的自己報告は因果的内部状態を追跡するか：反実仮想介入による内省忠実性と再マッピングの検証)

## 論文の構成案
提供された方針に従い、以下の章立てで統合する。

1. **Introduction**
   - 問題提起: 自己報告の中立化（neutralization）はどこで生じるのか。Erasure, Transformation, Suppression, Distributed remapping の4つの可能性を提示。
2. **Related Work**
   - 2.1 Behavioral affect and self-report (EmotionBench, introspection)
   - 2.2 Affective representations (VA geometry, AIPsy-Affect, appraisal)
   - 2.3 Causal routing and post-training (activation patching, steering, refusal direction)
3. **Experimental Framework**
   - 3.1 Operationalization (affect-relevant representation / constrained affective self-report distribution)
   - 3.2 Datasets (EmoBank / AIPsy-Affect)
   - 3.3 Models (Qwen2.5-1.5B Base/Instruct を主軸に)
   - 3.4 Sequence Likelihood Protocol (v1から移行。論文全体の共通 measurement layer)
4. **Behavioral Observation**
   - 4.1 Recognition vs Self-report (GPT-4o)
   - 4.2 Neutralization under likelihood scoring (Qwen Base/Instruct)
   - 4.3 Robustness (temperature / natural language formatting)
5. **Is Affect-Relevant Information Erased?**
   - Probing, intensity response, cross-decoding, RSA による H1(Erasure) / H2(Transformation) の検証。
   - 結論: Erasure alone is insufficient; representation geometry changes.
6. **Does the Representation Causally Affect Self-Report?**
   - Ablation, within-model patching, steering, controls.
   - 結論: Decodable information can causally influence self-report.
7. **Where Does Post-Training Change the Mapping?**
   - 7.1 Coupling (Mixed-effects model)
   - 7.2 Component patching
   - 7.3 Synergy
   - 7.4 Dose-response
   - 7.5 Late-residual substitution
   - 7.6 RMSNorm/unembedding swap
8. **Discussion & Conclusion**
   - 4段階での結論まとめ:
     1. Self-report neutralization exists.
     2. Affect-relevant information remains decodable.
     3. That information can causally affect report distributions.
     4. But post-training changes how that information is mapped into self-report, and the change is not well explained by simple global suppression or output-head modification.

## 変更・削除する内容（Appendix移行または削除）
- 「LLMが感情を経験しているか」の過度な議論（限界の段落に留める）
- v1のコード構造、batch size最適化、scaling ratioの発散結果、開発Phase番号、全プロンプト全文
- 異なるRecovery定義の混在（v1の83.3%とv2のWD改善を混同させない）

## 成果物
- 統合論文ファイル: `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md`
  - 日本語で執筆（プロンプトやモデル名などは英語併記）
  - 学術論文のフォーマットに従い、Markdown形式で出力。
- 作業履歴およびプロジェクトルールに基づく保存:
  - `task.md`, `walkthrough.md`, `implementation_plan.md` を `/mnt/nas/home/hiromi/src/emo/docs/integrate_v1_v2_paper/` 内に複製保存する。

> [!IMPORTANT]
> **User Review Required**
> 1. 上記の章立て・構成で `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` を全面的に上書きしてよろしいでしょうか？
> 2. 原稿は指定がない限り日本語で記述しますが、英語での執筆をご希望の場合はお知らせください。
