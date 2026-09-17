# 論文改訂計画 (Implementation Plan)

本計画は、査読に向けた論文原稿（`docs/paper_draft.md`）の改訂方針と具体的な修正手順をまとめたものです。ユーザーのレビューフィードバックに基づき、過剰な因果的主張を緩和し、実験結果の矛盾を解消し、論文構成を再編します。

## 目的 (Goal)
- 第10章（拡張実験計画・結果）を本文（第6〜8章）に統合し、論理的な一貫性を持つ構造に再編する。
- 強い因果的主張（「自己非感情化回路の確定」「Alignment Suppression確定」等）を適切な表現に修正し、学術的な信頼性を向上させる。
- 評価指標の数式表記の統一と、SteeringやAblationに関する解釈・符号の不整合を修正する。
- 実装の詳細（`prompt_last_token`やデータセットの件数等）を正確な記述に直す。

## User Review Required

> [!IMPORTANT]
> 以下の修正内容・方針に問題がないかご確認をお願いします。特に、Steeringの結果についての取り扱い（「効果未確認」とするか、符号反転による説明を追加するか）に関する判断が必要です。

## Open Questions

> [!WARNING]
> 1. **Steeringの結果について**: ご指摘の通り、$\alpha=\pm 2.0$ の $E[V]$ の差が極めて微小であり、かつ符号が想定と逆になっています。この結果を踏まえ、「滑らかで意図通りの制御が可能」という主張は撤回し、「Steeringによる意図通りの制御効果は確認されなかった（未確認）」と記述する方針でよろしいでしょうか？ または、係数の定義などを修正した上での再記述が必要でしょうか？（本計画ではひとまず「未確認」としてトーンダウンする方向で記載します）
> 2. **Ablation指標の符号について**: $\mathrm{EffectRemoved}$ の符号反転リスクを避けるため、提案いただいたVAベクトルに基づく $\mathrm{Recovery}_{VA}$ （あるいはEffect Removedのベクトル版）を主結果の指標として採用する方針でよろしいでしょうか？
> 3. **321件と450件の不一致について**: 現状の原稿には「3×3分割空間から各セル50件標的抽出」とありますが、実際の抽出件数（あるいは除外された件数）について、正確な数値がございましたら教えていただけますでしょうか？（ひとまず「特定のセルで不足があり計321件となった」といった記述に修正するか、450件抽出予定であったが品質フィルタリング等で321件に絞られた等の注釈を追加します）

## Proposed Changes

### 文書構成の再編
現在の第1〜9章と第10章が混在する構造を、以下の章立てに再構成します。
1. Introduction
2. Related Work
3. Task Definition and Research Questions
4. Data and Experimental Protocol
5. Behavioral Results: Recognition–Self-report Dissociation
6. Representation Results: Minimal-pair Probing
7. Causal Intervention Results
8. Base–Instruct Comparison
9. Discussion, Limitations, and Ethics
10. Conclusion

### 各章の具体的な修正内容

#### 第5章・第6章（および再編後の章）: ネガティブ結果とポジティブ結果の統合
- **修正前**: 「単一位置への介入では感情の回復は観測されなかった。局在的な因果媒介は支持されない」という断定。
- **修正後**: 初期のgreedy-decoding / JSONパース評価では安定した離散的出力の回復は観測されなかったが、後続のシーケンス尤度ベース評価では生成崩壊を伴わない微小な方向性変化を検出でき、一部の候補層で部分的なRecoveryが観察されたことを明記。評価指標の更新による結果の違いとしてストーリーを統合する。

#### Base vs Instruct の比較主張の緩和
- **修正前**: 「Alignment Suppression確定」「自己非感情化回路」「再配線を解明した」
- **修正後**: 「alignment suppression hypothesisと整合的な証拠」「Base-to-Instruct post-training差と関連する感度減衰」「自己非感情化出力に寄与する候補部品」「post-training依存の因果的乖離を観察した」等へトーンダウン。RLHF単独の効果ではなく、post-training全体の差異として慎重に記述する。

#### 数値・数式の整合性
- 期待値 $E[V \mid x], E[A \mid x]$ の数式を $\sum \sum$ を用いた統一的な表記に修正。
- Recovery分母の小ささによる不安定性を注記し、事前定義閾値未満のペアの除外や、絶対変化 $\Delta E[V]$ の併記、中央値・IQR・CIの報告について追記する。
- Ablation（EffectRemoved）の符号の解釈反転について注記し、必要に応じて Source/Targetの期待VAベクトルを用いた指標（$\mathrm{Recovery}_{VA,i}^{(l)}$）の式を提示する。

#### Probing と Patching/Ablation の解釈の限定
- **Probing**: 「セマンティック表現が強固に存在する」「極端な表層模倣説を明確に否定する」といった表現を避け、「線形復元可能であることを示す。ただしProbe精度のみから因果関係は結論づけない」と記述する。AIPsy-Affectのtopic-domain一般化やコントロールラベルに関する記載を追加。
- **Patching/Ablation**: 「因果的に必要なボトルネックである」という強い主張を避け、「候補位置である」「出力差を減少させた」「局在化の証明にはさらに解像度を上げた検証が必要」と修正。分布外介入（Out-of-Distribution）の可能性についても言及する。
- **Steering**: 現状の微小な変動値を踏まえ、「生成健全性を損なわず、滑らかかつ意図通りに制御できる」という主張を削除。効果が未確認であるか、あるいは結果が不安定であることを明記する。

#### 実装記述の修正
- `prompt_last_token` の算出方法について、単なる `inputs.input_ids.shape[1] - 1` ではなく、`attention_mask` から最後に1となる位置を取得している（または取得すべきである）旨を正確に記述する。
- データ件数（321件 vs 3×3×50件=450件）の不一致について補足説明を追加する。
- 81通りのJSON候補に対するシーケンス尤度の計算方法（固定prefix、終端規則など）について、公正な比較が行われた旨を付録または本文に追記する。

## Verification Plan

### 自動テスト (Automated Tests)
- なし（本タスクはMarkdownドキュメントの編集のみです）。

### 手動検証 (Manual Verification)
- 編集後の `paper_draft.md` を読み返し、構成が指定通りになっているか、主張がトーンダウンされているか、数式が正しくレンダリングされる形式になっているかを確認します。
- ユーザーによる最終レビュー（査読者目線でのチェック）。
