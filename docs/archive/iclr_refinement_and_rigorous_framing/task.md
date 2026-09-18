# タスク計画: ICLR 2027 基準に基づく paper.md の全面改訂と学術的精密化

## 目的
ユーザーから提示された ICLR 査読基準に基づく専門的フィードバックを全面的に反映し、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` を学術的・概念的・方法論的に厳密な論文ドラフトへと改訂する。

## 主要な改訂項目
1. **過度の断定表現の排除と謙虚かつ正確な表現への置換**:
   - 「完全再現」「世界初」「確定」等の表現を撤回し、制御された条件下の観察事実および整合的証拠として再定式化。
2. **擬人化用語の操作的・計算論的定義への置換**:
   - 「感情」「気分」「自己感情」を `affect-relevant activation state`, `first-person affect self-report distribution`, `valence-direction intervention`, `valence-congruent recognition shift`, `third-person affect recognition` 等に統一。
3. **小標本（n=10 pair）の適切な限定と統制されたケーススタディとしての位置づけ**:
   - 不当な一般化を抑え、Controlled mechanistic case study としての範囲、BCa bootstrap CI、pair-levelの散布を明記。
4. **因果介入とアライメントの要因計画（Factorial Design）の精密化**:
   - 抽出位置、Ridge写像の単位、within-model positive control、null resultの代替説明の診断。
5. **H3（一様抑制）とH4（分散的結合変化）の数理モデルの厳密化**:
   - 混合効果モデルの数式定義と標本単位（独立なペア）の明示。
6. **Sequence Likelihood Protocol の妥当性検証**:
   - テンプレート、tokenization、長さ正規化、および概念分離（Reader-rated text affect / Character attribution / First-person report likelihood）。
7. **文献引用の厳密化**:
   - 過剰な帰属を是正し、Measurement validation / Consistency checks として整理。
8. **N4（Valence-congruent recognition shift）の統制と交互作用モデル**:
   - 刺激の曖昧性と方向性（Valence vs Random/Arousal）の交互作用。
9. **Appendix D の法的意見書の削除と Ethics & Reproducibility Statement への置換**:
   - EU AI Actの過度な断定を削除し、国際査読標準の倫理声明および再現性声明に換装。
10. **ダブルブラインド（匿名性）の徹底**:
    - 特定組織名、絶対パス、Gitコミットハッシュの本文からの排除。

## タスクリスト
- [x] 改訂方針の整理とタスク策定（`task.md`）
- [x] 実装計画（`docs/iclr_refinement_and_rigorous_framing/implementation_plan.md`）の策定・保存
- [x] `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` の全面改訂
- [x] 変更確認（`docs/iclr_refinement_and_rigorous_framing/walkthrough.md`）の作成・保存
- [x] ユーザーへの詳細な日本語完了報告
