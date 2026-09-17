# タスク計画: paper.md の大幅拡充（v1/v2/v3統合・既存研究対比・AIガバナンス適合補遺）

## 目的
`/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` を大幅に加筆・拡充し、以下の要素を包括的かつ厳密に組み込む：
1. **v1, v2, v3 の全実験内容と結果の統合**:
   - v1: Sequence Likelihood Protocolの数理、EmoBankとAIPsy-Affect、スケーリング則（0.5B〜7B）、認識と自己報告の解離
   - v2: 4仮説（H1〜H4）、幾何変換（Ridge Alignment）、Aligned Cross-Model Patching、全層スイープ、混合効果モデル、8条件重みスワップ
   - v3: 厳密3分割検証、Dual-Outcome評価（自己報告 $E[V]$ と共感行動選択 $B(x)$）、気分一致バイアス（Mood Congruency Bias）の因果実証
2. **既存研究との詳細な対比（Related Work & Discussion）**:
   - プロービング研究（Decodability $\neq$ Causal Substitutability）
   - 行動評価（尤度空間での逆転現象）
   - アライメント機序（Distributed Remapping）
   - ステアリング研究（認知的判断の歪み・気分一致効果）
3. **Appendix への AIガバナンス・EU AI Act 非該当性論証の追加**:
   - 科学研究開発免除（Article 2(6), 2(8)）
   - 禁止される感情認識（Article 5(1)(f)）への非該当性（対象は生身の人間ではなくLLM自身の隠れ状態）
   - ハイリスク分類（Annex III）およびGPAI義務（Article 51-55）との関係
   - 個人情報非保持と倫理原則（AGENTS.md準拠）

## タスクリスト
- [x] 実装計画（`docs/paper_draft_enhancement_and_governance_integration/implementation_plan.md`）の策定・保存
- [x] `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` の改訂・大幅加筆
- [x] 変更確認（`docs/paper_draft_enhancement_and_governance_integration/walkthrough.md`）の作成・保存
- [x] ユーザーへの詳細な日本語完了報告
