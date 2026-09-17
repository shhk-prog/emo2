# タスクリスト: ICLR向け新規性位置づけの厳密化と論文改訂 (ICLR Novelty Alignment & Refinement)

- [x] 1. 新規性・査読批判の分析と論点整理 <!-- id: 0 -->
    - [x] 外部査読・新規性判定テキストの精査（N1〜N4の強み・リスク、先行研究6系統、4つの貢献文） <!-- id: 1 -->
    - [x] 現行の `v3/docs/paper.md` における記述乖離と過剰主張リスク箇所の特定 <!-- id: 2 -->
- [x] 2. 論文原稿 (`v3/docs/paper.md`) のナラティブ・フレームワーク改訂 <!-- id: 3 -->
    - [x] Abstract & Introduction: 4つの公式貢献文と「Decodability $\neq$ Causal Substitutability」を核とするナラティブへの刷新 <!-- id: 4 -->
    - [x] Section 2 (仮説設定): H4 (Distributed Remapping) を「単一部位・出力層単独説の体系的反証」として慎重化 <!-- id: 5 -->
    - [x] Section 4 (結果): N1〜N4の提示表現を厳密化（N2を主役化、N1を測定論的貢献、N3を控えめな否定証拠、N4を三要因交互作用に純化） <!-- id: 6 -->
    - [x] Section 5 (関連研究): 提示された6系統の最前線研究との直接的・詳細な差分化（Table 6の拡張） <!-- id: 7 -->
    - [x] Section 6 (考察・限界): サンプル規模 ($n=10$) の限界と今後の拡張ロードマップの明記 <!-- id: 8 -->
- [x] 3. 採択可能性向上のための実験ロードマップの策定 <!-- id: 9 -->
    - [x] AIPsy-Affect $n=50\sim 100$ ペアへの拡張計画の策定 <!-- id: 10 -->
    - [x] Multi-layer patching / Causal tracing / Mahalanobis OOD診断の技術設計 <!-- id: 11 -->
- [x] 4. 改訂内容の検証と成果物まとめ (`walkthrough.md`) <!-- id: 12 -->
    - [x] 変更箇所の整合性確認 <!-- id: 13 -->
    - [x] `walkthrough.md` の作成と作業ログ保存 <!-- id: 14 -->
