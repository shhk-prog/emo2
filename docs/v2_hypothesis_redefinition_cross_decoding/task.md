# タスク: v2におけるH1/H2仮説の再定式化（Complete ErasureからRepresentational Replacementへ）

## 目的
ユーザーからの極めて本質的な指摘に基づき、v1とv2の接続ロジックおよびCross-Decodingの理論的意義を再整理・再定式化する。

### 背景
- v1において、Instructモデル内部にも感情情報が存在すること（Affective vs Neutral の高精度デコード、$I(\text{Instruct};\text{affect}) > 0$）は既に実証されている。
- 従来の「H1 (Complete Erasure: 感情情報が内部から完全に消去された)」は、v1の結果を前提にすると藁人形（strawman）仮説となってしまっていた。
- v2のCross-Decodingが真に問うべきは、「Baseモデルで形成されていた感情表現が失われ別の表現へ置換されたのか（H1）、それとも変換可能な形で保持されているのか（H2）」である。

## 作業項目
- [x] v1 $\to$ v2 の論理的接続の明確化（Cross-Decodingの必然性）
- [x] H1/H2仮説の再定義と対比構造の整理
- [x] `docs/v2_hypothesis_redefinition_cross_decoding/` 配下に `task.md`, `implementation_plan.md`, `walkthrough.md` を作成
- [x] ユーザーへの詳細な整理・フィードバックの提示
