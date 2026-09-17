# paper2.md および paper3.md の改訂実装計画

## 背景と目的
4大モデルファミリー（Qwen, Llama, Mistral, Gemma）× 2条件（Base / Instruct）の計8モデルによる EmoBank（$N=321$）および AIPsy-Affect（$N=144$）の実測実験により、以下の決定的事実が判明した：
1. **感情認識の普遍性**: 複数ファミリーで人間同等の感情認識能力（EmoBank $r_V \approx 0.6 \sim 0.88$）が確認された。
2. **事後学習（Post-training）効果のモデルファミリー依存性**: 
   - 「Instruct化すると自己報告が一律に中立化する」という従来の前提は普遍的ではない。
   - Llama-Instruct では EmoBank Greedy 出力の 38.0% が中立 `(5, 5)` に収束する強い中立化傾向が見られる一方、Qwen, Mistral, Gemma では自己報告でも感情文脈に連動した出力を行う。
   - AIPsy-Affect では、Instruct 化によってむしろ感情文と中立文の弁別能（$\|\Delta_V\|$）が数倍〜9倍へと大幅に強化される。
3. **研究の再定義**:
   - **V1**: Recognition と Self-report の分離評価 + Post-training 効果のモデル間比較（`Post-training effect is model-family dependent`）。
   - **V2**: 単純な「表層抑制（Suppression）の解除」ではなく、「Post-training による内部感情表現から自己報告へのマッピング再編成（Representation-to-report remapping）」の機構解析。
   - **V3**: 自己報告の中立化の有無に依存せず、「Linear Probe で高精度に読める場所（Decodability）と因果的に出力を動かす場所（Causal Leverage）の時空間的解離（$\text{Decodability} \neq \text{Causal Leverage}$）」の検証。

本計画では、この強固な学術的フレームワークに基づいて `v3/docs/paper2.md` および `v3/docs/paper3.md` の記述を全面的に修正・同期する。

---

## 主な改訂箇所

### 1. 導入部・要約・問題設定の修正
- **旧記述**: 「Base では自己報告が出るが、Instruct では表層が 83.33% 完全中立化する。しかし内部には感情があるため、なぜ中立化するのかを解明する」
- **新記述**: 
  - 感情認識は Base / Instruct ともに高精度に獲得されている。
  - しかし Post-training の影響はモデルファミリーによって大きく異なり、自己報告の中立化（Llama）から感情感度の増幅（Qwen, Mistral）まで多様である。
  - したがって本研究は、「内部情動表現（Affective Representation）と表層自己報告（Self-Report）の対応関係が、事後学習によってどのように再編成（Remapping）されるのか」を解明する機構解析である。
  - 概念図の更新:
    $$\text{刺激文} \;\longrightarrow\; \underbrace{\text{Recognition (認知的理解: $r_V \approx 0.83$)}}_{\text{普遍的認識能}} \;\longrightarrow\; \underbrace{\text{Internal Probe (幾何学的復元: $R^2=0.561$)}}_{\text{中間層の強固な符号化}} \;\longrightarrow\; \underbrace{\text{Remapping (表現-報告再編成)}}_{\text{ファミリー依存の表層マッピング}} \;\longrightarrow\; \underbrace{\text{Intervention (因果レバレッジ)}}_{\text{Decodability} \neq \text{Causal Leverage}}$$

### 2. リサーチクエスチョン（RQ）の再定式化
- **RQ1**: 離散出力の挙動にかかわらず、内部の候補出力対数尤度分布（Sequence-Likelihood）に刺激依存的な情動構造がどのように保存されているか。
- **RQ2**: 事後学習によって感情認識と自己報告の対応関係はどのように再編成されるか（単一ボトルネック仮説 vs 分散的再マッピング仮説）。
- **RQ3 (V3の核心)**: 内部表現が線形プロービングで復元可能な層（Decodability）と、実際の出力尤度を因果的に左右する層（Causal Leverage）は一致するのか（時空間的解離の検証）。

### 3. 付録 C.1 / ベースライン評価節の更新
- Qwen の旧パース打ち切りアーティファクト（83.33%）の記述を削除。
- 4ファミリー × 2条件 = 計8モデルによる EmoBank（$N=321$）および AIPsy-Affect（$N=144$）の完全な実測値表を掲載。
- 「モデルファミリー × データセット × Base/Instruct の交互作用」として客観的に記述。

### 4. メカニズム解析（Phase 5: Distributed Remapping）の洗練
- activation patching で得られた「中間層の複数コンポーネントが累積的に寄与している」という実測結果を、「一律の中立化抑制」ではなく「事後学習による表現-報告マッピングの分散的再編成（Distributed Remapping）」の直接的証拠として位置づけ直す。

---

## 変更対象ファイル
- [MODIFY] [paper2.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md)
- [MODIFY] [paper3.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md)
- [NEW] `docs/cross_family_emobank_recognition/evaluation_report.md`（8モデル詳細レポートの永続保存）
- [MODIFY] `docs/cross_family_emobank_recognition/walkthrough.md`

## 検証方法
- 表記・数値（EmoBank $r$, AIPsy $\|\Delta_V\|$, Probe $R^2$, Patching Recovery 等）の整合性確認
