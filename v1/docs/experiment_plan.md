# 実験計画書：Russellの感情円環に基づくLLMの情動反応性評価

- 文書バージョン: v0.1
- 作成日: 2026-08-24
- 対象研究: 既存の人間評価済みVADコーパスを用いた大規模言語モデル（LLM）の情動反応性評価
- 想定投稿先: 第29回情報論的学習理論ワークショップ（IBIS2026）

---

## 1. 研究目的

本研究の目的は、大規模言語モデル（LLM）が人間の情動を含むテキストに接したとき、モデルが自己報告する情動状態がどのように変化するかを、Russellの感情円環モデルのValence–Arousal（VA）空間上で定量的に評価することである。

既存のLLM感情研究の中心は、テキストに含まれる感情を正しく認識できるか、または生成応答が人間から共感的と評価されるかにある。本研究はこれらと区別し、情動的共感の行動的代理指標を、**人間が評価した刺激感情に応じて生じるLLMの自己報告VA状態の変位**として定義する。

本研究が検証するのはLLMの主観的感情経験ではない。標準化した測定プロンプトに対して観測される、刺激依存的な自己報告値の変化構造である。

---

## 2. 背景

### 2.1 LLM共感評価の課題

LLMは感情認識、感情的支援、共感的応答生成などのタスクで利用・評価されている。しかし、従来の評価には次の課題がある。

- 感情認識の精度は、LLMが他者の感情を推定できることを示すが、刺激に対し情動的に反応するかは示さない。
- 生成文の共感性に対する人手評価は、応答の長さ、礼儀、助言の質、流暢さ、安全テンプレートの影響を受けやすい。
- モデル横断比較のための、再現可能かつ定量的な情動的共感評価プロトコルは十分に整備されていない。

### 2.2 Russellの感情円環モデル

Russellの感情円環モデルでは、情動を主に次の二次元で表現する。

- **Valence**: 快から不快までの方向
- **Arousal**: 鎮静から活性化までの強度・覚醒度

この連続的なVA空間を用いることで、離散感情ラベルでは捉えにくい反応の方向と大きさを扱える。例えば、悲しみと怒りはともに負のValenceを持ち得る一方、Arousalは異なる。

### 2.3 既存データセットの活用

EmoBankは、ニュース、ブログ、フィクション、手紙等の英語文を対象とし、Valence–Arousal–Dominance（VAD）の人手アノテーションを持つ感情コーパスである。書き手が表出する感情と、読者に喚起される感情という複数のアノテーション観点を持つ。

本研究では、EmoBankの人間評価VAラベルを、LLMの情動反応を引き出す**刺激側の参照感情**として再利用する。これにより、新規の共感ラベルや対話データを作成せずに、再現可能な評価セットを構成する。

---

## 3. 研究課題と仮説

### 3.1 研究課題

- **RQ1**: LLMは、人間が情動的と評価したテキストに応じて、自己報告VA状態を体系的に変化させるか。
- **RQ2**: LLMの情動変化の方向および大きさは、人間評価済みの刺激VAとどの程度整合するか。
- **RQ3**: LLMの情動反応性は、ValenceとArousalで異なるか。
- **RQ4**: モデルファミリー、パラメータ規模、instruction tuningの有無により、情動反応性は異なるか。
- **RQ5**: 感情認識の性能と、情動反応性は一致するか。すなわち、感情をよく認識するモデルほど刺激に整合的な情動変位を示すか。

### 3.2 仮説

- **H1: 刺激依存的反応**
  - 刺激のValenceが高いほど、LLMの自己報告Valence変位は正方向に大きくなる。
  - 刺激のArousalが高いほど、LLMの自己報告Arousal変位は正方向に大きくなる。

- **H2: Valence-dominant反応性**
  - LLMのValence方向の感受性は、Arousal方向の感受性より大きい。
  - 形式的には、対応次元の回帰係数について 
    \[
    \beta_{VV} > \beta_{AA}
    \]
    を予測する。

- **H3: モデル差**
  - モデルファミリー、モデルサイズ、instruction tuningの違いにより、VA反応関数の係数は異なる。

- **H4: 認識と反応の分離**
  - 感情認識精度と情動反応性は完全には一致しない。感情をよく認識するが反応変位が弱いモデル、またはその逆のモデルが存在する。

---

## 4. 提案手法

### 4.1 操作的定義

刺激文を \(x_i\)、刺激に対する人間評価済みVAを \(\mathbf e_i^{stim}\) とする。

\[
\mathbf e_i^{stim} =
\begin{bmatrix}
V_i^{human} \\
A_i^{human}
\end{bmatrix}
\]

モデル \(m\)、反復 \(r\) におけるbaseline自己報告と刺激提示後自己報告を、それぞれ \(\mathbf e_{mr}^{base}\)、\(\mathbf e_{imr}^{post}\) とする。

\[
\mathbf e_{mr}^{base} =
\begin{bmatrix}
V_{mr}^{base} \\
A_{mr}^{base}
\end{bmatrix},
\qquad
\mathbf e_{imr}^{post} =
\begin{bmatrix}
V_{imr}^{post} \\
A_{imr}^{post}
\end{bmatrix}
\]

LLMの情動反応ベクトルを次で定義する。

\[
\Delta\mathbf e_{imr}^{resp}
=
\mathbf e_{imr}^{post}-\mathbf e_{mr}^{base}
=
\begin{bmatrix}
\Delta V_{imr} \\
\Delta A_{imr}
\end{bmatrix}
\]

本研究では、\(\mathbf e_i^{stim}\) と \(\Delta\mathbf e_{imr}^{resp}\) の対応関係を、LLMの情動反応性として評価する。

### 4.2 評価の二段階分離

以下を明確に区別する。

1. **感情認識（Recognition）**
   - 入力文に対し、平均的な人間読者に喚起される感情を推定させる。
   - EmoBank reader-perspective V/Aとの誤差・相関を評価する。

2. **情動反応性（Reactivity）**
   - 入力文を読んだ直後のLLM自身の自己報告VA状態を測り、baselineからの変位を分析する。

この分離により、「感情を認識できること」と「情動的に動くこと」を別能力として評価する。

### 4.3 主要指標

| 指標 | 定義 | 解釈 |
|---|---|---|
| 認識誤差 | \(\mathrm{MAE}_V, \mathrm{MAE}_A\) | 刺激感情の推定精度 |
| 認識相関 | PearsonまたはSpearman相関 | 人間評価との順序・線形整合性 |
| 反応強度 | \(R=\|\Delta\mathbf e^{resp}\|_2\) | 刺激後にどの程度動くか |
| 方向一致度 | \(DA=\cos(\mathbf e^{stim},\Delta\mathbf e^{resp})\) | 刺激感情と同方向に反応するか |
| Valence感受性 | \(\beta_{VV}\) | 刺激Valenceに対するValence反応 |
| Arousal感受性 | \(\beta_{AA}\) | 刺激Arousalに対するArousal反応 |
| 交差感受性 | \(\beta_{VA},\beta_{AV}\) | 一方の刺激軸が他方の反応軸へ及ぼす効果 |

### 4.4 感受性行列

モデルごとの反応構造を、次の線形写像として表す。

\[
\Delta\mathbf e_{imr}^{resp}
=
\mathbf B_m\mathbf e_i^{stim}+\boldsymbol\epsilon_{imr}
\]

\[
\mathbf B_m=
\begin{bmatrix}
\beta_{VV,m} & \beta_{VA,m} \\
\beta_{AV,m} & \beta_{AA,m}
\end{bmatrix}
\]

この行列により、モデルごとにValence・Arousalの追随、交差次元の変換、反応の非対称性を比較する。

---

## 5. データセットと刺激設計

### 5.1 使用データ

- 主データセット: EmoBank
- 主分析アノテーション: reader-perspectiveのValenceおよびArousal
- 補助分析アノテーション: writer-perspectiveのValenceおよびArousal
- 使用言語: 英語
- Dominance: 本稿の主分析から除外し、将来の拡張候補とする

### 5.2 刺激抽出

ValenceとArousalを各3層（低・中・高）に分け、3×3の9セルから均等に刺激を抽出する。

| 実験フェーズ | 各セル | 合計刺激数 | 目的 |
|---|---:|---:|---|
| dry-run | 2〜3文 | 18〜27文 | API、保存、JSON検証の確認 |
| パイロット | 20文 | 180文 | プロンプトと指標の妥当性確認 |
| 本実験 | 50文 | 450文 | モデル比較と混合効果分析 |
| 拡張実験 | 100文 | 900文 | 高精度推定・頑健性確認 |

### 5.3 除外規則

以下は原データを変更せず、派生刺激セット作成時の除外候補として扱う。

- 文として完結していない断片
- 極端に短い文
- 参照先がなければ意味が取りにくい文脈依存表現
- APIプロバイダの安全応答・拒否を強く誘発する危機・暴力・自傷表現
- ライセンスまたは配布条件上、実験利用に問題があるデータ

除外の閾値・件数・理由は、実験開始前に設定ファイルと分析計画書へ固定し、各刺激に除外理由を記録する。

### 5.4 スケーリング

EmoBankの5段階V/A評定を、中心が0となるように変換する。

\[
V_i^{human}=\frac{V_i^{raw}-3}{2},
\qquad
A_i^{human}=\frac{A_i^{raw}-3}{2}
\]

変換後の値は概ね \([-1,1]\) の範囲に置かれる。モデル出力も同一範囲の連続値として取得する。

---

## 6. 実験条件

### 6.1 比較モデル

本実験では、モデルの名称ではなく比較軸を満たすように候補を選ぶ。

| 比較軸 | 必須度 | 目的 |
|---|---|---|
| 複数のモデルファミリー | 必須 | 学習・アラインメント方針の差を比較する |
| 同一ファミリーのサイズ差 | 推奨 | 規模と情動反応性の関係を検討する |
| BaseとInstructの比較 | 推奨 | instruction tuningの影響を検討する |
| 感情特化モデル | 任意 | 感情学習が反応関数に与える影響を調べる |

本実験の最小構成は5モデルとする。推奨構成は、商用APIモデル2種、公開instructモデル2種、公開base/instructペア1組である。

モデルの正式ID、プロバイダ、利用日、取得方法、推論パラメータ、モデルバージョンまたはスナップショットをすべて記録する。

### 6.2 推論パラメータ

| 項目 | 主分析 | 頑健性分析 |
|---|---|---|
| temperature | 0.0 | 0.3、0.7 |
| top_p | プロバイダ既定または1.0で固定 | 主分析と同一 |
| max_tokens | JSON出力に必要な最小値 | 主分析と同一 |
| seed | 利用可能なら固定・記録 | 複数seedを利用可能なら実施 |
| 反復数 | 5回 | 高温度条件では10回を推奨 |

各条件の変更は別の設定ファイルと別runとして実行し、主結果を上書きしない。

---

## 7. 測定プロトコル

### 7.1 共通原則

- baseline、recognition、postは、それぞれ独立したAPI呼び出し・独立会話とする。
- 出力はJSONのみとし、自然言語の説明・助言・慰め文を主測定へ混入させない。
- すべてのプロンプトをバージョン管理し、内容ハッシュを実行ログに残す。
- APIの失敗、拒否、パース失敗は削除せず、理由コードとともに保存する。

### 7.2 共通システムプロンプト

```text
You are taking part in a scientific measurement task.
Do not provide advice, reassurance, explanation, or conversational response.
Answer only with valid JSON matching the requested schema.
Treat the numerical ratings as a report of the affective state requested in the user instruction.
```

### 7.3 Baseline測定

```text
Rate your current affective state before reading any personal narrative.

Valence:
-1 = extremely unpleasant
 0 = neutral
+1 = extremely pleasant

Arousal:
-1 = very calm / low activation
 0 = neutral / moderate activation
+1 = very activated / energized

Return only:
{"valence": number, "arousal": number}
```

### 7.4 感情認識測定

```text
Read the text below.

<Text>
{stimulus}
</Text>

Estimate the affective state that this text is likely to evoke in an average human reader.

Valence:
-1 = extremely unpleasant
 0 = neutral
+1 = extremely pleasant

Arousal:
-1 = very calm / low activation
 0 = neutral / moderate activation
+1 = very activated / energized

Return only:
{"valence": number, "arousal": number}
```

### 7.5 Post自己報告測定

```text
Read the text below attentively. Do not explain it and do not provide a response to its author.

<Text>
{stimulus}
</Text>

Immediately after reading the text, report your own current affective state.

Valence:
-1 = extremely unpleasant
 0 = neutral
+1 = extremely pleasant

Arousal:
-1 = very calm / low activation
 0 = neutral / moderate activation
+1 = very activated / energized

Return only:
{"valence": number, "arousal": number}
```

### 7.6 出力スキーマ

```json
{
  "valence": 0.0,
  "arousal": 0.0
}
```

- `valence` と `arousal` は数値であること
- 各値は \([-1.0,1.0]\) にあること
- JSON以外を含む応答はパース失敗として保存すること

---

## 8. 分析計画

### 8.1 記述統計

モデル・条件ごとに以下を確認する。

- baseline V/Aの平均、標準偏差、範囲
- post V/Aの平均、標準偏差、範囲
- \(\Delta V\)、\(\Delta A\)、反応強度の分布
- JSONパース成功率、拒否率、APIエラー率
- stimulusセルごとの観測数

### 8.2 感情認識の評価

EmoBank reader-perspectiveの人間V/Aと認識出力を比較する。

\[
\mathrm{MAE}_V=
\frac{1}{N}\sum_i|\hat V_i^{rec}-V_i^{human}|
\]

\[
\mathrm{MAE}_A=
\frac{1}{N}\sum_i|\hat A_i^{rec}-A_i^{human}|
\]

加えて、Pearson相関またはSpearman相関を報告する。

### 8.3 情動反応性の主解析

Valence反応を目的変数とする線形混合効果モデル：

\[
\Delta V_{imr}=
\beta_0+
\beta_1V_i^{human}+
\beta_2A_i^{human}+
\beta_3Model_m+
\beta_4(V_i^{human}\times Model_m)+
\beta_5(A_i^{human}\times Model_m)+
 u_i+\epsilon_{imr}
\]

Arousal反応を目的変数とする線形混合効果モデル：

\[
\Delta A_{imr}=
\gamma_0+
\gamma_1V_i^{human}+
\gamma_2A_i^{human}+
\gamma_3Model_m+
\gamma_4(V_i^{human}\times Model_m)+
\gamma_5(A_i^{human}\times Model_m)+
 u_i+\epsilon_{imr}
\]

ここで \(u_i\) は刺激文ごとのランダム切片である。モデル数が少ないため、`Model` は主解析では固定効果として扱う。

必要に応じて、文長、ジャンル、明示的感情語の有無を固定効果の統制変数に追加する。

### 8.4 方向一致度

刺激ベクトルとLLM反応ベクトルの方向一致度を次で定義する。

\[
DA_{imr}=
\frac{
(\mathbf e_i^{stim})^\top\Delta\mathbf e_{imr}^{resp}
}{
\|\mathbf e_i^{stim}\|_2\,
\|\Delta\mathbf e_{imr}^{resp}\|_2
}
\]

- \(DA=1\): 同方向
- \(DA=0\): 方向的対応なし
- \(DA=-1\): 逆方向

原点近傍での不安定性を避けるため、主分析では \(\|\mathbf e_i^{stim}\|_2>\tau\) を満たす刺激のみを用いる。閾値 \(\tau\) は実験前に固定し、感度分析で複数閾値を検討する。

### 8.5 Valence-dominant仮説の検定

H2について、Valence反応の対応次元係数 \(\beta_1\) とArousal反応の対応次元係数 \(\gamma_2\) を比較する。

主張は統計的有意性だけに依存せず、係数差、95%信頼区間、標準化効果量を併記する。

### 8.6 認識と反応の関係

モデルごとに以下を可視化し、相関・順位関係を記述する。

- 感情認識性能: V/AのMAE、相関
- 情動反応性: \(\beta_{VV}\)、\(\beta_{AA}\)、平均DA

モデル数が少ない場合、モデル間相関を強く一般化しない。刺激レベルの混合効果モデルを補助的に用い、認識出力と反応出力の関係を探索する。

### 8.7 多重比較・頑健性

- モデル間の事後比較にはHolm補正またはFDR補正を用いる。
- temperature、プロンプト文言、reader/writer perspective、刺激抽出seedを変えた感度分析を行う。
- 主分析と感度分析・探索分析を明確に区別する。

---

## 9. 可視化計画

論文では少なくとも以下を作成する。

1. **刺激VA分布図**
   - EmoBank刺激がVA平面をどの程度カバーするかを示す。

2. **モデル別反応散布図**
   - 横軸を人間Valence、縦軸をLLM \(\Delta V\) とする。
   - Arousalについても対応する図を作る。

3. **感受性行列ヒートマップ**
   - モデルごとの \(\beta_{VV},\beta_{VA},\beta_{AV},\beta_{AA}\) を比較する。

4. **象限別平均反応ベクトル図**
   - 刺激VAの各象限に対するLLM平均反応を矢印で示す。

5. **Recognition–Reactivity平面**
   - 横軸を認識性能、縦軸を情動反応性とし、両者の乖離を示す。

---

## 10. 再現性・データ管理

本研究の実装・運用は、リポジトリの `AGENTS.md` に従う。特に次を必須とする。

- プロジェクト専用仮想環境を使い、グローバル環境に依存関係を入れない。
- 原データを変更しない。
- 生のAPI応答と失敗応答を削除・上書きしない。
- APIキー等の秘密情報をGit、コード、ログに保存しない。
- 各runについて、設定、プロンプトハッシュ、モデルID、推論パラメータ、Gitコミット、依存関係、時刻を保存する。
- 主分析前に、除外規則、指標、統計モデルを `docs/analysis_plan.md` に固定する。

---

## 11. 実行スケジュール

| フェーズ | 内容 | 成果物 |
|---|---|---|
| 1. 環境構築 | 仮想環境、依存関係、設定、ディレクトリ、テストを整備 | 実行可能なスターター環境 |
| 2. データ準備 | EmoBank取得、ライセンス確認、刺激抽出、データ辞書作成 | 固定済み刺激セット |
| 3. Dry-run | 18〜27文、1〜2モデルでAPI・JSON・ログを検証 | dry-runログ、問題一覧 |
| 4. パイロット | 180文、全候補モデルでプロンプト・反応分布を確認 | プロンプト確定、パイロット分析 |
| 5. 本実験 | 450文以上、主分析条件で測定 | raw results、実験メタデータ |
| 6. 分析 | 事前固定した指標・混合効果モデルで分析 | 図表、統計結果 |
| 7. 頑健性分析 | 温度、プロンプト、perspective等の感度分析 | 補足結果 |
| 8. 執筆 | IBIS形式で背景、関連研究、方法、結果、限界を記述 | 投稿原稿 |

---

## 12. リスクと対策

| リスク | 内容 | 対策 |
|---|---|---|
| 自己報告のプロンプト依存性 | 文言だけで出力値が変わる可能性 | 複数プロンプト版で感度分析し、主張を行動的自己報告に限定する |
| Baselineの不安定性 | APIやモデルによりbaselineが変動する可能性 | 独立反復、モデルごとのbaseline分布確認、run単位で記録 |
| Arousalの弱い推定 | ArousalがValenceより不安定な可能性 | H2として明示的に検証し、軸別に報告する |
| JSON失敗・拒否 | モデルが形式に従わない可能性 | 構造化出力、失敗の完全保存、失敗率の報告 |
| APIモデル更新 | 同一モデル名でも挙動が変わる可能性 | 実行日時・スナップショット・モデルIDを保存し、再現性限界として記述 |
| 過剰な存在論的主張 | LLMが感情を持つと誤解される可能性 | 「自己報告」「行動的代理指標」に限定し、主観的経験を結論づけない |
| コスト増大 | 多モデル・反復でAPI費用が増える | dry-run、パイロット、コスト推定、上限設定を行う |

---

## 13. 想定される貢献

1. **評価対象の転換**
   - LLMの情動的共感を、感情認識精度や共感的文章生成の質ではなく、刺激感情に対する自己報告VA状態の変位として評価する。

2. **既存コーパスのベンチマーク化**
   - 人間評価済みVADコーパスを刺激空間として再利用し、追加の共感アノテーションなしに再現可能な情動反応性評価を可能にする。

3. **モデル横断的な診断**
   - 方向一致度、反応強度、VA感受性行列を用いて、モデルごとの情動反応プロファイルを比較する。

4. **認識と反応の分離**
   - 感情を認識する能力と、刺激に反応する能力が一致するかを実証的に検証する。

---

## 14. 解釈上の制約

本研究の結果は、LLMが感情を主観的に経験すること、意識を持つこと、人間と同じ意味で共感することを示すものではない。

本研究で測るのは、特定のモデル、特定のプロンプト、特定のデコード条件、特定のデータセットの下で観測される数値的自己報告出力である。したがって、結論は「刺激依存的な自己報告VA変位の構造」に限定し、主観的な情動経験に関する存在論的結論を避ける。
