# EmoBank 3-Way VAD 考察・分析計画書 (Implementation Plan)

## 概要
EmoBank 実験結果（全8モデル、3タスク × 3次元 ＋ 内部結合度）の数値に基づき、感情認識・自己報告・認知的結合のメカニズムを定量的エビデンスに基づいて体系的に考察する。

## 考察フレームワークと構成

### 1. タスク別・次元別の基本性能考察（① Writer, ② Reader, ③ Self, ④ Coupling）
- **Valence（快-不快）**: 最も高く一貫したアライメント（$r \approx 0.45 \sim 0.66$）。
- **Arousal（覚醒度）**: 中程度の相関（$r \approx 0.10 \sim 0.35$）。テキスト表層からの情動強度の抽出難度。
- **Dominance（優位性）**: 最も相関が低く脆弱（$r \approx 0.00 \sim 0.25$）。特に事後学習による崩壊傾向。
- **Self vs. Reader/Writer**: なぜ自己報告（$S$）の方が人間読者（$R$）の正解ラベルとの相関が高いのか。

### 2. Base vs. Instruct の比較分析
- 指示追従学習（SFT / RLHF）が与える正の影響（出力フォーマット遵守、小型モデルでの認識能力の開花）。
- 負の副作用（安全化・中立化バイアスによる Arousal の低位偏向、キャリブレーション誤差 MAE の悪化、Dominance 表現の喪失）。

### 3. モデルファミリー・パラメータ規模の比較分析
- **Mistral 7B**: 最高峰の潜在表現力と、Base での貪欲出力中立退化（Collapse）、Instruct での低覚醒バイアス。
- **Qwen 2.5 1.5B**: 最も頑健でバランスの取れた性能（Base/Instructともに高水準、崩壊なし）。
- **Llama 3.2 1B & Gemma 2 2B**: Base での脆弱性と Instruct による劇的な飛躍。

### 4. 5大リサーチクエスチョン（RQ1〜RQ5）への学術的回答
- **RQ1: 人間情動状態の推論精度（Alignment vs Calibration）**
- **RQ2: 自己報告（Self）の人間対応性と系統的バイアス（Hypo-arousal, Positivity/Neutrality Shift）**
- **RQ3: 事前学習 vs 事後学習の影響（表現の覚醒 vs 抑制バイアス）**
- **RQ4: モデルファミリーと規模の差異**
- **RQ5: 他者認識（Reader）と自己報告（Self）の内部結合度（認知的鏡像仮説 Mirroring Hypothesis）**
