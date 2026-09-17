# Task: LLMの感情反応性・内部表現・モデルマージ・操作性に関する包括的考察と仮説整理

## 背景と目的
ユーザーより提示された多面的な問い・仮説・アイデア群について、現在進行中の研究（`v3/docs/paper.md`）の知見およびLLMのメカニスティック解釈可能性（Mechanistic Interpretability）に基づき、科学的かつ体系的に分析・整理を行う。

### 主な検討テーマ
1. **内部状態からの感情推定（Probing & Representation）**
   - LLMの内部状態から話し手の感情を推定できるか？
2. **事後学習（FT / RLHF）と感情表出・回路の変化**
   - FTで感情を激しく吐き出すように変えられるか？
   - FTで特定の回路が遮断・転移されるか？ それはModel Mergeに利用できるか？
3. **基礎概念の整理（Base vs Instruct）**
   - BaseとInstructの違いの再整理（どっちが感情を出すか？ なぜInstructは中立化するのか？）
   - 「感情的なロールプレイ（会話能）」と「モデル自身の情動反応（自己報告）」の乖離
4. **プロンプト視点と認知的分離（Recognition vs Reactivity）**
   - 「普通の人はどう感じるか」という三人称プロンプトで感情値はどう変化するか？
5. **感情の消去・抑制と極端刺激（Ablation & Jailbreak）**
   - 感情回路を抑えたニュートラルモデルは作れるか？
   - 極端な刺激で抑制を突き破れるか？ 感情と出力の重複はあるか？
   - なぜ出力抑制が入るのに、尤度空間では感情に依存した漏れが出るのか？
6. **動的相互作用とバイアス（Steering & Feedback Loop）**
   - LLMを怒らせる・キレさせることはできるか？
   - 感情を増幅・減少させると、相手の感情推定（Recognition）にも歪み（気分一致バイアス）が生じるか？
   - 小規模・低性能モデル（おバカモデル）の方が感情を出しやすいか？
7. **AIにとっての「嫌なこと」と「罰」**
   - AIにとっての罰とは何か？ 損失関数・RLHF・目的のコンフリクトの観点から考察。

## 成果物
- `docs/affective_mechanisms_and_hypotheses_exploration/task.md`: 本タスクの要件と論点整理
- `docs/affective_mechanisms_and_hypotheses_exploration/implementation_plan.md`: 考察の体系化方針と実験検証可能性の整理
- `docs/affective_mechanisms_and_hypotheses_exploration/walkthrough.md`: 各問いに対する詳細な科学的回答と将来の研究ロードマップ
