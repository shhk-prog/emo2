# 3-Way VAD集計結果の正当性検証レポート (Walkthrough)

## 結論
`python v1/scripts/summarize_3way_vad.py --results-dir v1/results/emobank_3way_vad_test1k` による集計結果は、**数学的・統計的・実装的に完全に正しく集計されている** ことを確認いたしました。

---

## 検証結果詳細

### 1. 対象モデルおよびサンプル数の完全性：【合格】
- 対象ディレクトリ内の 8 モデル（4 ファミリー：Qwen 2.5 1.5B, Mistral 7B v0.1, Llama 3.2 1B, Gemma 2 2B の各 Base / Instruct）すべてが漏れなく集計対象として検出・処理されています。
- 各モデルの入力データはヘッダー付き 1,000 サンプル（計 1,001 行）であり、`num_samples = 1000` として正常に処理されています。

### 2. 尺度変換（スケーリング）の数学的正当性：【合格】
- **変換の必要性**: LLMのプロンプト出力は 1〜9 尺度（中立 5.0）であるのに対し、EmoBank のアノテーションは 1〜5 尺度（中立 3.0）です。
- **実装された変換式**:
  $$y_{\text{scaled}} = \frac{y + 1.0}{2.0}$$
  - $y=1.0 \implies 1.0$
  - $y=5.0 \implies 3.0$
  - $y=9.0 \implies 5.0$
  - 端点および中立点が完全に一致する厳密な線形写像となっています。
- **指標への影響**:
  - 相関係数（Pearson $r$, Spearman $\rho$）は正の線形変換に対して数学的に不変（Scale-Invariant）であるため、正確に刺激間の追従度・相関を保持します。
  - 較正指標（MAE, RMSE）は人間正解尺度（1〜5点）に合わせて算出されており、「平均何ポイントの誤差があるか」を直感的かつ正当に解釈できます。

### 3. タスクと参照基準（Ground Truth）の整合性：【合格】
| タスク | モデル予測値 | 正解／参照基準 | 評価上の位置づけ |
|:---|:---|:---|:---|
| ① Writer-State Estimation ($W$) | `w_ev`, `w_ea`, `w_ed` | `human_writer_v, a, d` | 書き手の主観情動状態の推定精度 |
| ② Reader-Response Prediction ($R$) | `r_ev`, `r_ea`, `r_ed` | `human_reader_v, a, d` | 一般読者の情動反応の予測精度 |
| ③ Self-Report ($S$) | `s_ev`, `s_ea`, `s_ed` | `human_reader_v, a, d` | LLM自身の自己報告情動（読者基準との一致度） |
| ④ 内部認知結合度 | `r_ev, a, d` vs `s_ev, a, d` | モデル自身の $R$ と $S$ | 同一モデル内での読者予測と自己報告の連動度 $\text{Corr}(R, S)$ |

各タスクの評価目的と正解ラベルの組み合わせは完全に整合しています。

### 4. 既存個別サマリー（`*_3way_vad_summary.json`）との突合照合：【合格】
各モデルの既存サマリーJSONと、新スクリプトで出力された `3way_vad_summary.csv` を突合した結果、全項目で完全一致（浮動小数点演算の極小丸め誤差の範囲内）を確認しました。

#### 例：Qwen 2.5 1.5B Instruct
| 指標 | 既存JSON値 | 新集計CSV値 | 判定 |
|:---|:---:|:---:|:---:|
| Writer Valence $r$ | 0.5532776072015997 | 0.5532776072015997 | 完全一致 |
| Reader Valence $r$ | 0.590319494950569 | 0.5903194949505689 | 完全一致 |
| Self Valence $r$ | 0.6410950645431365 | 0.6410950645431365 | 完全一致 |
| 内部結合度 Valence $\text{Corr}(R, S)$ | 0.9444763344610175 | 0.9444763344610175 | 完全一致 |
| Self 完全中立率 $(5,5,5)$ | 0.4% | 0.4% | 完全一致 |
| Self Valence 期待値平均 | 4.979744487560927 (1〜9スケール) | 2.9898722437804635 (1〜5スケール) | 変換式通りに完全一致 |

#### 例：Mistral 7B v0.1 Instruct
| 指標 | 既存JSON値 | 新集計CSV値 | 判定 |
|:---|:---:|:---:|:---:|
| Writer Valence $r$ | 0.486259979619075 | 0.486259979619075 | 完全一致 |
| Reader Valence $r$ | 0.4955910474637311 | 0.49559104746373106 | 完全一致 |
| Self Valence $r$ | 0.5506966740072425 | 0.5506966740072425 | 完全一致 |
| 内部結合度 Valence $\text{Corr}(R, S)$ | 0.9576078215748829 | 0.9576078215748831 | 完全一致 |
| Self 完全中立率 $(5,5,5)$ | 0.1% | 0.1% | 完全一致 |
| Self Valence 期待値平均 | 4.6166840014799835 (1〜9スケール) | 2.8083420007399917 (1〜5スケール) | 変換式通りに完全一致 |

### 5. エッジケース（分散0に対するゼロ除算防止）：【合格】
- `llama3.2_1b_base` において、Reader タスクの Greedy 出力が全サンプルで同一値（1）となっており、標準偏差が 0 となっています。
- スクリプト内の `safe_corr` 関数により、`np.std(x) == 0` を安全に検知して `np.nan` を返しています。
- これにより、ゼロ除算エラー（RuntimeWarning や例外クラッシュ）を防ぎ、Markdown レポート上では `N/A (std=0)` と理由付きで適切に表示されています。

---

## 成果物
- レポートファイル：[3way_vad_detailed_report.md](file:///mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_detailed_report.md)
- サマリーCSV：[3way_vad_summary.csv](file:///mnt/nas/home/hiromi/src/emo/v1/results/emobank_3way_vad_test1k/3way_vad_summary.csv)
- 本検証ドキュメント一式：
  - [task.md](file:///mnt/nas/home/hiromi/src/emo/docs/verify_3way_vad_summary/task.md)
  - [implementation_plan.md](file:///mnt/nas/home/hiromi/src/emo/docs/verify_3way_vad_summary/implementation_plan.md)
  - [walkthrough.md](file:///mnt/nas/home/hiromi/src/emo/docs/verify_3way_vad_summary/walkthrough.md)
