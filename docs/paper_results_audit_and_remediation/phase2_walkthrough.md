# V2/V3 生成ロジック・結果表・LaTeX 監査および修正完了確認レポート (Walkthrough)

## 概要

本ドキュメントは、ユーザーからの監査指摘事項（V2/V3 の生成ロジックの不一致、ハードコード残存、判定基準の Methods 逆転、LaTeX コンパイルエラー等）全15項目に対する修正内容と検証エビデンスをまとめた完了確認レポートである。

---

## 修正項目と対応状況一覧

| # | 指摘項目 | 修正内容 | 対応ファイル / エビデンス | 判定 |
|---|---|---|---|:---:|
| 1 | **V2 recovery table の stale 値および CSV トークン整合** | `table_v2_4_recovery.csv` を実測値（Gemma, OLMo）および NaN（未実施の Llama, Qwen）で再作成し、空フィールドのカンマ数（8列定義）を完全整合。旧ダミー値（0.720, 0.450 等）を完全消去した上で `v2_distribution_recovery.tex` を再生成。 | `results/derived/paper_summary/tables/table_v2_4_recovery.csv`<br>`iclr2027/tables/v2_distribution_recovery.tex` | **RESOLVED** |
| 2 | **V2 H1/H2 hard-coded fallback 削除** | `scripts/summarize_v2_reorganization.py` の `generate_h1_h2_table()` 内の固定タプル（`"0.353"`, `"[0.330, 0.370]"` 等）を全廃。データ不在時は例外送出または `---` 表示へ変更。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_h1_h2_reorganization.tex` | **RESOLVED** |
| 3 | **V2 confirmatory H3/H4 の追加** | `CONF_MAP` に H3（Alignment 主効果、Depth/Task 交互作用項）および H4（Matched-Plain 回復量、Self-Reader 差）を追加。Caption（H1--H4）と表内容を完全一致。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_confirmatory_summary.tex` | **RESOLVED** |
| 4 | **V2 q値の一律 `< 0.001` ハードコード削除** | 全行一律 `< 0.001` 付与を廃止。LMM の実測 $p$ 値を反映し、検定非実施項目は `---` に統一。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_confirmatory_summary.tex` | **RESOLVED** |
| 5 | **V2 causal center を真の重心（Center-of-Mass）へ修正** | `d_peak` の流用をやめ、Methods 定義 $\bar{d}_C = \frac{\sum_l d_l \max(C(l),0)}{\sum_l \max(C(l),0)}$ に基づく計算を実装。 | `v2/scripts/build_paper_summary.py` | **RESOLVED** |
| 6 | **V2 causal controls 表への Task 列追加** | テーブルヘッダーを `Family \| Task \| Condition \| Axis` に改修。Reader と Self の行を分離し、行の重複感を解消。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_causal_controls.tex` | **RESOLVED** |
| 7 | **V2 `C_net_rand > 0 = 有意` Note の修正** | Note を「$C_{\text{net,rand}}>0$ は平均介入効果がランダム統制より大きい方向にあることを示すのみであり、有意性は CI および推測統計から判断する」旨に修正。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_causal_controls.tex` | **RESOLVED** |
| 8 | **V2 LMM caption を $C_{\text{net,rand}}$ 目的変数へ修正** | ピーク相対深度ではなく、実モデルの response variable である $C_{\text{net,rand}}$ を明記した Caption に改訂。 | `scripts/summarize_v2_reorganization.py`<br>`iclr2027/tables/v2_h3_causal_lmm.tex` | **RESOLVED** |
| 9 | **V3 H1 符号判定を Methods 通り正方向へ修正** | Methods 定義 $\Delta d^* = d_C^* - d_D^* > 0$ かつ $\Delta \bar{d} > 0$ に基づき、判定ロジックを $\text{CI}_{\text{low}} > 0 \land \text{estimate} > 0$ に修正。負の乖離（$-0.867$ 等）を正しく FAIL と判定。 | `table_v3_4_confirmatory.csv`<br>`table_v3_confirmatory_matrix.csv`<br>`v3/scripts/build_paper_summary.py`<br>`scripts/summarize_v3_causal_utilization.py` | **RESOLVED** |
| 10 | **V3 H1 に center criterion を追加** | `iclr2027/tables/v3_confirmatory_details.tex` に Peak と Center の両行を出力し、両方の信頼区間下限が正であることを Joint Criterion として要求。 | `scripts/summarize_v3_causal_utilization.py`<br>`iclr2027/tables/v3_confirmatory_details.tex` | **RESOLVED** |
| 11 | **V3 H3 を `M > 0 AND M_net > 0` へ修正** | ヒューリスティック `0.05` を全廃し、Methods 通り $\text{CI}_{\text{low}}(M) > 0 \land \text{CI}_{\text{low}}(M_{\text{net}}) > 0$ で厳密判定。 | `scripts/summarize_v3_causal_utilization.py`<br>`iclr2027/tables/v3_confirmatory_details.tex` | **RESOLVED** |
| 12 | **V3 H4 を `CI_low > 0` へ修正** | ヒューリスティック `0.05` を全廃し、Methods 通り $\text{CI}_{\text{low}}(\Delta C^{\text{temporal}}) > 0$ で判定。全3モデルで二軸達成（PASS）。 | `scripts/summarize_v3_causal_utilization.py`<br>`iclr2027/tables/v3_confirmatory_details.tex` | **RESOLVED** |
| 13 | **V3 mediated attenuation のハードコード値削除** | `m_rand = "0.0000"` や $T, R$ の逆算を全廃。`table_v3_3_mediated_attenuation.csv` から実測値（$T, R, M, M_{\text{rand}}, M_{\text{net}}$, CI）を直接読み込み・出力。 | `results/derived/paper_summary/tables/table_v3_3_mediated_attenuation.csv`<br>`scripts/summarize_v3_causal_utilization.py`<br>`iclr2027/tables/v3_mediated_attenuation.tex` | **RESOLVED** |
| 14 | **V3 canonical builder と presentation summarizer の一致** | `v3/scripts/build_paper_summary.py` を Valence/Arousal 両軸化、ファミリー固有 CI 抽出、厳密基準判定に改修し、二者の判定ロジックを完全一致化。 | `v3/scripts/build_paper_summary.py`<br>`scripts/summarize_v3_causal_utilization.py` | **RESOLVED** |
| 15 | **paper_summary CSV + manifest の整備** | `results/derived/paper_summary/tables/` に全25ファイルの正本 CSV を配置。データ出自の追跡可能性を確保。 | `results/derived/paper_summary/tables/*.csv`<br>`results/derived/paper_summary/paper_summary_manifest.json` | **RESOLVED** |
| 16 | **LaTeX コンパイルエラーの解消** | display math 内の空行（Natural Direct/Indirect Effect）統合、数式モード内での `\texttt{candidate\_start}` を `t_{\mathrm{start}}` に修正、`\multirow{10}{*}` 整合、欠落テーブル（`v2_causal_relocation.tex`, `v3_gate_decision.tex`, `v3_spatiotemporal_dynamics.tex`）の作成配置。 | `iclr2027/iclr2027_conference2.tex`<br>`iclr2027/tables/*.tex` | **RESOLVED** |

---

## 結論

指摘された15の必須修正項目および LaTeX 構文上のブロッカーはすべて解消され、データ生成コード、正本 CSV、TeX テーブル、およびメイン論文 `iclr2027_conference2.tex` の整合性が完全に確立されました。
結果フォルダ（`results/`）およびテーブル群（`iclr2027/tables/`）の修正作業はこれにて完了とし、安心して次のフェーズ（**Behavioral 考察 → V1 考察 → V2 考察 → V3 考察 → 全体考察**の執筆）へ進むことができます。
